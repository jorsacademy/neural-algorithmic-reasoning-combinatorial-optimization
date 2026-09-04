"""Deterministic training for trace-supervised and terminal-only neural baselines."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from narcopt.dataset import KnapsackDataset
from narcopt.domain import KnapsackSolution, solve_dynamic_programming
from narcopt.models import BellmanReasoner, DirectPolicy, policy_features
from narcopt.reasoning import build_trace_cell_dataset, policy_advice, rollout_reasoner
from narcopt.utils import seed_everything

TrainingMode = Literal["trace_reasoner", "direct_policy"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    value_loss_weight: float = 0.5
    gradient_clip_norm: float = 5.0
    validation_every: int = 2
    patience_checks: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if self.value_loss_weight < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("loss weight must be nonnegative and clipping norm positive")
        if self.validation_every <= 0 or self.patience_checks <= 0:
            raise ValueError("validation cadence and patience must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_loss: float | None
    validation_candidate_relative_gap: float | None
    validation_optimal_hit_rate: float | None
    gradient_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    mode: TrainingMode
    epochs_completed: int
    best_validation_score: float
    parameter_count: int
    history: tuple[EpochRecord, ...]
    config: TrainingConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "epochs_completed": self.epochs_completed,
            "best_validation_score": self.best_validation_score,
            "parameter_count": self.parameter_count,
            "history": [record.to_dict() for record in self.history],
            "config": asdict(self.config),
        }


def _gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
            total += value * value
    return math.sqrt(total)


def _exact_solution(instance_index: int, dataset: KnapsackDataset) -> KnapsackSolution:
    result = solve_dynamic_programming(dataset.instances[instance_index])
    if not isinstance(result, KnapsackSolution):
        raise RuntimeError("unexpected dynamic-programming return type")
    return result


def _candidate_metrics(
    model: BellmanReasoner | DirectPolicy,
    dataset: KnapsackDataset,
) -> tuple[float, float]:
    gaps: list[float] = []
    hits: list[float] = []
    for index, instance in enumerate(dataset.instances):
        optimum = _exact_solution(index, dataset)
        if isinstance(model, BellmanReasoner):
            candidate = rollout_reasoner(model, instance).advice.candidate
        else:
            candidate = policy_advice(model, instance).candidate
        gap = optimum.objective - candidate.objective
        if gap < 0:
            raise RuntimeError("neural candidate exceeds the exact optimum")
        gaps.append(gap / max(1.0, float(optimum.objective)))
        hits.append(float(gap == 0))
    return float(np.mean(gaps)), float(np.mean(hits))


def _trace_validation_loss(model: BellmanReasoner, dataset: KnapsackDataset) -> float:
    cells = build_trace_cell_dataset(dataset, device=model.device)
    model.eval()
    with torch.no_grad():
        logits = model(cells.features)
        feasible_logits = logits[cells.feasible]
        feasible_targets = cells.target_take[cells.feasible]
        classification = F.binary_cross_entropy_with_logits(feasible_logits, feasible_targets)
        predicted = torch.where(
            cells.feasible,
            cells.skip + torch.sigmoid(logits) * (cells.take - cells.skip),
            cells.skip,
        )
        regression = F.mse_loss(predicted, cells.target_value)
        loss = classification + 0.5 * regression
    return float(loss.cpu())


def train_trace_reasoner(
    model: BellmanReasoner,
    training: KnapsackDataset,
    validation: KnapsackDataset,
    *,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    config = config or TrainingConfig()
    seed_everything(config.seed)
    cells = build_trace_cell_dataset(training, device=model.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_score = math.inf
    checks_without_improvement = 0
    history: list[EpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(cells), generator=generator)
        losses: list[float] = []
        last_gradient_norm = 0.0
        for start in range(0, len(cells), config.batch_size):
            indices = permutation[start : start + config.batch_size].to(cells.features.device)
            features = cells.features[indices]
            target_take = cells.target_take[indices]
            target_value = cells.target_value[indices]
            skip = cells.skip[indices]
            take = cells.take[indices]
            feasible = cells.feasible[indices]
            logits = model(features)
            if torch.any(feasible):
                classification = F.binary_cross_entropy_with_logits(
                    logits[feasible], target_take[feasible]
                )
            else:
                classification = torch.zeros((), device=model.device)
            predicted = torch.where(
                feasible,
                skip + torch.sigmoid(logits) * (take - skip),
                skip,
            )
            regression = F.mse_loss(predicted, target_value)
            loss = classification + config.value_loss_weight * regression
            if not torch.isfinite(loss):
                raise RuntimeError("trace training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            last_gradient_norm = _gradient_norm(model)
            if not math.isfinite(last_gradient_norm):
                raise RuntimeError("trace training produced non-finite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_loss: float | None = None
        relative_gap: float | None = None
        hit_rate: float | None = None
        if epoch % config.validation_every == 0 or epoch == config.epochs:
            validation_loss = _trace_validation_loss(model, validation)
            relative_gap, hit_rate = _candidate_metrics(model, validation)
            score = validation_loss + relative_gap
            if score < best_score - 1e-10:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
        history.append(
            EpochRecord(
                epoch=epoch,
                training_loss=float(np.mean(losses)),
                validation_loss=validation_loss,
                validation_candidate_relative_gap=relative_gap,
                validation_optimal_hit_rate=hit_rate,
                gradient_norm=last_gradient_norm,
            )
        )
        if checks_without_improvement >= config.patience_checks:
            break

    model.load_state_dict(best_state, strict=True)
    return TrainingSummary(
        mode="trace_reasoner",
        epochs_completed=len(history),
        best_validation_score=best_score,
        parameter_count=model.parameter_count,
        history=tuple(history),
        config=config,
    )


def _policy_loss(model: DirectPolicy, dataset: KnapsackDataset) -> Tensor:
    losses: list[Tensor] = []
    for index, instance in enumerate(dataset.instances):
        optimum = _exact_solution(index, dataset)
        target = torch.tensor(
            optimum.selection,
            dtype=torch.float32,
            device=model.device,
        )
        logits = model(policy_features(instance, device=model.device))
        weights = torch.tensor(instance.values, dtype=torch.float32, device=model.device)
        weights = weights / torch.mean(weights)
        losses.append(F.binary_cross_entropy_with_logits(logits, target, weight=weights))
    return torch.stack(losses).mean()


def train_direct_policy(
    model: DirectPolicy,
    training: KnapsackDataset,
    validation: KnapsackDataset,
    *,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    config = config or TrainingConfig(epochs=60, batch_size=8)
    seed_everything(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_score = math.inf
    checks_without_improvement = 0
    history: list[EpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(training.instances))
        losses: list[float] = []
        last_gradient_norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for position, instance_index in enumerate(order, start=1):
            instance = training.instances[int(instance_index)]
            optimum = _exact_solution(int(instance_index), training)
            target = torch.tensor(optimum.selection, dtype=torch.float32, device=model.device)
            logits = model(policy_features(instance, device=model.device))
            item_weights = torch.tensor(instance.values, dtype=torch.float32, device=model.device)
            item_weights = item_weights / torch.mean(item_weights)
            loss = F.binary_cross_entropy_with_logits(logits, target, weight=item_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("policy training produced a non-finite loss")
            (loss / config.batch_size).backward()  # type: ignore[no-untyped-call]
            pending += 1
            losses.append(float(loss.detach().cpu()))
            if pending == config.batch_size or position == len(order):
                last_gradient_norm = _gradient_norm(model)
                if not math.isfinite(last_gradient_norm):
                    raise RuntimeError("policy training produced non-finite gradients")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0

        validation_loss: float | None = None
        relative_gap: float | None = None
        hit_rate: float | None = None
        if epoch % config.validation_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                validation_loss = float(_policy_loss(model, validation).cpu())
            relative_gap, hit_rate = _candidate_metrics(model, validation)
            score = validation_loss + relative_gap
            if score < best_score - 1e-10:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
        history.append(
            EpochRecord(
                epoch=epoch,
                training_loss=float(np.mean(losses)),
                validation_loss=validation_loss,
                validation_candidate_relative_gap=relative_gap,
                validation_optimal_hit_rate=hit_rate,
                gradient_norm=last_gradient_norm,
            )
        )
        if checks_without_improvement >= config.patience_checks:
            break

    model.load_state_dict(best_state, strict=True)
    return TrainingSummary(
        mode="direct_policy",
        epochs_completed=len(history),
        best_validation_score=best_score,
        parameter_count=model.parameter_count,
        history=tuple(history),
        config=config,
    )


def train_model(
    model: BellmanReasoner | DirectPolicy,
    training: KnapsackDataset,
    validation: KnapsackDataset,
    *,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    if isinstance(model, BellmanReasoner):
        return train_trace_reasoner(model, training, validation, config=config)
    if isinstance(model, DirectPolicy):
        return train_direct_policy(model, training, validation, config=config)
    raise TypeError("unsupported model type")
