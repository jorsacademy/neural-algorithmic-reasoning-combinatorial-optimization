"""Trace supervision, autoregressive Bellman rollouts, and neural advice."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from narcopt.dataset import KnapsackDataset
from narcopt.domain import (
    DynamicProgrammingTrace,
    KnapsackInstance,
    KnapsackSolution,
    repair_selection,
    solve_dynamic_programming,
)
from narcopt.models import BellmanReasoner, DirectPolicy, bellman_cell_features, policy_features


@dataclass(frozen=True, slots=True)
class HeuristicAdvice:
    source: str
    preferred_selection: tuple[int, ...]
    scores: tuple[float, ...]
    candidate: KnapsackSolution

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "preferred_selection": list(self.preferred_selection),
            "scores": list(self.scores),
            "candidate": self.candidate.to_dict(),
        }


@dataclass(slots=True)
class BellmanRollout:
    values: np.ndarray
    logits: np.ndarray
    advice: HeuristicAdvice

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.logits.shape != self.values.shape:
            raise ValueError("rollout values and logits must be aligned matrices")


@dataclass(frozen=True, slots=True)
class TraceDiagnostics:
    value_rmse: float
    final_value_absolute_error: float
    final_value_relative_error: float
    take_accuracy: float
    candidate_objective_gap: int
    candidate_relative_gap: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceCellDataset:
    features: Tensor
    target_take: Tensor
    target_value: Tensor
    skip: Tensor
    take: Tensor
    feasible: Tensor

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        if self.features.ndim != 2 or self.features.shape[1] != 7:
            raise ValueError("trace features must have shape [cells, 7]")
        vectors = (self.target_take, self.target_value, self.skip, self.take, self.feasible)
        if any(vector.ndim != 1 or vector.shape[0] != count for vector in vectors):
            raise ValueError("trace supervision vectors must align with features")

    def __len__(self) -> int:
        return self.features.shape[0]


def build_trace_cell_dataset(
    dataset: KnapsackDataset,
    *,
    device: torch.device | str = "cpu",
) -> TraceCellDataset:
    features: list[Tensor] = []
    targets_take: list[Tensor] = []
    targets_value: list[Tensor] = []
    skips: list[Tensor] = []
    takes: list[Tensor] = []
    feasible_masks: list[Tensor] = []
    for instance in dataset.instances:
        result = solve_dynamic_programming(instance, return_trace=True)
        if not isinstance(result, tuple):
            raise RuntimeError("dynamic-programming trace was not returned")
        _solution, trace = result
        scale = float(instance.total_value)
        for item_index, (weight, value) in enumerate(
            zip(instance.weights, instance.values, strict=True), start=1
        ):
            previous = torch.tensor(
                trace.values[item_index - 1] / scale,
                dtype=torch.float32,
                device=device,
            )
            cell_features, skip, take, feasible = bellman_cell_features(
                previous,
                item_weight=weight,
                item_value=value,
                capacity=instance.capacity,
                value_scale=scale,
            )
            features.append(cell_features)
            targets_take.append(
                torch.tensor(trace.take[item_index], dtype=torch.float32, device=device)
            )
            targets_value.append(
                torch.tensor(trace.values[item_index] / scale, dtype=torch.float32, device=device)
            )
            skips.append(skip)
            takes.append(take)
            feasible_masks.append(feasible)
    return TraceCellDataset(
        features=torch.cat(features, dim=0),
        target_take=torch.cat(targets_take, dim=0),
        target_value=torch.cat(targets_value, dim=0),
        skip=torch.cat(skips, dim=0),
        take=torch.cat(takes, dim=0),
        feasible=torch.cat(feasible_masks, dim=0),
    )


def _solution_from_backtracking(
    instance: KnapsackInstance,
    logits: np.ndarray,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    selection = [0] * instance.item_count
    scores = [0.0] * instance.item_count
    remaining = instance.capacity
    for item_index in range(instance.item_count, 0, -1):
        weight = instance.weights[item_index - 1]
        score = float(logits[item_index, remaining])
        scores[item_index - 1] = score
        if weight <= remaining and score > 0.0:
            selection[item_index - 1] = 1
            remaining -= weight
    return tuple(selection), tuple(scores)


def rollout_reasoner(model: BellmanReasoner, instance: KnapsackInstance) -> BellmanRollout:
    model.eval()
    scale = float(instance.total_value)
    previous = torch.zeros(instance.capacity + 1, dtype=torch.float32, device=model.device)
    value_rows = [previous.detach().cpu().numpy() * scale]
    logit_rows = [np.full(instance.capacity + 1, -20.0, dtype=np.float64)]
    with torch.no_grad():
        for weight, value in zip(instance.weights, instance.values, strict=True):
            previous, step_logits = model.step(
                previous,
                item_weight=weight,
                item_value=value,
                capacity=instance.capacity,
                value_scale=scale,
            )
            value_rows.append(previous.detach().cpu().double().numpy() * scale)
            logit_rows.append(step_logits.detach().cpu().double().numpy())
    values = np.stack(value_rows)
    logit_table = np.stack(logit_rows)
    preferred, scores = _solution_from_backtracking(instance, logit_table)
    candidate = repair_selection(instance, scores, preferred)
    return BellmanRollout(
        values=values,
        logits=logit_table,
        advice=HeuristicAdvice("trace_reasoner", preferred, scores, candidate),
    )


def policy_advice(model: DirectPolicy, instance: KnapsackInstance) -> HeuristicAdvice:
    model.eval()
    with torch.no_grad():
        logits = model(policy_features(instance, device=model.device))
    score_values = tuple(float(value) for value in logits.detach().cpu().double().numpy())
    preferred = tuple(1 if score > 0.0 else 0 for score in score_values)
    candidate = repair_selection(instance, score_values, preferred)
    return HeuristicAdvice("direct_policy", preferred, score_values, candidate)


def density_advice(instance: KnapsackInstance) -> HeuristicAdvice:
    scores = tuple(
        float(value) / float(weight)
        for weight, value in zip(instance.weights, instance.values, strict=True)
    )
    preferred = tuple(1 for _ in scores)
    candidate = repair_selection(instance, scores, preferred)
    return HeuristicAdvice("density", preferred, scores, candidate)


def trace_diagnostics(
    rollout: BellmanRollout,
    instance: KnapsackInstance,
    trace: DynamicProgrammingTrace | None = None,
) -> TraceDiagnostics:
    exact_result = solve_dynamic_programming(instance, return_trace=True)
    if not isinstance(exact_result, tuple):
        raise RuntimeError("dynamic-programming trace was not returned")
    optimum, exact_trace = exact_result
    trace = trace or exact_trace
    value_errors = rollout.values - trace.values
    value_rmse = float(np.sqrt(np.mean(value_errors**2)))
    final_error = abs(float(rollout.values[-1, instance.capacity]) - optimum.objective)
    final_relative = final_error / max(1.0, float(optimum.objective))
    predicted_take = rollout.logits[1:] > 0.0
    exact_take = trace.take[1:]
    take_accuracy = float(np.mean(predicted_take == exact_take))
    gap = optimum.objective - rollout.advice.candidate.objective
    return TraceDiagnostics(
        value_rmse=value_rmse,
        final_value_absolute_error=final_error,
        final_value_relative_error=final_relative,
        take_accuracy=take_accuracy,
        candidate_objective_gap=gap,
        candidate_relative_gap=gap / max(1.0, float(optimum.objective)),
    )
