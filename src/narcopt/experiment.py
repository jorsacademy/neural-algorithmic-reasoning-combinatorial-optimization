"""Frozen trace-supervision and exact-search transfer protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from narcopt.dataset import KnapsackDataset, collect_dataset
from narcopt.evaluation import (
    ApproximationMetrics,
    EvaluationReport,
    SearchMetrics,
    TraceMetrics,
    evaluate_models,
)
from narcopt.models import (
    BellmanReasoner,
    BellmanReasonerConfig,
    DirectPolicy,
    DirectPolicyConfig,
    save_checkpoint,
)
from narcopt.training import (
    TrainingConfig,
    TrainingSummary,
    train_direct_policy,
    train_trace_reasoner,
)
from narcopt.utils import write_json


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    train_samples: int = 80
    validation_samples: int = 24
    evaluation_samples: int = 20
    train_item_counts: tuple[int, ...] = (8, 10, 12, 14)
    reasoner_hidden_dim: int = 64
    policy_hidden_dim: int = 64
    hidden_layers: int = 2
    reasoner_epochs: int = 30
    policy_epochs: int = 40
    batch_size: int = 2048
    learning_rate: float = 2e-3
    bootstrap_draws: int = 300
    seed: int = 2026

    def __post_init__(self) -> None:
        counts = (
            self.train_samples,
            self.validation_samples,
            self.evaluation_samples,
            self.reasoner_hidden_dim,
            self.policy_hidden_dim,
            self.hidden_layers,
            self.reasoner_epochs,
            self.policy_epochs,
            self.batch_size,
            self.bootstrap_draws,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("research counts and dimensions must be positive")
        if not self.train_item_counts or any(value < 2 for value in self.train_item_counts):
            raise ValueError("train_item_counts must contain values of at least two")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    training: dict[str, dict[str, object]]
    approximation_rows: tuple[ApproximationMetrics, ...]
    search_rows: tuple[SearchMetrics, ...]
    trace_rows: tuple[TraceMetrics, ...]
    scenario_metadata: dict[str, dict[str, object]]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "training": self.training,
            "approximation_rows": [row.to_dict() for row in self.approximation_rows],
            "search_rows": [row.to_dict() for row in self.search_rows],
            "trace_rows": [row.to_dict() for row in self.trace_rows],
            "scenario_metadata": self.scenario_metadata,
            "metadata": self.metadata,
        }


def _training_datasets(config: ResearchConfig) -> tuple[KnapsackDataset, KnapsackDataset]:
    training = collect_dataset(
        count=config.train_samples,
        item_counts=config.train_item_counts,
        regimes=("iid", "iid", "weakly_correlated", "strongly_correlated"),
        seed=config.seed + 1_000,
    )
    validation = collect_dataset(
        count=config.validation_samples,
        item_counts=config.train_item_counts,
        regimes=("iid", "weakly_correlated", "strongly_correlated"),
        seed=config.seed + 2_000,
    )
    return training, validation


def _scenario_specs() -> tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...]:
    return (
        ("interpolation", (12,), ("iid",)),
        ("size_18", (18,), ("iid",)),
        ("size_24", (24,), ("iid",)),
        ("tight_capacity", (18,), ("tight_capacity",)),
        ("loose_capacity", (18,), ("loose_capacity",)),
        ("inverse_correlated", (18,), ("inverse_correlated",)),
        ("heavy_tail", (18,), ("heavy_tail",)),
        ("clustered", (18,), ("clustered",)),
        ("value_scale_shift", (18,), ("value_scale_shift",)),
    )


def _train_models(
    config: ResearchConfig,
    training: KnapsackDataset,
    validation: KnapsackDataset,
) -> tuple[BellmanReasoner, DirectPolicy, TrainingSummary, TrainingSummary]:
    reasoner = BellmanReasoner(
        BellmanReasonerConfig(config.reasoner_hidden_dim, config.hidden_layers)
    )
    policy = DirectPolicy(DirectPolicyConfig(config.policy_hidden_dim, config.hidden_layers))
    common = {
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "validation_every": 2,
        "patience_checks": 8,
    }
    reasoner_summary = train_trace_reasoner(
        reasoner,
        training,
        validation,
        config=TrainingConfig(
            epochs=config.reasoner_epochs,
            seed=config.seed + 10_000,
            **common,
        ),
    )
    policy_summary = train_direct_policy(
        policy,
        training,
        validation,
        config=TrainingConfig(
            epochs=config.policy_epochs,
            seed=config.seed + 20_000,
            **common,
        ),
    )
    return reasoner, policy, reasoner_summary, policy_summary


def run_research_experiment(
    config: ResearchConfig | None = None,
    *,
    checkpoint_directory: str | Path | None = None,
) -> tuple[dict[str, BellmanReasoner | DirectPolicy], ResearchReport]:
    config = config or ResearchConfig()
    training, validation = _training_datasets(config)
    reasoner, policy, reasoner_summary, policy_summary = _train_models(
        config,
        training,
        validation,
    )
    if checkpoint_directory is not None:
        directory = Path(checkpoint_directory)
        directory.mkdir(parents=True, exist_ok=True)
        common_metadata = {
            "train_fingerprint": training.fingerprint,
            "validation_fingerprint": validation.fingerprint,
            "research_config": asdict(config),
        }
        save_checkpoint(
            reasoner,
            directory / "trace-reasoner.safetensors",
            metadata={"training_mode": "trace_reasoner", **common_metadata},
        )
        save_checkpoint(
            policy,
            directory / "direct-policy.safetensors",
            metadata={"training_mode": "direct_policy", **common_metadata},
        )

    approximation_rows: list[ApproximationMetrics] = []
    search_rows: list[SearchMetrics] = []
    trace_rows: list[TraceMetrics] = []
    scenario_metadata: dict[str, dict[str, object]] = {}
    for scenario_index, (name, item_counts, regimes) in enumerate(_scenario_specs()):
        dataset = collect_dataset(
            count=config.evaluation_samples,
            item_counts=item_counts,
            regimes=regimes,
            seed=config.seed + 100_000 + 10_000 * scenario_index,
        )
        evaluation: EvaluationReport = evaluate_models(
            reasoner,
            policy,
            dataset,
            scenario=name,
            bootstrap_seed=config.seed + 200_000 + scenario_index,
            bootstrap_draws=config.bootstrap_draws,
        )
        approximation_rows.extend(evaluation.approximation_rows)
        search_rows.extend(evaluation.search_rows)
        trace_rows.append(evaluation.trace_metrics)
        scenario_metadata[name] = evaluation.metadata

    report = ResearchReport(
        training={
            "trace_reasoner": reasoner_summary.to_dict(),
            "direct_policy": policy_summary.to_dict(),
        },
        approximation_rows=tuple(approximation_rows),
        search_rows=tuple(search_rows),
        trace_rows=tuple(trace_rows),
        scenario_metadata=scenario_metadata,
        metadata={
            "config": asdict(config),
            "train_fingerprint": training.fingerprint,
            "validation_fingerprint": validation.fingerprint,
            "training_item_counts": list(training.item_counts),
            "training_regimes": list(training.regimes),
            "scenario_order": [name for name, _item_counts, _regimes in _scenario_specs()],
            "research_question": (
                "Does intermediate Bellman-trace supervision extrapolate more reliably and "
                "provide more useful exact-search guidance than terminal-only supervision?"
            ),
            "claims_boundary": (
                "This is a small synthetic 0-1 knapsack methodology benchmark. Neural models "
                "do not certify optimality; dynamic programming and branch-and-bound do."
            ),
        },
    )
    return {"trace_reasoner": reasoner, "direct_policy": policy}, report


def save_research_report(report: ResearchReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)
