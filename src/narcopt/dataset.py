"""Deterministic knapsack generation, exact labels, and tamper-evident corpora."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from narcopt.domain import (
    KnapsackInstance,
    KnapsackSolution,
    audit_solution,
    solve_dynamic_programming,
)
from narcopt.utils import canonical_json, sha256_json

DATASET_SCHEMA_VERSION = "1.0"
SUPPORTED_REGIMES: tuple[str, ...] = (
    "iid",
    "weakly_correlated",
    "strongly_correlated",
    "inverse_correlated",
    "heavy_tail",
    "clustered",
    "tight_capacity",
    "loose_capacity",
    "value_scale_shift",
)


def generate_instance(
    *,
    item_count: int,
    regime: str = "iid",
    seed: int,
    capacity_ratio: float | None = None,
    instance_id: str | None = None,
) -> KnapsackInstance:
    """Generate one positive-integer 0-1 knapsack instance."""

    if item_count < 2:
        raise ValueError("item_count must be at least two")
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported generation regime: {regime}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)

    if regime == "clustered":
        small = rng.integers(2, 7, size=item_count)
        large = rng.integers(15, 25, size=item_count)
        cluster = rng.integers(0, 2, size=item_count)
        weights = np.where(cluster == 0, small, large)
    else:
        weights = rng.integers(1, 21, size=item_count)

    if regime in {"weakly_correlated", "tight_capacity", "loose_capacity"}:
        values = weights + rng.integers(1, 16, size=item_count)
    elif regime == "strongly_correlated":
        values = weights + 5
    elif regime == "inverse_correlated":
        values = np.maximum(1, 24 - weights + rng.integers(0, 7, size=item_count))
    elif regime == "heavy_tail":
        values = np.clip(np.rint(2.0 + 8.0 * rng.pareto(2.0, size=item_count)), 1, 120)
    elif regime == "clustered":
        values = np.maximum(1, np.rint(1.4 * weights + rng.normal(4.0, 4.0, size=item_count)))
    elif regime == "value_scale_shift":
        values = 3 * rng.integers(1, 31, size=item_count)
    else:
        values = rng.integers(1, 31, size=item_count)

    ratio = capacity_ratio
    if ratio is None:
        if regime == "tight_capacity":
            ratio = 0.25
        elif regime == "loose_capacity":
            ratio = 0.65
        else:
            ratio = 0.42
    if not math.isfinite(ratio) or not 0.1 <= ratio <= 0.9:
        raise ValueError("capacity_ratio must lie in [0.1, 0.9]")
    total_weight = int(np.sum(weights))
    capacity = min(total_weight - 1, max(1, int(round(ratio * total_weight))))
    identifier = instance_id or f"knapsack-{item_count:03d}-{regime}-seed{seed}"
    return KnapsackInstance(
        tuple(int(weight) for weight in weights),
        tuple(int(value) for value in values),
        capacity,
        instance_id=identifier,
        regime=regime,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class KnapsackDataset:
    instances: tuple[KnapsackInstance, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.instances:
            raise ValueError("dataset must contain at least one instance")
        identifiers = [instance.instance_id for instance in self.instances]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset instance identifiers must be unique")

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({instance.regime for instance in self.instances}))

    @property
    def item_counts(self) -> tuple[int, ...]:
        return tuple(sorted({instance.item_count for instance in self.instances}))

    @property
    def fingerprint(self) -> str:
        records = [_record_payload(instance) for instance in self.instances]
        return sha256_json(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "metadata": self.metadata,
                "records": records,
            }
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "record_type": "manifest",
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_count": len(self.instances),
            "fingerprint": self.fingerprint,
            "item_counts": list(self.item_counts),
            "regimes": list(self.regimes),
            "metadata": self.metadata,
        }


def _exact_solution(instance: KnapsackInstance) -> KnapsackSolution:
    solution = solve_dynamic_programming(instance)
    if not isinstance(solution, KnapsackSolution):
        raise RuntimeError("unexpected trace return while labeling dataset")
    return solution


def _record_payload(instance: KnapsackInstance) -> dict[str, object]:
    return {
        "record_type": "instance",
        "instance": instance.to_dict(),
        "exact_solution": _exact_solution(instance).to_dict(),
    }


def collect_dataset(
    *,
    count: int,
    item_counts: tuple[int, ...] = (8, 10, 12, 14),
    regimes: tuple[str, ...] = ("iid", "iid", "weakly_correlated"),
    seed: int = 1000,
) -> KnapsackDataset:
    if count <= 0:
        raise ValueError("count must be positive")
    if not item_counts or any(item_count < 2 for item_count in item_counts):
        raise ValueError("item_counts must contain values of at least two")
    if not regimes or any(regime not in SUPPORTED_REGIMES for regime in regimes):
        raise ValueError("regimes contain an unsupported value")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    instances: list[KnapsackInstance] = []
    for index in range(count):
        item_count = item_counts[index % len(item_counts)]
        regime = regimes[index % len(regimes)]
        observation_seed = seed + 104_729 * (index + 1)
        instances.append(
            generate_instance(
                item_count=item_count,
                regime=regime,
                seed=observation_seed,
                instance_id=f"sample-{index:05d}-{regime}-n{item_count}-seed{observation_seed}",
            )
        )
    return KnapsackDataset(
        tuple(instances),
        metadata={
            "count": count,
            "item_counts": list(item_counts),
            "regimes": list(regimes),
            "seed": seed,
        },
    )


def save_dataset(dataset: KnapsackDataset, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json(dataset.to_manifest())]
    lines.extend(canonical_json(_record_payload(instance)) for instance in dataset.instances)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def load_dataset(path: str | Path) -> KnapsackDataset:
    source = Path(path)
    raw_lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) < 2:
        raise ValueError("dataset must contain a manifest and at least one instance")
    manifest = _json_object(json.loads(raw_lines[0]), name="manifest")
    if manifest.get("record_type") != "manifest":
        raise ValueError("first JSONL record must be a manifest")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema version")
    metadata = _json_object(manifest.get("metadata", {}), name="metadata")

    instances: list[KnapsackInstance] = []
    for line_number, raw_line in enumerate(raw_lines[1:], start=2):
        record = _json_object(json.loads(raw_line), name=f"record on line {line_number}")
        if record.get("record_type") != "instance":
            raise ValueError(f"line {line_number} is not an instance record")
        instance_payload = _json_object(record.get("instance"), name="instance")
        solution_payload = _json_object(record.get("exact_solution"), name="exact_solution")
        instance = KnapsackInstance.from_dict(instance_payload)
        selection = solution_payload.get("selection")
        objective = solution_payload.get("objective")
        if not isinstance(selection, list) or not isinstance(objective, int):
            raise ValueError("stored exact solution has an invalid shape")
        audit = audit_solution(
            instance,
            selection,
            reported_objective=objective,
            verify_optimality=True,
        )
        if not (
            audit.feasible
            and audit.binary
            and audit.reported_objective_consistent
            and audit.optimal is True
        ):
            raise ValueError(f"stored exact solution failed audit on line {line_number}")
        instances.append(instance)

    dataset = KnapsackDataset(tuple(instances), metadata)
    expected_count = manifest.get("record_count")
    expected_fingerprint = manifest.get("fingerprint")
    if expected_count != len(dataset.instances):
        raise ValueError("dataset record count does not match manifest")
    if expected_fingerprint != dataset.fingerprint:
        raise ValueError("dataset fingerprint does not match manifest")
    return dataset
