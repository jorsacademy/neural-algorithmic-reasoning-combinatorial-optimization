"""Exact 0-1 knapsack domain, dynamic-programming traces, and audits."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class KnapsackInstance:
    """A positive-integer 0-1 knapsack instance."""

    weights: tuple[int, ...]
    values: tuple[int, ...]
    capacity: int
    instance_id: str = "instance"
    regime: str = "unspecified"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.weights or len(self.weights) != len(self.values):
            raise ValueError("weights and values must be aligned nonempty vectors")
        if any(isinstance(weight, bool) or weight <= 0 for weight in self.weights):
            raise ValueError("weights must be positive integers")
        if any(isinstance(value, bool) or value <= 0 for value in self.values):
            raise ValueError("values must be positive integers")
        if isinstance(self.capacity, bool) or self.capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if self.capacity >= sum(self.weights):
            raise ValueError("capacity must leave at least one item potentially excluded")
        if not self.instance_id:
            raise ValueError("instance_id must be nonempty")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def item_count(self) -> int:
        return len(self.weights)

    @property
    def total_weight(self) -> int:
        return sum(self.weights)

    @property
    def total_value(self) -> int:
        return sum(self.values)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": list(self.weights),
            "values": list(self.values),
            "capacity": self.capacity,
            "instance_id": self.instance_id,
            "regime": self.regime,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> KnapsackInstance:
        weights = payload.get("weights")
        values = payload.get("values")
        if not isinstance(weights, list) or not isinstance(values, list):
            raise ValueError("instance weights and values must be JSON arrays")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in weights):
            raise ValueError("instance weights must contain integers")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValueError("instance values must contain integers")
        capacity = payload.get("capacity")
        seed = payload.get("seed", 0)
        instance_id = payload.get("instance_id", "instance")
        regime = payload.get("regime", "unspecified")
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError("instance capacity must be an integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("instance seed must be an integer")
        if not isinstance(instance_id, str) or not isinstance(regime, str):
            raise ValueError("instance identifiers must be strings")
        return cls(
            tuple(weights),
            tuple(values),
            capacity,
            instance_id=instance_id,
            regime=regime,
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class KnapsackSolution:
    selection: tuple[int, ...]
    objective: int
    total_weight: int

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(index for index, chosen in enumerate(self.selection) if chosen)

    def to_dict(self) -> dict[str, object]:
        return {
            "selection": list(self.selection),
            "selected_indices": list(self.selected_indices),
            "objective": self.objective,
            "total_weight": self.total_weight,
        }


@dataclass(frozen=True, slots=True)
class SolutionAudit:
    feasible: bool
    binary: bool
    total_weight: int
    objective: int
    reported_objective_consistent: bool
    optimal: bool | None
    optimality_gap: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DynamicProgrammingTrace:
    """Exact Bellman value and deterministic take/skip tables."""

    values: np.ndarray
    take: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.take.shape != self.values.shape:
            raise ValueError("trace values and decisions must be aligned matrices")
        if self.values.dtype.kind not in {"i", "u"}:
            raise ValueError("trace values must be integer-valued")
        if self.take.dtype != np.bool_:
            raise ValueError("trace decisions must be Boolean")

    @property
    def item_count(self) -> int:
        return self.values.shape[0] - 1

    @property
    def capacity(self) -> int:
        return self.values.shape[1] - 1


def _solution_from_selection(
    instance: KnapsackInstance, selection: Sequence[int]
) -> KnapsackSolution:
    if len(selection) != instance.item_count:
        raise ValueError("selection length does not match item count")
    normalized: list[int] = []
    for chosen in selection:
        if isinstance(chosen, bool):
            normalized.append(int(chosen))
        elif isinstance(chosen, (int, np.integer)) and int(chosen) in {0, 1}:
            normalized.append(int(chosen))
        else:
            raise ValueError("selection entries must be binary")
    total_weight = sum(
        weight * chosen
        for weight, chosen in zip(instance.weights, normalized, strict=True)
    )
    objective = sum(
        value * chosen
        for value, chosen in zip(instance.values, normalized, strict=True)
    )
    return KnapsackSolution(tuple(normalized), objective, total_weight)


def solve_dynamic_programming(
    instance: KnapsackInstance,
    *,
    return_trace: bool = False,
) -> KnapsackSolution | tuple[KnapsackSolution, DynamicProgrammingTrace]:
    """Solve 0-1 knapsack exactly with deterministic skip-on-tie backtracking."""

    values = np.zeros((instance.item_count + 1, instance.capacity + 1), dtype=np.int64)
    take = np.zeros_like(values, dtype=np.bool_)
    for item_index, (weight, value) in enumerate(
        zip(instance.weights, instance.values, strict=True), start=1
    ):
        previous = values[item_index - 1]
        current = values[item_index]
        current[:] = previous
        for capacity in range(weight, instance.capacity + 1):
            take_value = value + int(previous[capacity - weight])
            skip_value = int(previous[capacity])
            if take_value > skip_value:
                current[capacity] = take_value
                take[item_index, capacity] = True

    selection = [0] * instance.item_count
    remaining = instance.capacity
    for item_index in range(instance.item_count, 0, -1):
        if take[item_index, remaining]:
            selection[item_index - 1] = 1
            remaining -= instance.weights[item_index - 1]
    solution = _solution_from_selection(instance, selection)
    if solution.objective != int(values[-1, instance.capacity]):
        raise RuntimeError("dynamic-programming backtracking is inconsistent with the value table")
    if return_trace:
        return solution, DynamicProgrammingTrace(values=values, take=take)
    return solution


def solve_brute_force(
    instance: KnapsackInstance,
    *,
    maximum_items: int = 24,
) -> KnapsackSolution:
    """Enumerate all subsets for independent verification of small instances."""

    if instance.item_count > maximum_items:
        raise ValueError("instance exceeds brute-force verification limit")
    best = KnapsackSolution((0,) * instance.item_count, 0, 0)
    for selection in itertools.product((0, 1), repeat=instance.item_count):
        candidate = _solution_from_selection(instance, selection)
        if candidate.total_weight > instance.capacity:
            continue
        if candidate.objective > best.objective:
            best = candidate
        elif candidate.objective == best.objective and candidate.selection < best.selection:
            best = candidate
    return best


def audit_solution(
    instance: KnapsackInstance,
    selection: Sequence[int],
    *,
    reported_objective: int | None = None,
    verify_optimality: bool = False,
) -> SolutionAudit:
    binary = len(selection) == instance.item_count and all(
        isinstance(chosen, (bool, int, np.integer)) and int(chosen) in {0, 1}
        for chosen in selection
    )
    if not binary:
        return SolutionAudit(
            feasible=False,
            binary=False,
            total_weight=-1,
            objective=-1,
            reported_objective_consistent=False,
            optimal=False if verify_optimality else None,
            optimality_gap=None,
        )
    solution = _solution_from_selection(instance, selection)
    feasible = solution.total_weight <= instance.capacity
    consistent = reported_objective is None or reported_objective == solution.objective
    optimal: bool | None = None
    gap: int | None = None
    if verify_optimality:
        optimum = solve_dynamic_programming(instance)
        if not isinstance(optimum, KnapsackSolution):
            raise RuntimeError("unexpected dynamic-programming return type")
        gap = optimum.objective - solution.objective
        optimal = feasible and gap == 0
    return SolutionAudit(
        feasible=feasible,
        binary=True,
        total_weight=solution.total_weight,
        objective=solution.objective,
        reported_objective_consistent=consistent,
        optimal=optimal,
        optimality_gap=gap,
    )


def greedy_density_solution(instance: KnapsackInstance) -> KnapsackSolution:
    order = sorted(
        range(instance.item_count),
        key=lambda index: (
            instance.values[index] / instance.weights[index],
            instance.values[index],
            -instance.weights[index],
            -index,
        ),
        reverse=True,
    )
    selection = [0] * instance.item_count
    remaining = instance.capacity
    for index in order:
        if instance.weights[index] <= remaining:
            selection[index] = 1
            remaining -= instance.weights[index]
    return _solution_from_selection(instance, selection)


def repair_selection(
    instance: KnapsackInstance,
    scores: Sequence[float],
    preferred: Sequence[int] | None = None,
) -> KnapsackSolution:
    """Construct a feasible candidate by score-prioritized insertion."""

    if len(scores) != instance.item_count:
        raise ValueError("score vector length does not match item count")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    if preferred is not None and len(preferred) != instance.item_count:
        raise ValueError("preferred selection length does not match item count")
    preferred_values = (
        [int(value) for value in preferred]
        if preferred is not None
        else [1 if score >= 0.0 else 0 for score in scores]
    )
    order = sorted(
        range(instance.item_count),
        key=lambda index: (
            preferred_values[index],
            float(scores[index]),
            instance.values[index] / instance.weights[index],
            instance.values[index],
            -index,
        ),
        reverse=True,
    )
    selection = [0] * instance.item_count
    remaining = instance.capacity
    for index in order:
        if preferred_values[index] and instance.weights[index] <= remaining:
            selection[index] = 1
            remaining -= instance.weights[index]
    for index in order:
        if not selection[index] and instance.weights[index] <= remaining:
            selection[index] = 1
            remaining -= instance.weights[index]
    return _solution_from_selection(instance, selection)
