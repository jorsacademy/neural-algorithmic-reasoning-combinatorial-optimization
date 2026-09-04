"""Exact branch-and-bound whose efficiency, but not correctness, may use neural advice."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from narcopt.domain import (
    KnapsackInstance,
    KnapsackSolution,
    audit_solution,
    greedy_density_solution,
    solve_dynamic_programming,
)
from narcopt.reasoning import HeuristicAdvice, density_advice


@dataclass(frozen=True, slots=True)
class BranchAndBoundResult:
    solution: KnapsackSolution
    advice_source: str
    node_count: int
    pruned_count: int
    incumbent_updates: int
    initial_incumbent_objective: int
    runtime_seconds: float
    verified_against_dynamic_programming: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["solution"] = self.solution.to_dict()
        return payload


def _fractional_upper_bound(
    instance: KnapsackInstance,
    remaining_indices: tuple[int, ...],
    current_weight: int,
    current_value: int,
) -> float:
    capacity_left = instance.capacity - current_weight
    if capacity_left < 0:
        return -math.inf
    upper = float(current_value)
    order = sorted(
        remaining_indices,
        key=lambda index: (
            instance.values[index] / instance.weights[index],
            instance.values[index],
            -instance.weights[index],
            -index,
        ),
        reverse=True,
    )
    for index in order:
        weight = instance.weights[index]
        value = instance.values[index]
        if weight <= capacity_left:
            upper += value
            capacity_left -= weight
        else:
            upper += value * (capacity_left / weight)
            break
    return upper


def _branch_order(instance: KnapsackInstance, advice: HeuristicAdvice) -> tuple[int, ...]:
    if len(advice.scores) != instance.item_count:
        raise ValueError("advice score vector does not match item count")
    if advice.source == "density":
        return tuple(
            sorted(
                range(instance.item_count),
                key=lambda index: (
                    instance.values[index] / instance.weights[index],
                    instance.values[index],
                    -instance.weights[index],
                    -index,
                ),
                reverse=True,
            )
        )
    return tuple(
        sorted(
            range(instance.item_count),
            key=lambda index: (
                abs(float(advice.scores[index])) * instance.values[index],
                instance.values[index] / instance.weights[index],
                instance.values[index],
                -index,
            ),
            reverse=True,
        )
    )


def exact_branch_and_bound(
    instance: KnapsackInstance,
    *,
    advice: HeuristicAdvice | None = None,
    verify: bool = True,
) -> BranchAndBoundResult:
    """Solve exactly while using advice only for incumbent and traversal order."""

    advice = advice or density_advice(instance)
    advice_audit = audit_solution(
        instance,
        advice.candidate.selection,
        reported_objective=advice.candidate.objective,
    )
    if not advice_audit.feasible or not advice_audit.reported_objective_consistent:
        raise ValueError("search advice must contain a feasible, internally consistent candidate")
    density_candidate = greedy_density_solution(instance)
    incumbent = (
        advice.candidate
        if advice.candidate.objective >= density_candidate.objective
        else density_candidate
    )
    initial_objective = incumbent.objective
    order = _branch_order(instance, advice)
    preferred = incumbent.selection
    stack: list[tuple[int, int, int, tuple[int, ...]]] = [(0, 0, 0, (0,) * instance.item_count)]
    nodes = 0
    pruned = 0
    updates = 0
    start = time.perf_counter()

    while stack:
        depth, current_weight, current_value, selection = stack.pop()
        nodes += 1
        remaining = order[depth:]
        upper = _fractional_upper_bound(
            instance,
            remaining,
            current_weight,
            current_value,
        )
        if math.floor(upper + 1e-10) <= incumbent.objective:
            pruned += 1
            continue
        if depth == instance.item_count:
            candidate = KnapsackSolution(selection, current_value, current_weight)
            if candidate.objective > incumbent.objective:
                incumbent = candidate
                updates += 1
            continue

        item_index = order[depth]
        weight = instance.weights[item_index]
        value = instance.values[item_index]
        branches = (preferred[item_index], 1 - preferred[item_index])
        for chosen in reversed(branches):
            next_weight = current_weight + chosen * weight
            if next_weight > instance.capacity:
                continue
            next_selection = list(selection)
            next_selection[item_index] = chosen
            stack.append(
                (
                    depth + 1,
                    next_weight,
                    current_value + chosen * value,
                    tuple(next_selection),
                )
            )

    runtime = time.perf_counter() - start
    verified = False
    if verify:
        optimum = solve_dynamic_programming(instance)
        if not isinstance(optimum, KnapsackSolution):
            raise RuntimeError("unexpected dynamic-programming return type")
        if incumbent.objective != optimum.objective:
            raise RuntimeError("branch-and-bound disagrees with dynamic programming")
        audit = audit_solution(
            instance,
            incumbent.selection,
            reported_objective=incumbent.objective,
            verify_optimality=True,
        )
        if not (audit.feasible and audit.optimal is True and audit.reported_objective_consistent):
            raise RuntimeError("branch-and-bound solution failed independent audit")
        verified = True
    return BranchAndBoundResult(
        solution=incumbent,
        advice_source=advice.source,
        node_count=nodes,
        pruned_count=pruned,
        incumbent_updates=updates,
        initial_incumbent_objective=initial_objective,
        runtime_seconds=runtime,
        verified_against_dynamic_programming=verified,
    )
