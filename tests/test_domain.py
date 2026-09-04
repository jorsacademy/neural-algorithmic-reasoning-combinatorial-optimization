import numpy as np
import pytest

from narcopt.domain import (
    KnapsackInstance,
    KnapsackSolution,
    audit_solution,
    greedy_density_solution,
    repair_selection,
    solve_brute_force,
    solve_dynamic_programming,
)


def test_known_dynamic_programming_solution_and_trace() -> None:
    instance = KnapsackInstance((2, 3, 4, 5), (3, 4, 5, 8), 8)
    result = solve_dynamic_programming(instance, return_trace=True)
    assert isinstance(result, tuple)
    solution, trace = result
    assert solution.objective == 12
    assert solution.total_weight == 8
    assert solution.selection == (0, 1, 0, 1)
    assert trace.values.shape == (5, 9)
    assert trace.take.shape == trace.values.shape
    assert trace.values[-1, -1] == 12
    for item in range(1, trace.item_count + 1):
        weight = instance.weights[item - 1]
        value = instance.values[item - 1]
        for capacity in range(trace.capacity + 1):
            expected = trace.values[item - 1, capacity]
            if capacity >= weight:
                expected = max(expected, value + trace.values[item - 1, capacity - weight])
            assert trace.values[item, capacity] == expected


def test_dynamic_programming_matches_brute_force() -> None:
    rng = np.random.default_rng(11)
    for index in range(10):
        weights = tuple(int(value) for value in rng.integers(1, 9, size=8))
        values = tuple(int(value) for value in rng.integers(1, 15, size=8))
        instance = KnapsackInstance(weights, values, max(1, sum(weights) // 3), seed=index)
        dynamic = solve_dynamic_programming(instance)
        brute = solve_brute_force(instance)
        assert isinstance(dynamic, KnapsackSolution)
        assert dynamic.objective == brute.objective


def test_audit_and_repairs() -> None:
    instance = KnapsackInstance((4, 3, 2), (7, 5, 4), 5)
    audit = audit_solution(instance, (0, 1, 1), reported_objective=9, verify_optimality=True)
    assert audit.feasible
    assert audit.binary
    assert audit.reported_objective_consistent
    assert audit.optimal is True
    invalid = audit_solution(instance, (2, 0, 0), verify_optimality=True)
    assert not invalid.binary
    assert not invalid.feasible
    overweight = audit_solution(instance, (1, 1, 0))
    assert not overweight.feasible
    inconsistent = audit_solution(instance, (0, 1, 1), reported_objective=8)
    assert not inconsistent.reported_objective_consistent

    repaired = repair_selection(instance, (1.0, 2.0, 3.0), (1, 1, 1))
    assert repaired.total_weight <= instance.capacity
    greedy = greedy_density_solution(instance)
    assert greedy.total_weight <= instance.capacity


def test_invalid_instances_and_limits() -> None:
    with pytest.raises(ValueError):
        KnapsackInstance((), (), 1)
    with pytest.raises(ValueError):
        KnapsackInstance((1,), (1, 2), 1)
    with pytest.raises(ValueError):
        KnapsackInstance((1, -1), (1, 2), 1)
    with pytest.raises(ValueError):
        KnapsackInstance((1, 2), (1, 2), 3)
    instance = KnapsackInstance(tuple([1] * 25), tuple([1] * 25), 10)
    with pytest.raises(ValueError):
        solve_brute_force(instance)
    with pytest.raises(ValueError):
        repair_selection(KnapsackInstance((1, 2), (2, 3), 2), (1.0,))
