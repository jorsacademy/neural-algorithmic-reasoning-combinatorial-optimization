"""Approximation, trace-fidelity, and exact-search evaluation."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

from narcopt.dataset import KnapsackDataset
from narcopt.domain import KnapsackSolution, audit_solution, solve_dynamic_programming
from narcopt.models import BellmanReasoner, DirectPolicy
from narcopt.reasoning import (
    HeuristicAdvice,
    density_advice,
    policy_advice,
    rollout_reasoner,
    trace_diagnostics,
)
from narcopt.search import BranchAndBoundResult, exact_branch_and_bound
from narcopt.utils import write_json


@dataclass(frozen=True, slots=True)
class ApproximationMetrics:
    scenario: str
    method: str
    instance_count: int
    mean_objective_gap: float
    mean_relative_gap: float
    median_relative_gap: float
    p90_relative_gap: float
    optimal_hit_rate: float
    feasible_rate: float
    mean_weight_utilization: float
    mean_relative_gap_ci_low: float
    mean_relative_gap_ci_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchMetrics:
    scenario: str
    guidance: str
    instance_count: int
    mean_node_count: float
    median_node_count: float
    p90_node_count: float
    mean_pruned_count: float
    mean_runtime_seconds: float
    exact_solution_rate: float
    mean_initial_relative_gap: float
    mean_node_reduction_vs_density: float
    mean_node_reduction_ci_low: float
    mean_node_reduction_ci_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    scenario: str
    instance_count: int
    mean_value_rmse: float
    mean_final_value_relative_error: float
    mean_take_accuracy: float
    mean_candidate_relative_gap: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    approximation_rows: tuple[ApproximationMetrics, ...]
    search_rows: tuple[SearchMetrics, ...]
    trace_metrics: TraceMetrics
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "approximation_rows": [row.to_dict() for row in self.approximation_rows],
            "search_rows": [row.to_dict() for row in self.search_rows],
            "trace_metrics": self.trace_metrics.to_dict(),
            "metadata": self.metadata,
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def _bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    if not values or draws <= 0:
        raise ValueError("bootstrap requires values and a positive draw count")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, data.size, size=data.size)
        means[draw] = float(np.mean(data[indices]))
    return (
        float(np.percentile(means, 2.5, method="linear")),
        float(np.percentile(means, 97.5, method="linear")),
    )


def _optimum(instance_index: int, dataset: KnapsackDataset) -> KnapsackSolution:
    solution = solve_dynamic_programming(dataset.instances[instance_index])
    if not isinstance(solution, KnapsackSolution):
        raise RuntimeError("unexpected dynamic-programming return type")
    return solution


def _approximation_row(
    *,
    scenario: str,
    method: str,
    advices: list[HeuristicAdvice],
    optima: list[KnapsackSolution],
    dataset: KnapsackDataset,
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> ApproximationMetrics:
    gaps: list[float] = []
    relative_gaps: list[float] = []
    feasible: list[float] = []
    utilization: list[float] = []
    hits: list[float] = []
    for instance, advice, optimum in zip(dataset.instances, advices, optima, strict=True):
        audit = audit_solution(
            instance,
            advice.candidate.selection,
            reported_objective=advice.candidate.objective,
        )
        gap = optimum.objective - advice.candidate.objective
        if gap < 0:
            raise RuntimeError("candidate objective exceeds the exact optimum")
        gaps.append(float(gap))
        relative_gaps.append(gap / max(1.0, float(optimum.objective)))
        feasible.append(float(audit.feasible and audit.reported_objective_consistent))
        utilization.append(advice.candidate.total_weight / instance.capacity)
        hits.append(float(gap == 0 and audit.feasible))
    ci_low, ci_high = _bootstrap_mean_interval(
        relative_gaps,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
    )
    return ApproximationMetrics(
        scenario=scenario,
        method=method,
        instance_count=len(advices),
        mean_objective_gap=float(np.mean(gaps)),
        mean_relative_gap=float(np.mean(relative_gaps)),
        median_relative_gap=float(median(relative_gaps)),
        p90_relative_gap=_percentile(relative_gaps, 90.0),
        optimal_hit_rate=float(np.mean(hits)),
        feasible_rate=float(np.mean(feasible)),
        mean_weight_utilization=float(np.mean(utilization)),
        mean_relative_gap_ci_low=ci_low,
        mean_relative_gap_ci_high=ci_high,
    )


def _search_row(
    *,
    scenario: str,
    guidance: str,
    results: list[BranchAndBoundResult],
    optima: list[KnapsackSolution],
    baseline_nodes: list[float],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> SearchMetrics:
    nodes = [float(result.node_count) for result in results]
    pruned = [float(result.pruned_count) for result in results]
    runtimes = [result.runtime_seconds for result in results]
    exact = [float(result.verified_against_dynamic_programming) for result in results]
    initial_gaps = [
        (optimum.objective - result.initial_incumbent_objective)
        / max(1.0, float(optimum.objective))
        for result, optimum in zip(results, optima, strict=True)
    ]
    reductions = [
        (baseline - node_count) / max(1.0, baseline)
        for baseline, node_count in zip(baseline_nodes, nodes, strict=True)
    ]
    ci_low, ci_high = _bootstrap_mean_interval(
        reductions,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
    )
    return SearchMetrics(
        scenario=scenario,
        guidance=guidance,
        instance_count=len(results),
        mean_node_count=float(np.mean(nodes)),
        median_node_count=float(median(nodes)),
        p90_node_count=_percentile(nodes, 90.0),
        mean_pruned_count=float(np.mean(pruned)),
        mean_runtime_seconds=float(np.mean(runtimes)),
        exact_solution_rate=float(np.mean(exact)),
        mean_initial_relative_gap=float(np.mean(initial_gaps)),
        mean_node_reduction_vs_density=float(np.mean(reductions)),
        mean_node_reduction_ci_low=ci_low,
        mean_node_reduction_ci_high=ci_high,
    )


def evaluate_models(
    reasoner: BellmanReasoner,
    policy: DirectPolicy,
    dataset: KnapsackDataset,
    *,
    scenario: str,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 500,
) -> EvaluationReport:
    if not scenario:
        raise ValueError("scenario must be nonempty")
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")

    optima: list[KnapsackSolution] = []
    density_advices: list[HeuristicAdvice] = []
    policy_advices: list[HeuristicAdvice] = []
    reasoner_advices: list[HeuristicAdvice] = []
    density_search: list[BranchAndBoundResult] = []
    policy_search: list[BranchAndBoundResult] = []
    reasoner_search: list[BranchAndBoundResult] = []
    trace_value_rmse: list[float] = []
    trace_final_error: list[float] = []
    trace_accuracy: list[float] = []
    trace_candidate_gap: list[float] = []

    for index, instance in enumerate(dataset.instances):
        optimum = _optimum(index, dataset)
        density = density_advice(instance)
        policy_result = policy_advice(policy, instance)
        rollout = rollout_reasoner(reasoner, instance)
        diagnostics = trace_diagnostics(rollout, instance)

        optima.append(optimum)
        density_advices.append(density)
        policy_advices.append(policy_result)
        reasoner_advices.append(rollout.advice)
        density_search.append(exact_branch_and_bound(instance, advice=density, verify=True))
        policy_search.append(exact_branch_and_bound(instance, advice=policy_result, verify=True))
        reasoner_search.append(exact_branch_and_bound(instance, advice=rollout.advice, verify=True))
        trace_value_rmse.append(diagnostics.value_rmse)
        trace_final_error.append(diagnostics.final_value_relative_error)
        trace_accuracy.append(diagnostics.take_accuracy)
        trace_candidate_gap.append(diagnostics.candidate_relative_gap)

    approximation_rows = (
        _approximation_row(
            scenario=scenario,
            method="density_greedy",
            advices=density_advices,
            optima=optima,
            dataset=dataset,
            bootstrap_seed=bootstrap_seed,
            bootstrap_draws=bootstrap_draws,
        ),
        _approximation_row(
            scenario=scenario,
            method="direct_policy",
            advices=policy_advices,
            optima=optima,
            dataset=dataset,
            bootstrap_seed=bootstrap_seed + 1,
            bootstrap_draws=bootstrap_draws,
        ),
        _approximation_row(
            scenario=scenario,
            method="trace_reasoner",
            advices=reasoner_advices,
            optima=optima,
            dataset=dataset,
            bootstrap_seed=bootstrap_seed + 2,
            bootstrap_draws=bootstrap_draws,
        ),
    )
    baseline_nodes = [float(result.node_count) for result in density_search]
    search_rows = (
        _search_row(
            scenario=scenario,
            guidance="density",
            results=density_search,
            optima=optima,
            baseline_nodes=baseline_nodes,
            bootstrap_seed=bootstrap_seed + 10,
            bootstrap_draws=bootstrap_draws,
        ),
        _search_row(
            scenario=scenario,
            guidance="direct_policy",
            results=policy_search,
            optima=optima,
            baseline_nodes=baseline_nodes,
            bootstrap_seed=bootstrap_seed + 11,
            bootstrap_draws=bootstrap_draws,
        ),
        _search_row(
            scenario=scenario,
            guidance="trace_reasoner",
            results=reasoner_search,
            optima=optima,
            baseline_nodes=baseline_nodes,
            bootstrap_seed=bootstrap_seed + 12,
            bootstrap_draws=bootstrap_draws,
        ),
    )
    trace_metrics = TraceMetrics(
        scenario=scenario,
        instance_count=len(dataset.instances),
        mean_value_rmse=float(np.mean(trace_value_rmse)),
        mean_final_value_relative_error=float(np.mean(trace_final_error)),
        mean_take_accuracy=float(np.mean(trace_accuracy)),
        mean_candidate_relative_gap=float(np.mean(trace_candidate_gap)),
    )
    return EvaluationReport(
        approximation_rows=approximation_rows,
        search_rows=search_rows,
        trace_metrics=trace_metrics,
        metadata={
            "scenario": scenario,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_regimes": list(dataset.regimes),
            "dataset_item_counts": list(dataset.item_counts),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "all_exact_search_results_verified": all(
                result.verified_against_dynamic_programming
                for result in density_search + policy_search + reasoner_search
            ),
            "claims_boundary": (
                "Small synthetic 0-1 knapsack benchmark; neural outputs guide but never "
                "certify exact search."
            ),
        },
    )


def save_report_json(report: EvaluationReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)


def save_report_csv(report: EvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for row in report.approximation_rows:
        payload = row.to_dict()
        payload["row_type"] = "approximation"
        rows.append(payload)
    for row in report.search_rows:
        payload = row.to_dict()
        payload["row_type"] = "search"
        rows.append(payload)
    trace_payload = report.trace_metrics.to_dict()
    trace_payload["row_type"] = "trace"
    rows.append(trace_payload)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
