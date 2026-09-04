import math

from narcopt.dataset import generate_instance
from narcopt.domain import KnapsackSolution, solve_dynamic_programming
from narcopt.models import BellmanReasoner, BellmanReasonerConfig, DirectPolicy, DirectPolicyConfig
from narcopt.reasoning import density_advice, policy_advice, rollout_reasoner
from narcopt.search import exact_branch_and_bound
from narcopt.training import TrainingConfig, train_direct_policy, train_trace_reasoner


def test_branch_and_bound_is_exact_under_all_advice() -> None:
    reasoner = BellmanReasoner(BellmanReasonerConfig(8, 1))
    policy = DirectPolicy(DirectPolicyConfig(8, 1))
    for seed, regime in enumerate(("iid", "inverse_correlated", "heavy_tail"), start=1):
        instance = generate_instance(item_count=10, regime=regime, seed=seed)
        optimum = solve_dynamic_programming(instance)
        assert isinstance(optimum, KnapsackSolution)
        advices = (
            density_advice(instance),
            policy_advice(policy, instance),
            rollout_reasoner(reasoner, instance).advice,
        )
        for advice in advices:
            result = exact_branch_and_bound(instance, advice=advice)
            assert result.solution.objective == optimum.objective
            assert result.verified_against_dynamic_programming
            assert result.node_count > 0
            assert result.pruned_count > 0


def test_training_smoke(tiny_train, tiny_validation) -> None:
    reasoner = BellmanReasoner(BellmanReasonerConfig(12, 1))
    reasoner_summary = train_trace_reasoner(
        reasoner,
        tiny_train,
        tiny_validation,
        config=TrainingConfig(
            epochs=2,
            batch_size=64,
            validation_every=1,
            patience_checks=3,
            seed=3,
        ),
    )
    assert reasoner_summary.mode == "trace_reasoner"
    assert reasoner_summary.epochs_completed == 2
    assert math.isfinite(reasoner_summary.best_validation_score)

    policy = DirectPolicy(DirectPolicyConfig(12, 1))
    policy_summary = train_direct_policy(
        policy,
        tiny_train,
        tiny_validation,
        config=TrainingConfig(
            epochs=2,
            batch_size=2,
            validation_every=1,
            patience_checks=3,
            seed=4,
        ),
    )
    assert policy_summary.mode == "direct_policy"
    assert policy_summary.epochs_completed == 2
    assert math.isfinite(policy_summary.best_validation_score)
