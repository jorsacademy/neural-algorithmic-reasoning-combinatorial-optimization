from pathlib import Path

import numpy as np
import pytest
import torch

from narcopt.domain import KnapsackInstance
from narcopt.models import (
    BellmanReasoner,
    BellmanReasonerConfig,
    DirectPolicy,
    DirectPolicyConfig,
    bellman_cell_features,
    load_checkpoint,
    policy_features,
    save_checkpoint,
)
from narcopt.reasoning import (
    build_trace_cell_dataset,
    density_advice,
    policy_advice,
    rollout_reasoner,
    trace_diagnostics,
)


def test_feature_shapes_and_reasoner_step(tiny_train) -> None:
    instance = tiny_train.instances[0]
    previous = torch.zeros(instance.capacity + 1)
    features, skip, take, feasible = bellman_cell_features(
        previous,
        item_weight=instance.weights[0],
        item_value=instance.values[0],
        capacity=instance.capacity,
        value_scale=float(instance.total_value),
    )
    assert features.shape == (instance.capacity + 1, 7)
    assert skip.shape == take.shape == feasible.shape
    model = BellmanReasoner(BellmanReasonerConfig(8, 1))
    next_values, logits = model.step(
        previous,
        item_weight=instance.weights[0],
        item_value=instance.values[0],
        capacity=instance.capacity,
        value_scale=float(instance.total_value),
    )
    assert next_values.shape == logits.shape == previous.shape
    assert torch.all(next_values[~feasible] == 0)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3))


def test_rollout_policy_and_diagnostics_are_feasible(tiny_train) -> None:
    instance = tiny_train.instances[1]
    reasoner = BellmanReasoner(BellmanReasonerConfig(8, 1))
    policy = DirectPolicy(DirectPolicyConfig(8, 1))
    rollout = rollout_reasoner(reasoner, instance)
    advice = policy_advice(policy, instance)
    assert rollout.advice.candidate.total_weight <= instance.capacity
    assert advice.candidate.total_weight <= instance.capacity
    assert density_advice(instance).candidate.total_weight <= instance.capacity
    diagnostics = trace_diagnostics(rollout, instance)
    assert diagnostics.value_rmse >= 0.0
    assert 0.0 <= diagnostics.take_accuracy <= 1.0
    assert diagnostics.candidate_objective_gap >= 0
    features = policy_features(instance)
    assert features.shape == (instance.item_count, 5)


def test_trace_cell_dataset_alignment(tiny_train) -> None:
    cells = build_trace_cell_dataset(tiny_train)
    assert len(cells) > 0
    assert cells.features.shape[0] == cells.target_take.shape[0]
    assert torch.any(cells.feasible)


def test_safe_checkpoint_round_trip(tmp_path: Path) -> None:
    for model in (
        BellmanReasoner(BellmanReasonerConfig(8, 1)),
        DirectPolicy(DirectPolicyConfig(8, 1)),
    ):
        path = tmp_path / f"{type(model).__name__}.safetensors"
        save_checkpoint(model, path, metadata={"purpose": "test"})
        loaded, metadata = load_checkpoint(path)
        assert type(loaded) is type(model)
        assert metadata == {"purpose": "test"}
        for left, right in zip(
            model.state_dict().values(), loaded.state_dict().values(), strict=True
        ):
            assert torch.equal(left, right)


def test_invalid_feature_inputs() -> None:
    instance = KnapsackInstance((2, 3), (4, 5), 3)
    with pytest.raises(ValueError):
        bellman_cell_features(
            torch.zeros(2), item_weight=2, item_value=4, capacity=3, value_scale=9.0
        )
    policy = DirectPolicy(DirectPolicyConfig(8, 1))
    with pytest.raises(ValueError):
        policy(torch.zeros(0, 5))
    with pytest.raises(ValueError):
        policy(torch.zeros(2, 4))
    values = policy_features(instance).numpy()
    assert np.all(np.isfinite(values))
