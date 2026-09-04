"""Size-equivariant Bellman processor, direct policy, and safe checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from narcopt.domain import KnapsackInstance

CHECKPOINT_SCHEMA_VERSION = "1.0"
FEATURE_SCHEMA_VERSION = "knapsack-trace-v1"


@dataclass(frozen=True, slots=True)
class BellmanReasonerConfig:
    hidden_dim: int = 64
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("reasoner hidden dimensions must be positive")


@dataclass(frozen=True, slots=True)
class DirectPolicyConfig:
    hidden_dim: int = 64
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("policy hidden dimensions must be positive")


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        modules.extend([nn.Linear(width, hidden_dim), nn.SiLU()])
        width = hidden_dim
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


class BellmanReasoner(nn.Module):
    """Learn the local max operator used by the knapsack Bellman recurrence."""

    network: nn.Sequential

    def __init__(self, config: BellmanReasonerConfig | None = None) -> None:
        super().__init__()
        self.config = config or BellmanReasonerConfig()
        self.network = _mlp(7, self.config.hidden_dim, self.config.hidden_layers, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 7:
            raise ValueError("Bellman features must have shape [cells, 7]")
        logits = cast(Tensor, self.network(features)).squeeze(-1)
        if not torch.all(torch.isfinite(logits)):
            raise RuntimeError("Bellman reasoner produced non-finite logits")
        return logits

    def step(
        self,
        previous: Tensor,
        *,
        item_weight: int,
        item_value: int,
        capacity: int,
        value_scale: float,
    ) -> tuple[Tensor, Tensor]:
        features, skip, take, feasible = bellman_cell_features(
            previous,
            item_weight=item_weight,
            item_value=item_value,
            capacity=capacity,
            value_scale=value_scale,
        )
        raw_logits = self(features)
        logits = torch.where(feasible, raw_logits, torch.full_like(raw_logits, -20.0))
        gate = torch.sigmoid(logits)
        next_values = torch.where(feasible, skip + gate * (take - skip), skip)
        return next_values, logits


class DirectPolicy(nn.Module):
    """Permutation-equivariant terminal-only item-selection policy."""

    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, config: DirectPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectPolicyConfig()
        self.encoder = _mlp(
            5,
            self.config.hidden_dim,
            self.config.hidden_layers,
            self.config.hidden_dim,
        )
        self.decoder = _mlp(
            3 * self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.hidden_layers,
            1,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, item_features: Tensor) -> Tensor:
        if item_features.ndim != 2 or item_features.shape[1] != 5:
            raise ValueError("policy features must have shape [items, 5]")
        if item_features.shape[0] == 0:
            raise ValueError("policy requires at least one item")
        local = cast(Tensor, self.encoder(item_features))
        mean_pool = torch.mean(local, dim=0, keepdim=True).expand_as(local)
        max_pool = torch.max(local, dim=0, keepdim=True).values.expand_as(local)
        logits = cast(
            Tensor,
            self.decoder(torch.cat([local, mean_pool, max_pool], dim=1)),
        ).squeeze(-1)
        if not torch.all(torch.isfinite(logits)):
            raise RuntimeError("direct policy produced non-finite logits")
        return logits


def bellman_cell_features(
    previous: Tensor,
    *,
    item_weight: int,
    item_value: int,
    capacity: int,
    value_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if previous.ndim != 1 or previous.shape[0] != capacity + 1:
        raise ValueError("previous Bellman row has the wrong shape")
    if item_weight <= 0 or item_value <= 0 or capacity <= 0:
        raise ValueError("item and capacity parameters must be positive")
    if not np.isfinite(value_scale) or value_scale <= 0.0:
        raise ValueError("value_scale must be finite and positive")
    device = previous.device
    dtype = previous.dtype
    indices = torch.arange(capacity + 1, device=device)
    feasible = indices >= item_weight
    skip = previous
    take = previous.clone()
    if item_weight <= capacity:
        take[item_weight:] = previous[:-item_weight] + float(item_value) / value_scale
    capacity_position = indices.to(dtype=dtype) / float(capacity)
    weight_fraction = torch.full_like(previous, float(item_weight) / float(capacity))
    value_fraction = torch.full_like(previous, float(item_value) / value_scale)
    difference = take - skip
    features = torch.stack(
        [
            skip,
            take,
            difference,
            capacity_position,
            weight_fraction,
            value_fraction,
            feasible.to(dtype=dtype),
        ],
        dim=1,
    )
    return features, skip, take, feasible


def policy_features(instance: KnapsackInstance, *, device: torch.device | str = "cpu") -> Tensor:
    weights = np.asarray(instance.weights, dtype=np.float32)
    values = np.asarray(instance.values, dtype=np.float32)
    densities = values / weights
    max_density = max(float(np.max(densities)), 1.0)
    features = np.column_stack(
        [
            weights / float(instance.capacity),
            values / float(instance.total_value),
            densities / max_density,
            np.full(
                instance.item_count,
                instance.capacity / instance.total_weight,
                dtype=np.float32,
            ),
            np.full(instance.item_count, 1.0 / instance.item_count, dtype=np.float32),
        ]
    ).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32, device=device)


def _checkpoint_header(model: BellmanReasoner | DirectPolicy) -> dict[str, str]:
    if isinstance(model, BellmanReasoner):
        model_type = "bellman_reasoner"
        config = asdict(model.config)
    elif isinstance(model, DirectPolicy):
        model_type = "direct_policy"
        config = asdict(model.config)
    else:
        raise TypeError("unsupported checkpoint model type")
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_type": model_type,
        "model_config": json.dumps(config, sort_keys=True),
    }


def save_checkpoint(
    model: BellmanReasoner | DirectPolicy,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = _checkpoint_header(model)
    header["metadata"] = json.dumps(metadata or {}, sort_keys=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(tensors, str(output), metadata=header)


def _config_integer(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint model field {name!r} must be an integer")
    return value


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[BellmanReasoner | DirectPolicy, dict[str, object]]:
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata()
        tensors = {
            key: handle.get_tensor(key)
            for key in handle.keys()  # noqa: SIM118 -- Safetensors is not iterable.
        }
    if header is None:
        raise ValueError("checkpoint metadata is missing")
    if header.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    raw_config: object = json.loads(header["model_config"])
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    config = cast(dict[str, object], raw_config)
    model_type = header.get("model_type")
    if model_type == "bellman_reasoner":
        model: BellmanReasoner | DirectPolicy = BellmanReasoner(
            BellmanReasonerConfig(
                hidden_dim=_config_integer(config, "hidden_dim"),
                hidden_layers=_config_integer(config, "hidden_layers"),
            )
        )
    elif model_type == "direct_policy":
        model = DirectPolicy(
            DirectPolicyConfig(
                hidden_dim=_config_integer(config, "hidden_dim"),
                hidden_layers=_config_integer(config, "hidden_layers"),
            )
        )
    else:
        raise ValueError("checkpoint model type is unsupported")
    model.load_state_dict(tensors, strict=True)
    model.to(device)
    raw_metadata: object = json.loads(header.get("metadata", "{}"))
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata payload is invalid")
    return model, cast(dict[str, object], raw_metadata)
