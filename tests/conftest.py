from pathlib import Path

import pytest

from narcopt.dataset import KnapsackDataset, collect_dataset


@pytest.fixture
def tiny_train() -> KnapsackDataset:
    return collect_dataset(
        count=8,
        item_counts=(4, 5, 6),
        regimes=("iid", "weakly_correlated"),
        seed=100,
    )


@pytest.fixture
def tiny_validation() -> KnapsackDataset:
    return collect_dataset(
        count=4,
        item_counts=(5, 6),
        regimes=("iid", "strongly_correlated"),
        seed=200,
    )


@pytest.fixture
def output_directory(tmp_path: Path) -> Path:
    path = tmp_path / "artifacts"
    path.mkdir()
    return path
