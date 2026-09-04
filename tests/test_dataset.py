import json
from pathlib import Path

import pytest

from narcopt.dataset import (
    SUPPORTED_REGIMES,
    collect_dataset,
    generate_instance,
    load_dataset,
    save_dataset,
)


def test_generation_is_deterministic_and_regimes_are_valid() -> None:
    for regime in SUPPORTED_REGIMES:
        first = generate_instance(item_count=9, regime=regime, seed=77)
        second = generate_instance(item_count=9, regime=regime, seed=77)
        assert first == second
        assert first.capacity < first.total_weight
        assert min(first.weights) > 0
        assert min(first.values) > 0
    with pytest.raises(ValueError):
        generate_instance(item_count=1, regime="iid", seed=1)
    with pytest.raises(ValueError):
        generate_instance(item_count=4, regime="missing", seed=1)
    with pytest.raises(ValueError):
        generate_instance(item_count=4, regime="iid", seed=1, capacity_ratio=0.01)


def test_dataset_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    dataset = collect_dataset(
        count=6,
        item_counts=(4, 6),
        regimes=("iid", "heavy_tail"),
        seed=21,
    )
    path = tmp_path / "dataset.jsonl"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert loaded.fingerprint == dataset.fingerprint
    assert loaded.item_counts == (4, 6)
    assert loaded.regimes == ("heavy_tail", "iid")

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["exact_solution"]["objective"] += 1
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_collection_validation() -> None:
    with pytest.raises(ValueError):
        collect_dataset(count=0)
    with pytest.raises(ValueError):
        collect_dataset(count=1, item_counts=())
    with pytest.raises(ValueError):
        collect_dataset(count=1, regimes=("unknown",))
