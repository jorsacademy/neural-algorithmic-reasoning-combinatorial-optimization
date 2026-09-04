import json
from pathlib import Path

from narcopt.cli import main


def test_cli_end_to_end(tmp_path: Path) -> None:
    generated = tmp_path / "generated.json"
    assert main(["generate", "--item-count", "5", "--seed", "5", "--output", str(generated)]) == 0
    assert json.loads(generated.read_text(encoding="utf-8"))["audit"]["optimal"]

    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    assert (
        main(
            [
                "collect",
                "--count",
                "6",
                "--item-counts",
                "4",
                "5",
                "--seed",
                "10",
                "--output",
                str(train),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "collect",
                "--count",
                "3",
                "--item-counts",
                "4",
                "5",
                "--seed",
                "20",
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    oracle = tmp_path / "oracle.json"
    assert main(["oracle", str(validation), "--output", str(oracle)]) == 0
    assert json.loads(oracle.read_text(encoding="utf-8"))["verified"]

    reasoner = tmp_path / "reasoner.safetensors"
    policy = tmp_path / "policy.safetensors"
    for model, checkpoint in (("trace_reasoner", reasoner), ("direct_policy", policy)):
        assert (
            main(
                [
                    "train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--model",
                    model,
                    "--epochs",
                    "1",
                    "--hidden-dim",
                    "8",
                    "--hidden-layers",
                    "1",
                    "--batch-size",
                    "32",
                    "--checkpoint",
                    str(checkpoint),
                ]
            )
            == 0
        )

    benchmark = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"
    assert (
        main(
            [
                "benchmark",
                str(validation),
                "--reasoner-checkpoint",
                str(reasoner),
                "--policy-checkpoint",
                str(policy),
                "--bootstrap-draws",
                "5",
                "--output-json",
                str(benchmark),
                "--output-csv",
                str(benchmark_csv),
            ]
        )
        == 0
    )
    assert json.loads(benchmark.read_text(encoding="utf-8"))["metadata"][
        "all_exact_search_results_verified"
    ]


def test_cli_reports_invalid_arguments() -> None:
    assert main(["generate", "--item-count", "1", "--output", "/tmp/invalid.json"]) == 2
