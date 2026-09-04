"""Command-line workflows for data, trace learning, exact search, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from narcopt.dataset import (
    SUPPORTED_REGIMES,
    collect_dataset,
    generate_instance,
    load_dataset,
    save_dataset,
)
from narcopt.domain import (
    KnapsackSolution,
    audit_solution,
    solve_brute_force,
    solve_dynamic_programming,
)
from narcopt.evaluation import evaluate_models, save_report_csv, save_report_json
from narcopt.experiment import ResearchConfig, run_research_experiment, save_research_report
from narcopt.models import (
    BellmanReasoner,
    BellmanReasonerConfig,
    DirectPolicy,
    DirectPolicyConfig,
    load_checkpoint,
    save_checkpoint,
)
from narcopt.training import TrainingConfig, train_model
from narcopt.utils import write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narcopt",
        description="Trace-supervised neural algorithmic reasoning for exact knapsack search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one exact-labeled instance")
    generate.add_argument("--item-count", type=int, default=12)
    generate.add_argument("--regime", choices=SUPPORTED_REGIMES, default="iid")
    generate.add_argument("--capacity-ratio", type=float)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="build a deterministic JSONL corpus")
    collect.add_argument("--count", type=int, required=True)
    collect.add_argument("--item-counts", type=int, nargs="+", default=[8, 10, 12, 14])
    collect.add_argument("--regimes", nargs="+", choices=SUPPORTED_REGIMES, default=["iid"])
    collect.add_argument("--seed", type=int, default=1000)
    collect.add_argument("--output", type=Path, required=True)

    oracle = subparsers.add_parser("oracle", help="verify DP against exhaustive enumeration")
    oracle.add_argument("dataset", type=Path)
    oracle.add_argument("--sample-index", type=int, default=0)
    oracle.add_argument("--maximum-items", type=int, default=24)
    oracle.add_argument("--output", type=Path)

    train = subparsers.add_parser("train", help="train trace-supervised or terminal-only model")
    train.add_argument("dataset", type=Path)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--model", choices=("trace_reasoner", "direct_policy"), required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--hidden-layers", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--output-report", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare neural candidates and exact search guidance"
    )
    benchmark.add_argument("dataset", type=Path)
    benchmark.add_argument("--reasoner-checkpoint", type=Path, required=True)
    benchmark.add_argument("--policy-checkpoint", type=Path, required=True)
    benchmark.add_argument("--scenario", default="benchmark")
    benchmark.add_argument("--bootstrap-draws", type=int, default=300)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output-json", type=Path, required=True)
    benchmark.add_argument("--output-csv", type=Path)

    research = subparsers.add_parser("research", help="run the frozen transfer protocol")
    research.add_argument("--config", type=Path)
    research.add_argument("--checkpoint-directory", type=Path)
    research.add_argument("--output-report", type=Path, required=True)
    return parser


def _json_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _load_research_config(path: Path | None) -> ResearchConfig:
    if path is None:
        return ResearchConfig()
    payload = _json_object(json.loads(path.read_text(encoding="utf-8")), name="research config")
    defaults = ResearchConfig()
    allowed = {
        "train_samples",
        "validation_samples",
        "evaluation_samples",
        "train_item_counts",
        "reasoner_hidden_dim",
        "policy_hidden_dim",
        "hidden_layers",
        "reasoner_epochs",
        "policy_epochs",
        "batch_size",
        "learning_rate",
        "bootstrap_draws",
        "seed",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown research config fields: {sorted(unknown)}")

    def integer(name: str, default: int) -> int:
        value = payload.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"research config field {name!r} must be an integer")
        return value

    def number(name: str, default: float) -> float:
        value = payload.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"research config field {name!r} must be numeric")
        return float(value)

    raw_counts = payload.get("train_item_counts", list(defaults.train_item_counts))
    if not isinstance(raw_counts, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw_counts
    ):
        raise ValueError("train_item_counts must be an integer array")
    return ResearchConfig(
        train_samples=integer("train_samples", defaults.train_samples),
        validation_samples=integer("validation_samples", defaults.validation_samples),
        evaluation_samples=integer("evaluation_samples", defaults.evaluation_samples),
        train_item_counts=tuple(raw_counts),
        reasoner_hidden_dim=integer("reasoner_hidden_dim", defaults.reasoner_hidden_dim),
        policy_hidden_dim=integer("policy_hidden_dim", defaults.policy_hidden_dim),
        hidden_layers=integer("hidden_layers", defaults.hidden_layers),
        reasoner_epochs=integer("reasoner_epochs", defaults.reasoner_epochs),
        policy_epochs=integer("policy_epochs", defaults.policy_epochs),
        batch_size=integer("batch_size", defaults.batch_size),
        learning_rate=number("learning_rate", defaults.learning_rate),
        bootstrap_draws=integer("bootstrap_draws", defaults.bootstrap_draws),
        seed=integer("seed", defaults.seed),
    )


def _load_models(
    reasoner_path: Path,
    policy_path: Path,
) -> tuple[BellmanReasoner, DirectPolicy]:
    reasoner_model, _ = load_checkpoint(reasoner_path)
    policy_model, _ = load_checkpoint(policy_path)
    if not isinstance(reasoner_model, BellmanReasoner):
        raise ValueError("reasoner checkpoint does not contain a BellmanReasoner")
    if not isinstance(policy_model, DirectPolicy):
        raise ValueError("policy checkpoint does not contain a DirectPolicy")
    return reasoner_model, policy_model


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "generate":
        instance = generate_instance(
            item_count=args.item_count,
            regime=args.regime,
            seed=args.seed,
            capacity_ratio=args.capacity_ratio,
        )
        solution = solve_dynamic_programming(instance)
        if not isinstance(solution, KnapsackSolution):
            raise RuntimeError("unexpected dynamic-programming return type")
        payload = {
            "instance": instance.to_dict(),
            "exact_solution": solution.to_dict(),
            "audit": audit_solution(
                instance,
                solution.selection,
                reported_objective=solution.objective,
                verify_optimality=True,
            ).to_dict(),
        }
        write_json(payload, args.output)
        return {"output": str(args.output), **payload}

    if args.command == "collect":
        dataset = collect_dataset(
            count=args.count,
            item_counts=tuple(args.item_counts),
            regimes=tuple(args.regimes),
            seed=args.seed,
        )
        save_dataset(dataset, args.output)
        return {
            "output": str(args.output),
            "record_count": len(dataset.instances),
            "fingerprint": dataset.fingerprint,
            "item_counts": list(dataset.item_counts),
            "regimes": list(dataset.regimes),
        }

    if args.command == "oracle":
        dataset = load_dataset(args.dataset)
        if not 0 <= args.sample_index < len(dataset.instances):
            raise ValueError("sample index is outside the dataset")
        instance = dataset.instances[args.sample_index]
        dynamic = solve_dynamic_programming(instance)
        brute_force = solve_brute_force(instance, maximum_items=args.maximum_items)
        if not isinstance(dynamic, KnapsackSolution):
            raise RuntimeError("unexpected dynamic-programming return type")
        payload = {
            "dataset_fingerprint": dataset.fingerprint,
            "instance_id": instance.instance_id,
            "dynamic_programming": dynamic.to_dict(),
            "brute_force": brute_force.to_dict(),
            "verified": dynamic.objective == brute_force.objective,
        }
        if not payload["verified"]:
            raise RuntimeError("dynamic programming disagrees with exhaustive enumeration")
        if args.output is not None:
            write_json(payload, args.output)
        return payload

    if args.command == "train":
        dataset = load_dataset(args.dataset)
        validation = load_dataset(args.validation)
        model: BellmanReasoner | DirectPolicy
        if args.model == "trace_reasoner":
            model = BellmanReasoner(BellmanReasonerConfig(args.hidden_dim, args.hidden_layers))
        else:
            model = DirectPolicy(DirectPolicyConfig(args.hidden_dim, args.hidden_layers))
        config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_every=1,
            patience_checks=max(2, args.epochs),
        )
        summary = train_model(model, dataset, validation, config=config)
        save_checkpoint(
            model,
            args.checkpoint,
            metadata={
                "training_mode": args.model,
                "train_fingerprint": dataset.fingerprint,
                "validation_fingerprint": validation.fingerprint,
                "training_config": asdict(config),
            },
        )
        payload = {"checkpoint": str(args.checkpoint), **summary.to_dict()}
        if args.output_report is not None:
            write_json(payload, args.output_report)
        return payload

    if args.command == "benchmark":
        dataset = load_dataset(args.dataset)
        reasoner, policy = _load_models(args.reasoner_checkpoint, args.policy_checkpoint)
        report = evaluate_models(
            reasoner,
            policy,
            dataset,
            scenario=args.scenario,
            bootstrap_seed=args.seed,
            bootstrap_draws=args.bootstrap_draws,
        )
        save_report_json(report, args.output_json)
        if args.output_csv is not None:
            save_report_csv(report, args.output_csv)
        return report.to_dict()

    if args.command == "research":
        config = _load_research_config(args.config)
        _models, report = run_research_experiment(
            config,
            checkpoint_directory=args.checkpoint_directory,
        )
        save_research_report(report, args.output_report)
        return report.to_dict()

    raise RuntimeError("unreachable command")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        payload = _run(args)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
