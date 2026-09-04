from pathlib import Path

from narcopt.dataset import collect_dataset
from narcopt.evaluation import evaluate_models, save_report_csv, save_report_json
from narcopt.experiment import ResearchConfig, run_research_experiment, save_research_report
from narcopt.models import BellmanReasoner, BellmanReasonerConfig, DirectPolicy, DirectPolicyConfig


def test_evaluation_reports_exact_search(tmp_path: Path) -> None:
    dataset = collect_dataset(count=3, item_counts=(7,), regimes=("iid",), seed=55)
    reasoner = BellmanReasoner(BellmanReasonerConfig(8, 1))
    policy = DirectPolicy(DirectPolicyConfig(8, 1))
    report = evaluate_models(
        reasoner,
        policy,
        dataset,
        scenario="test",
        bootstrap_draws=10,
        bootstrap_seed=9,
    )
    assert len(report.approximation_rows) == 3
    assert len(report.search_rows) == 3
    assert report.metadata["all_exact_search_results_verified"] is True
    assert all(row.exact_solution_rate == 1.0 for row in report.search_rows)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json_path.exists()
    assert "row_type" in csv_path.read_text(encoding="utf-8")


def test_tiny_research_protocol(tmp_path: Path) -> None:
    config = ResearchConfig(
        train_samples=6,
        validation_samples=3,
        evaluation_samples=1,
        train_item_counts=(4, 5),
        reasoner_hidden_dim=8,
        policy_hidden_dim=8,
        hidden_layers=1,
        reasoner_epochs=1,
        policy_epochs=1,
        batch_size=32,
        bootstrap_draws=5,
        seed=123,
    )
    models, report = run_research_experiment(
        config,
        checkpoint_directory=tmp_path / "checkpoints",
    )
    assert set(models) == {"trace_reasoner", "direct_policy"}
    assert len(report.trace_rows) == 9
    assert len(report.approximation_rows) == 27
    assert len(report.search_rows) == 27
    output = tmp_path / "research.json"
    save_research_report(report, output)
    assert output.exists()
