"""Tests for MVP diagnostics reporting."""

from __future__ import annotations

import json

from src.diagnostics import (
    build_diagnostics_report,
    format_text_report,
    main,
    summarize_distribution,
)
from src.train import train_model
from tests.test_api import write_labeled_files
from tests.test_train import make_labeled_data


def test_summarize_distribution_counts_and_percentages() -> None:
    data = make_labeled_data(periods=4)["rule_label"]

    summary = summarize_distribution(data)

    assert summary["total"] == 4
    assert summary["labels"]["stable_growth"] == {"count": 1, "pct": 0.25}


def test_build_diagnostics_report_without_model(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    reports_dir.mkdir()
    write_labeled_files(make_labeled_data(periods=8), processed_dir)

    report = build_diagnostics_report(
        ["SPY"],
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert report["model"]["available"] is False
    assert report["by_ticker"]["SPY"]["row_count"] == 8
    assert report["by_ticker"]["SPY"]["predicted_regime_distribution"]["total"] == 0
    assert "no valid completed model artifact found." in report["warnings"]


def test_build_diagnostics_report_with_model_predictions(tmp_path) -> None:
    data = make_labeled_data(tickers=("SPY", "QQQ"), periods=24)
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    reports_dir.mkdir()
    write_labeled_files(data, processed_dir)
    result = train_model(
        data,
        "random_forest",
        models_dir=models_dir,
        reports_dir=reports_dir,
        created_at="2024-05-01T00:00:00+00:00",
    )

    report = build_diagnostics_report(
        ["SPY", "QQQ"],
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert report["model"]["available"] is True
    assert report["model"]["experiment_id"] == result.experiment_id
    assert report["model"]["artifact_environment"]["sklearn_version"]
    assert report["model"]["current_environment"]["sklearn_version"]
    assert report["by_ticker"]["SPY"]["predicted_regime_distribution"]["total"] == 24
    assert report["combined"]["row_count"] == 48


def test_diagnostics_warns_when_label_dominates(tmp_path) -> None:
    data = make_labeled_data(periods=10)
    data["rule_label"] = "stable_growth"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    reports_dir.mkdir()
    write_labeled_files(data, processed_dir)

    report = build_diagnostics_report(
        ["SPY"],
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    warnings = report["by_ticker"]["SPY"]["warnings"]
    assert any("dominates 100.0%" in warning for warning in warnings)


def test_text_report_includes_key_sections(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    reports_dir.mkdir()
    write_labeled_files(make_labeled_data(periods=8), processed_dir)
    report = build_diagnostics_report(
        ["SPY"],
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    text = format_text_report(report)

    assert "RegimeLab diagnostics" in text
    assert "SPY" in text
    assert "Combined" in text


def test_diagnostics_cli_json_output(tmp_path, capsys) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    reports_dir.mkdir()
    write_labeled_files(make_labeled_data(periods=8), processed_dir)

    exit_code = main(
        [
            "--tickers",
            "SPY",
            "--processed-data-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["tickers"] == ["SPY"]
