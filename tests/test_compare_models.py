"""Tests for model comparison reports."""

from __future__ import annotations

import json

import pytest

from src.compare_models import (
    InvalidComparisonModelError,
    build_model_comparison_report,
    build_parser,
    flatten_comparison_row,
    run_model_comparison,
    validate_model_types,
)
from tests.test_api import write_labeled_files
from tests.test_train import make_labeled_data


def test_validate_model_types_deduplicates_and_rejects_invalid() -> None:
    assert validate_model_types(["random_forest", "random_forest"]) == [
        "random_forest"
    ]

    with pytest.raises(InvalidComparisonModelError):
        validate_model_types(["xgboost"])


def test_build_model_comparison_report_trains_and_evaluates_models(tmp_path) -> None:
    data = make_labeled_data(periods=28)
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    write_labeled_files(data, processed_dir)

    report = build_model_comparison_report(
        ["SPY"],
        model_types=["logistic_regression", "random_forest"],
        processed_data_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert report["analysis_type"] == "model_comparison"
    assert report["tickers"] == ["SPY"]
    assert report["selection_metric"] == "macro_f1"
    assert report["best_model_type"] in {"logistic_regression", "random_forest"}
    assert [row["model_type"] for row in report["models"]] == [
        "logistic_regression",
        "random_forest",
    ]
    for row in report["models"]:
        assert {
            "model_type",
            "experiment_id",
            "accuracy",
            "macro_f1",
            "training_date_range",
            "test_date_range",
            "artifact_path",
            "feature_version",
            "labeling_version",
        } <= set(row)


def test_flatten_comparison_row_for_csv() -> None:
    row = {
        "model_type": "random_forest",
        "experiment_id": "exp",
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "training_date_range": {
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
        },
        "test_date_range": {
            "start_date": "2021-01-02",
            "end_date": "2022-01-01",
        },
        "artifact_path": "models/exp.joblib",
        "feature_version": "v1",
        "labeling_version": "v1",
    }

    flattened = flatten_comparison_row(row)

    assert flattened["train_start"] == "2020-01-01"
    assert flattened["train_end"] == "2021-01-01"
    assert flattened["test_start"] == "2021-01-02"
    assert flattened["test_end"] == "2022-01-01"


def test_run_model_comparison_saves_json_and_optional_csv(tmp_path) -> None:
    data = make_labeled_data(periods=28)
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    write_labeled_files(data, processed_dir)

    result = run_model_comparison(
        ["SPY"],
        model_types=["random_forest"],
        processed_data_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        save_csv=True,
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert result.report_path.exists()
    assert result.csv_path is not None
    assert result.csv_path.exists()
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["models"][0]["model_type"] == "random_forest"
    csv_text = result.csv_path.read_text(encoding="utf-8")
    assert "macro_f1" in csv_text
    assert "train_start" in csv_text


def test_compare_models_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--tickers",
            "SPY",
            "QQQ",
            "--models",
            "logistic_regression",
            "random_forest",
            "--save-csv",
        ]
    )

    assert args.tickers == ["SPY", "QQQ"]
    assert args.models == ["logistic_regression", "random_forest"]
    assert args.save_csv is True
