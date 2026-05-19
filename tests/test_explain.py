"""Tests for lightweight explanation reports."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from src.explain import (
    MissingExplanationFeaturesError,
    build_explanation_report,
    build_parser,
    feature_importance_for_artifact,
    latest_explanation_for_ticker,
    run_explanation_report,
)
from src.features import FEATURE_COLUMNS
from src.train import train_model
from tests.test_api import write_labeled_files
from tests.test_train import make_labeled_data


def train_explain_fixture(
    tmp_path,
    *,
    model_type: str = "random_forest",
    periods: int = 24,
):
    data = make_labeled_data(periods=periods)
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    reports_dir.mkdir()
    write_labeled_files(data, processed_dir)
    result = train_model(
        data,
        model_type,
        models_dir=models_dir,
        reports_dir=reports_dir,
        created_at="2024-05-01T00:00:00+00:00",
    )
    artifact = joblib.load(result.artifact_path)
    return artifact, processed_dir, reports_dir, result


def test_random_forest_feature_importance_is_ranked(tmp_path) -> None:
    artifact, _, _, _ = train_explain_fixture(tmp_path, model_type="random_forest")

    importance, warnings = feature_importance_for_artifact(artifact)

    assert warnings == []
    assert len(importance) == len(FEATURE_COLUMNS)
    assert {item["feature"] for item in importance} == set(FEATURE_COLUMNS)
    assert importance == sorted(
        importance,
        key=lambda item: item["importance"],
        reverse=True,
    )


def test_logistic_regression_coefficients_are_reported_with_warning(tmp_path) -> None:
    artifact, _, _, _ = train_explain_fixture(tmp_path, model_type="logistic_regression")

    coefficients, warnings = feature_importance_for_artifact(artifact)

    assert coefficients
    assert all("class_label" in row for row in coefficients)
    assert all("coefficients" in row for row in coefficients)
    assert any("coefficients" in warning for warning in warnings)


def test_latest_explanation_for_ticker_includes_prediction_context(tmp_path) -> None:
    artifact, processed_dir, _, _ = train_explain_fixture(tmp_path)

    explanation = latest_explanation_for_ticker(
        "SPY",
        artifact,
        processed_data_dir=processed_dir,
    )

    assert explanation["ticker"] == "SPY"
    assert explanation["as_of"] == "2024-01-24"
    assert explanation["predicted_regime"] in artifact["label_names"]
    assert explanation["confidence"] is not None
    assert set(explanation["key_feature_values"]) == {
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "ma_50_distance",
        "ma_200_distance",
        "drawdown_60d",
        "volume_change_20d",
    }


def test_latest_explanation_raises_for_missing_model_features(tmp_path) -> None:
    artifact, processed_dir, _, _ = train_explain_fixture(tmp_path)
    feature_path = next(processed_dir.glob("*_labeled_*.csv"))
    data = pd.read_csv(feature_path).drop(columns=["return_20d"])
    data.to_csv(feature_path, index=False)

    with pytest.raises(MissingExplanationFeaturesError, match="return_20d"):
        latest_explanation_for_ticker(
            "SPY",
            artifact,
            processed_data_dir=processed_dir,
        )


def test_build_explanation_report_contains_expected_shape(tmp_path) -> None:
    _, processed_dir, reports_dir, result = train_explain_fixture(tmp_path)

    report = build_explanation_report(
        ["SPY"],
        reports_dir=reports_dir,
        processed_data_dir=processed_dir,
        created_at="2024-05-02T00:00:00+00:00",
    )

    assert report["analysis_type"] == "model_explanation"
    assert report["experiment_id"] == result.experiment_id
    assert report["model_type"] == "random_forest"
    assert report["tickers"] == ["SPY"]
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert len(report["predictions"]) == 1
    assert report["global_feature_importance"]
    assert any("not financial advice" in warning for warning in report["warnings"])
    assert any("SHAP" in warning for warning in report["warnings"])


def test_run_explanation_report_saves_json(tmp_path) -> None:
    _, processed_dir, reports_dir, result = train_explain_fixture(tmp_path)

    output = run_explanation_report(
        ["SPY"],
        reports_dir=reports_dir,
        processed_data_dir=processed_dir,
        created_at="2024-05-03T00:00:00+00:00",
    )

    assert output.report_path.exists()
    saved = json.loads(output.report_path.read_text(encoding="utf-8"))
    assert saved["experiment_id"] == result.experiment_id
    assert saved["predictions"][0]["ticker"] == "SPY"


def test_explain_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(["--tickers", "SPY", "QQQ"])

    assert args.tickers == ["SPY", "QQQ"]
