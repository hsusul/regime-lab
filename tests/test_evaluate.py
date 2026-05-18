"""Tests for model evaluation."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from src.evaluate import (
    MissingArtifactMetadataError,
    MissingEvaluationColumnsError,
    artifact_environment_metadata,
    artifact_path_from_experiment,
    evaluate_artifact,
    load_model_artifact,
    resolve_experiment_record,
    validate_evaluation_columns,
)
from src.features import FEATURE_COLUMNS
from src.labeling import REGIME_LABELS
from src.train import EXPERIMENTS_FILENAME, train_model
from tests.test_train import make_labeled_data


def write_labeled_feature_files(data: pd.DataFrame, processed_dir) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for ticker, group in data.groupby("ticker"):
        path = processed_dir / f"{ticker}_synthetic_features_v1_labeled_v1.csv"
        group.to_csv(path, index=False)


def train_artifact_with_processed_data(
    tmp_path,
    *,
    model_type: str = "random_forest",
    tickers: tuple[str, ...] = ("SPY",),
    periods: int = 24,
    created_at: str = "2024-02-01T00:00:00+00:00",
):
    data = make_labeled_data(tickers=tickers, periods=periods)
    processed_dir = tmp_path / "processed"
    write_labeled_feature_files(data, processed_dir)
    result = train_model(
        data,
        model_type,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        created_at=created_at,
    )
    return result, processed_dir, tmp_path / "reports"


def test_artifact_loading_round_trip(tmp_path) -> None:
    result, _, _ = train_artifact_with_processed_data(tmp_path)

    artifact = load_model_artifact(result.artifact_path)

    assert artifact["experiment_id"] == result.experiment_id
    assert artifact["model_type"] == "random_forest"
    assert artifact["feature_columns"] == FEATURE_COLUMNS


def test_required_metadata_validation(tmp_path) -> None:
    bad_artifact = tmp_path / "bad.joblib"
    joblib.dump({"model": object()}, bad_artifact)

    with pytest.raises(MissingArtifactMetadataError, match="feature_columns"):
        load_model_artifact(bad_artifact)


def test_artifact_sklearn_version_mismatch_warns(tmp_path) -> None:
    result, _, _ = train_artifact_with_processed_data(tmp_path)
    artifact = joblib.load(result.artifact_path)
    artifact["sklearn_version"] = "0.0.0"
    artifact["environment"]["sklearn_version"] = "0.0.0"
    joblib.dump(artifact, result.artifact_path)

    with pytest.warns(UserWarning, match="differs from current sklearn_version"):
        loaded = load_model_artifact(result.artifact_path)

    assert loaded["sklearn_version"] == "0.0.0"


def test_artifact_environment_metadata_supports_top_level_fallback() -> None:
    metadata = artifact_environment_metadata(
        {
            "python_version": "3.x",
            "sklearn_version": "1.x",
            "pandas_version": "2.x",
            "numpy_version": "2.x",
        }
    )

    assert metadata == {
        "python_version": "3.x",
        "sklearn_version": "1.x",
        "pandas_version": "2.x",
        "numpy_version": "2.x",
    }


def test_feature_column_validation() -> None:
    data = make_labeled_data().drop(columns=["return_20d"])

    with pytest.raises(MissingEvaluationColumnsError, match="return_20d"):
        validate_evaluation_columns(data, FEATURE_COLUMNS)


def test_evaluation_reconstructs_test_split(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(
        tmp_path,
        periods=24,
    )

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert evaluation.metrics["train_period"] == result.metadata["training_date_range"]
    assert evaluation.metrics["test_period"] == result.metadata["test_date_range"]
    assert evaluation.warnings == []


def test_accuracy_and_macro_f1_output_keys(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(tmp_path)

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert "accuracy" in evaluation.metrics
    assert "macro_f1" in evaluation.metrics
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert 0.0 <= evaluation.macro_f1 <= 1.0


def test_per_class_metrics_shape(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(tmp_path)

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert set(evaluation.metrics["per_class"]) == set(REGIME_LABELS)
    for metrics in evaluation.metrics["per_class"].values():
        assert set(metrics) == {"precision", "recall", "f1", "support"}


def test_confusion_matrix_shape_and_label_order(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(tmp_path)

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    matrix = evaluation.metrics["confusion_matrix"]
    assert evaluation.metrics["label_names"] == REGIME_LABELS
    assert len(matrix) == len(REGIME_LABELS)
    assert all(len(row) == len(REGIME_LABELS) for row in matrix)


def test_random_forest_feature_importance_extraction(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(
        tmp_path,
        model_type="random_forest",
    )

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    importance = evaluation.metrics["feature_importance"]
    assert len(importance) == len(FEATURE_COLUMNS)
    assert {item["feature"] for item in importance} == set(FEATURE_COLUMNS)
    assert importance == sorted(
        importance,
        key=lambda item: item["importance"],
        reverse=True,
    )


def test_logistic_regression_warning_and_empty_feature_importance(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(
        tmp_path,
        model_type="logistic_regression",
    )

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert evaluation.metrics["feature_importance"] == []
    assert any("logistic_regression" in warning for warning in evaluation.warnings)


def test_metrics_json_saving(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(tmp_path)

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert evaluation.metrics_path.exists()
    with evaluation.metrics_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["experiment_id"] == result.experiment_id
    assert saved["accuracy"] == evaluation.accuracy


def test_latest_experiment_resolution(tmp_path) -> None:
    first, _, reports_dir = train_artifact_with_processed_data(
        tmp_path,
        created_at="2024-02-01T00:00:00+00:00",
    )
    second, _, _ = train_artifact_with_processed_data(
        tmp_path,
        created_at="2024-02-02T00:00:00+00:00",
    )

    record = resolve_experiment_record("latest", reports_dir=reports_dir)
    artifact_path = artifact_path_from_experiment("latest", reports_dir=reports_dir)

    assert record["experiment_id"] == second.experiment_id
    assert artifact_path == second.artifact_path
    assert first.experiment_id != second.experiment_id


def test_latest_experiment_ignores_invalid_artifact_path(tmp_path) -> None:
    valid, _, reports_dir = train_artifact_with_processed_data(
        tmp_path,
        created_at="2024-02-01T00:00:00+00:00",
    )
    experiments_path = reports_dir / "experiments.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiments.append(
        {
            "experiment_id": "newer-but-missing",
            "created_at": "2024-02-02T00:00:00+00:00",
            "status": "completed",
            "model_type": "random_forest",
            "tickers": ["SPY"],
            "artifact_path": str(tmp_path / "models" / "missing.joblib"),
        }
    )
    experiments_path.write_text(json.dumps(experiments), encoding="utf-8")

    record = resolve_experiment_record("latest", reports_dir=reports_dir)

    assert record["experiment_id"] == valid.experiment_id


def test_label_names_ordering_is_preserved_in_metrics(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(tmp_path)

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert evaluation.metrics["label_names"] == REGIME_LABELS


def test_experiments_json_updated_with_accuracy_and_macro_f1(tmp_path) -> None:
    result, processed_dir, reports_dir = train_artifact_with_processed_data(tmp_path)

    evaluation = evaluate_artifact(
        result.artifact_path,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
    )

    with (reports_dir / EXPERIMENTS_FILENAME).open("r", encoding="utf-8") as file:
        experiments = json.load(file)

    matching = [
        record
        for record in experiments
        if record["experiment_id"] == result.experiment_id
    ][0]
    assert matching["accuracy"] == evaluation.accuracy
    assert matching["macro_f1"] == evaluation.macro_f1
    assert matching["metrics_path"] == str(evaluation.metrics_path)
