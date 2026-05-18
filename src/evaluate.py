"""Model evaluation for trained RegimeLab artifacts.

This module implements Milestone 6 only. It evaluates trained supervised
artifacts against held-out chronological test rows labeled by the rule-based
heuristic target.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score

from src.features import PROCESSED_DATA_DIR
from src.train import (
    EXPERIMENTS_FILENAME,
    REPORTS_DIR,
    TRAINING_TARGET_COLUMN,
    TrainingError,
    current_environment_metadata,
    load_experiments,
    load_labeled_tickers,
    prepare_training_data,
    time_based_split,
)


REQUIRED_ARTIFACT_KEYS = {
    "model",
    "feature_columns",
    "label_names",
    "model_type",
    "training_date_range",
    "test_date_range",
    "ticker_universe",
    "feature_version",
    "labeling_version",
    "created_at",
    "experiment_id",
}


class EvaluationError(Exception):
    """Base exception for evaluation failures."""


class MissingArtifactMetadataError(EvaluationError):
    """Raised when an artifact is missing required metadata."""


class MissingEvaluationColumnsError(EvaluationError):
    """Raised when labeled data cannot satisfy artifact feature requirements."""


class ExperimentNotFoundError(EvaluationError):
    """Raised when requested experiment metadata cannot be resolved."""


@dataclass(frozen=True)
class EvaluationResult:
    """Result metadata for an evaluation run."""

    experiment_id: str
    model_type: str
    accuracy: float
    macro_f1: float
    metrics_path: Path
    warnings: list[str]
    metrics: dict[str, Any]


def artifact_environment_metadata(artifact: dict[str, Any]) -> dict[str, str | None]:
    """Return environment metadata from artifact, supporting older artifacts."""
    environment = artifact.get("environment")
    if isinstance(environment, dict):
        return {
            "python_version": environment.get("python_version"),
            "sklearn_version": environment.get("sklearn_version"),
            "pandas_version": environment.get("pandas_version"),
            "numpy_version": environment.get("numpy_version"),
        }
    return {
        "python_version": artifact.get("python_version"),
        "sklearn_version": artifact.get("sklearn_version"),
        "pandas_version": artifact.get("pandas_version"),
        "numpy_version": artifact.get("numpy_version"),
    }


def artifact_compatibility_warnings(artifact: dict[str, Any]) -> list[str]:
    """Return reproducibility warnings for artifact dependency metadata."""
    artifact_environment = artifact_environment_metadata(artifact)
    current_environment = current_environment_metadata()
    artifact_sklearn = artifact_environment.get("sklearn_version")
    current_sklearn = current_environment["sklearn_version"]

    if artifact_sklearn is None:
        return [
            "Artifact does not contain sklearn_version metadata; retrain the "
            "model in the current environment for reproducibility."
        ]
    if artifact_sklearn != current_sklearn:
        return [
            "Artifact sklearn_version "
            f"{artifact_sklearn!r} differs from current sklearn_version "
            f"{current_sklearn!r}; retrain the model in the current environment."
        ]
    return []


def load_model_artifact(artifact_path: Path) -> dict[str, Any]:
    """Load and validate a trained joblib artifact."""
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise MissingArtifactMetadataError(
            f"Artifact must be a dictionary: {artifact_path}"
        )
    missing = sorted(REQUIRED_ARTIFACT_KEYS - set(artifact))
    if missing:
        raise MissingArtifactMetadataError(
            f"Artifact missing required metadata keys: {', '.join(missing)}"
        )
    for warning in artifact_compatibility_warnings(artifact):
        warnings.warn(warning, UserWarning, stacklevel=2)
    return artifact


def validate_evaluation_columns(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """Validate labeled data contains target and artifact feature columns."""
    required = set(feature_columns) | {TRAINING_TARGET_COLUMN, "date", "ticker"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise MissingEvaluationColumnsError(
            f"Missing required evaluation columns: {', '.join(missing)}"
        )


def feature_importance_for_artifact(
    artifact: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, float | str]]:
    """Extract feature importance when the model exposes it directly."""
    feature_columns = list(artifact["feature_columns"])
    model_type = artifact["model_type"]
    model = artifact["model"]

    if model_type == "random_forest" and hasattr(model, "feature_importances_"):
        importances = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(
                feature_columns,
                model.feature_importances_,
                strict=True,
            )
        ]
        return sorted(importances, key=lambda item: item["importance"], reverse=True)

    if model_type == "logistic_regression":
        warnings.append(
            "feature_importance omitted for logistic_regression because raw "
            "coefficients require careful class-specific interpretation."
        )
        return []

    warnings.append(f"feature_importance unavailable for model_type={model_type}.")
    return []


def build_metrics(
    artifact: dict[str, Any],
    test_data: pd.DataFrame,
    predictions: Sequence[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the metrics JSON payload."""
    label_names = list(artifact["label_names"])
    y_true = test_data[TRAINING_TARGET_COLUMN]

    report = classification_report(
        y_true,
        predictions,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in label_names
    }

    return {
        "experiment_id": artifact["experiment_id"],
        "model_type": artifact["model_type"],
        "tickers": list(artifact["ticker_universe"]),
        "train_period": artifact["training_date_range"],
        "test_period": artifact["test_date_range"],
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=label_names,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=label_names,
        ).tolist(),
        "label_names": label_names,
        "feature_importance": feature_importance_for_artifact(artifact, warnings),
        "warnings": warnings,
    }


def metrics_path_for_experiment(
    experiment_id: str,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Return metrics output path for an experiment."""
    return reports_dir / f"metrics_{experiment_id}.json"


def update_experiment_metrics(
    experiments_path: Path,
    experiment_id: str,
    accuracy: float,
    macro_f1: float,
    metrics_path: Path,
) -> None:
    """Update a matching experiment record with headline metrics."""
    experiments = load_experiments(experiments_path)
    updated = False
    for record in experiments:
        if record.get("experiment_id") == experiment_id:
            record["accuracy"] = accuracy
            record["macro_f1"] = macro_f1
            record["metrics_path"] = str(metrics_path)
            updated = True
            break

    if not updated:
        return

    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    with experiments_path.open("w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=2)
        file.write("\n")


def evaluate_artifact(
    artifact_path: Path,
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    update_experiments: bool = True,
) -> EvaluationResult:
    """Evaluate one artifact against reconstructed chronological test data."""
    artifact = load_model_artifact(artifact_path)
    warnings: list[str] = []
    feature_columns = list(artifact["feature_columns"])

    data = load_labeled_tickers(
        artifact["ticker_universe"],
        processed_data_dir=processed_data_dir,
    )
    validate_evaluation_columns(data, feature_columns)
    cleaned = prepare_training_data(data)
    split = time_based_split(cleaned)

    reconstructed_train_period = {
        "start_date": split.train["date"].min().date().isoformat(),
        "end_date": split.train["date"].max().date().isoformat(),
    }
    reconstructed_test_period = {
        "start_date": split.test["date"].min().date().isoformat(),
        "end_date": split.test["date"].max().date().isoformat(),
    }
    if reconstructed_train_period != artifact["training_date_range"]:
        warnings.append("reconstructed training date range differs from artifact.")
    if reconstructed_test_period != artifact["test_date_range"]:
        warnings.append("reconstructed test date range differs from artifact.")

    predictions = artifact["model"].predict(split.test[feature_columns])
    metrics = build_metrics(artifact, split.test, predictions, warnings)

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_path_for_experiment(artifact["experiment_id"], reports_dir)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    if update_experiments:
        update_experiment_metrics(
            reports_dir / EXPERIMENTS_FILENAME,
            artifact["experiment_id"],
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics_path,
        )
        from src.experiment_registry import update_experiment_metrics as update_registry

        update_registry(
            artifact["experiment_id"],
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            },
            metrics_path=metrics_path,
            db_path=reports_dir / "regimelab.db",
        )

    return EvaluationResult(
        experiment_id=artifact["experiment_id"],
        model_type=artifact["model_type"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        metrics_path=metrics_path,
        warnings=warnings,
        metrics=metrics,
    )


def resolve_experiment_record(
    experiment_id: str,
    *,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Resolve an experiment record by ID or latest completed experiment."""
    if experiment_id == "latest":
        from src.experiment_registry import active_experiment

        active = active_experiment(db_path=reports_dir / "regimelab.db")
        if active is not None:
            return active

    experiments_path = reports_dir / EXPERIMENTS_FILENAME
    experiments = load_experiments(experiments_path)

    if experiment_id == "latest":
        candidates = [
            record
            for record in experiments
            if record.get("status") == "completed"
            and record.get("artifact_path")
            and Path(record["artifact_path"]).exists()
        ]
        if not candidates:
            raise ExperimentNotFoundError(
                f"No completed experiments with valid artifacts in {experiments_path}."
            )
        return sorted(candidates, key=lambda item: item.get("created_at", ""))[-1]

    for record in experiments:
        if record.get("experiment_id") == experiment_id:
            return record

    raise ExperimentNotFoundError(
        f"Experiment {experiment_id!r} not found in {experiments_path}."
    )


def artifact_path_from_experiment(
    experiment_id: str,
    *,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Resolve artifact path from experiment metadata."""
    record = resolve_experiment_record(experiment_id, reports_dir=reports_dir)
    artifact_path = Path(record["artifact_path"])
    if not artifact_path.exists():
        raise ExperimentNotFoundError(f"Artifact does not exist: {artifact_path}")
    return artifact_path


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model artifact.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--experiment-id",
        default="latest",
        help="Experiment ID to evaluate, or 'latest'. Defaults to latest.",
    )
    source.add_argument("--artifact-path", type=Path, help="Artifact path to evaluate.")
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing labeled processed feature CSV files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for metrics output and experiment metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for model evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        artifact_path = args.artifact_path or artifact_path_from_experiment(
            args.experiment_id,
            reports_dir=args.reports_dir,
        )
        result = evaluate_artifact(
            artifact_path,
            processed_data_dir=args.processed_data_dir,
            reports_dir=args.reports_dir,
        )
    except TrainingError as exc:
        raise EvaluationError(str(exc)) from exc

    print(f"experiment_id: {result.experiment_id}")
    print(f"model_type: {result.model_type}")
    print(f"accuracy: {result.accuracy:.6f}")
    print(f"macro_f1: {result.macro_f1:.6f}")
    print(f"metrics_path: {result.metrics_path}")
    if result.warnings:
        print(f"warnings: {result.warnings}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
