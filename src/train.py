"""Supervised model training for RegimeLab.

This module implements Milestone 5 only. It trains baseline classifiers from
heuristically labeled feature data and does not expose prediction endpoints or
evaluation reports beyond basic training metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import validate_ticker
from src.features import FEATURE_COLUMNS, FEATURE_VERSION, PROCESSED_DATA_DIR
from src.labeling import LABELING_VERSION, REGIME_LABELS


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
EXPERIMENTS_FILENAME = "experiments.json"
TRAINING_TARGET_COLUMN = "rule_label"
SUPPORTED_MODEL_TYPES = ["logistic_regression", "random_forest"]
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8


class TrainingError(Exception):
    """Base exception for training failures."""


class MissingTrainingColumnsError(TrainingError):
    """Raised when required feature or target columns are missing."""


class InvalidModelTypeError(TrainingError):
    """Raised when a requested model type is unsupported."""


class InsufficientTrainingDataError(TrainingError):
    """Raised when the dataset cannot support a chronological split."""


class LabeledDataNotFoundError(TrainingError):
    """Raised when a labeled processed CSV cannot be found."""


@dataclass(frozen=True)
class TimeSplit:
    """Chronological train/test split."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_date: str


@dataclass(frozen=True)
class TrainingResult:
    """Metadata returned after a successful training run."""

    experiment_id: str
    model_type: str
    artifact_path: Path
    summary_path: Path
    train_rows: int
    test_rows: int
    label_distribution: dict[str, int]
    metadata: dict[str, Any]


def validate_training_columns(data: pd.DataFrame) -> None:
    """Validate feature columns and target column for training."""
    required = set(FEATURE_COLUMNS) | {TRAINING_TARGET_COLUMN, "date", "ticker"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise MissingTrainingColumnsError(
            f"Missing required training columns: {', '.join(missing)}"
        )


def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and sort labeled feature data for training."""
    validate_training_columns(data)
    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared = prepared.dropna(subset=FEATURE_COLUMNS + [TRAINING_TARGET_COLUMN])
    prepared = prepared[prepared[TRAINING_TARGET_COLUMN].isin(REGIME_LABELS)]
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    if prepared.empty:
        raise InsufficientTrainingDataError(
            "No rows remain after dropping missing features or labels."
        )
    return prepared


def time_based_split(
    data: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> TimeSplit:
    """Split data by a shared chronological cutoff date with no shuffling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    prepared = data.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    unique_dates = prepared["date"].drop_duplicates().sort_values().reset_index(
        drop=True
    )
    if len(unique_dates) < 2:
        raise InsufficientTrainingDataError(
            "At least two unique dates are required for a time-based split."
        )

    train_date_count = int(len(unique_dates) * train_fraction)
    train_date_count = max(1, min(train_date_count, len(unique_dates) - 1))
    cutoff_date = unique_dates.iloc[train_date_count - 1]

    train = prepared[prepared["date"] <= cutoff_date].reset_index(drop=True)
    test = prepared[prepared["date"] > cutoff_date].reset_index(drop=True)
    if train.empty or test.empty:
        raise InsufficientTrainingDataError(
            "Time-based split produced an empty train or test set."
        )

    return TimeSplit(
        train=train,
        test=test,
        cutoff_date=cutoff_date.date().isoformat(),
    )


def create_model(model_type: str) -> Pipeline | RandomForestClassifier:
    """Create a deterministic baseline classifier."""
    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    raise InvalidModelTypeError(
        f"Unsupported model type {model_type!r}. "
        f"Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
    )


def date_range_for(data: pd.DataFrame) -> dict[str, str]:
    """Return inclusive date range metadata for a split."""
    dates = pd.to_datetime(data["date"])
    return {
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
    }


def label_distribution_for(data: pd.DataFrame) -> dict[str, int]:
    """Return label counts for the cleaned training dataset."""
    counts = data[TRAINING_TARGET_COLUMN].value_counts().sort_index()
    return {label: int(count) for label, count in counts.items()}


def make_experiment_id(model_type: str, created_at: str) -> str:
    """Create a filesystem-safe experiment identifier."""
    timestamp = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    return f"{timestamp}_{model_type}"


def current_environment_metadata() -> dict[str, str]:
    """Return dependency versions relevant to artifact reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }


def build_artifact(
    *,
    model: Pipeline | RandomForestClassifier,
    model_type: str,
    split: TimeSplit,
    ticker_universe: list[str],
    created_at: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Build the persisted model artifact dictionary."""
    environment = current_environment_metadata()
    return {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_names": list(REGIME_LABELS),
        "model_type": model_type,
        "training_date_range": date_range_for(split.train),
        "test_date_range": date_range_for(split.test),
        "ticker_universe": ticker_universe,
        "feature_version": FEATURE_VERSION,
        "labeling_version": LABELING_VERSION,
        "created_at": created_at,
        "experiment_id": experiment_id,
        "python_version": environment["python_version"],
        "sklearn_version": environment["sklearn_version"],
        "pandas_version": environment["pandas_version"],
        "numpy_version": environment["numpy_version"],
        "environment": environment,
    }


def load_experiments(experiments_path: Path) -> list[dict[str, Any]]:
    """Load existing file-based experiment metadata."""
    if not experiments_path.exists():
        return []
    with experiments_path.open("r", encoding="utf-8") as file:
        content = json.load(file)
    if isinstance(content, list):
        return content
    if isinstance(content, dict) and isinstance(content.get("experiments"), list):
        return content["experiments"]
    raise TrainingError(f"Invalid experiments metadata format: {experiments_path}")


def append_experiment(
    experiments_path: Path,
    record: dict[str, Any],
) -> None:
    """Append one experiment record to reports/experiments.json."""
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(experiments_path)
    experiments.append(record)
    with experiments_path.open("w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=2)
        file.write("\n")


def train_model(
    data: pd.DataFrame,
    model_type: str,
    *,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    created_at: str | None = None,
) -> TrainingResult:
    """Train a baseline classifier and persist its artifact and metadata."""
    cleaned = prepare_training_data(data)
    split = time_based_split(cleaned)
    model = create_model(model_type)

    x_train = split.train[FEATURE_COLUMNS]
    y_train = split.train[TRAINING_TARGET_COLUMN]
    model.fit(x_train, y_train)

    timestamp = created_at or datetime.now(UTC).isoformat(timespec="microseconds")
    experiment_id = make_experiment_id(model_type, timestamp)
    ticker_universe = sorted(cleaned["ticker"].unique().tolist())
    label_distribution = label_distribution_for(cleaned)

    artifact = build_artifact(
        model=model,
        model_type=model_type,
        split=split,
        ticker_universe=ticker_universe,
        created_at=timestamp,
        experiment_id=experiment_id,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"{experiment_id}.joblib"
    joblib.dump(artifact, artifact_path)

    metadata = {
        "experiment_id": experiment_id,
        "created_at": timestamp,
        "status": "completed",
        "model_type": model_type,
        "tickers": ticker_universe,
        "artifact_path": str(artifact_path),
        "feature_version": FEATURE_VERSION,
        "labeling_version": LABELING_VERSION,
        "training_date_range": artifact["training_date_range"],
        "test_date_range": artifact["test_date_range"],
        "label_distribution": label_distribution,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "environment": artifact["environment"],
    }

    experiments_path = reports_dir / EXPERIMENTS_FILENAME
    append_experiment(experiments_path, metadata)

    summary_path = reports_dir / f"{experiment_id}_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return TrainingResult(
        experiment_id=experiment_id,
        model_type=model_type,
        artifact_path=artifact_path,
        summary_path=summary_path,
        train_rows=len(split.train),
        test_rows=len(split.test),
        label_distribution=label_distribution,
        metadata=metadata,
    )


def find_labeled_feature_file(
    ticker: str,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Find the newest labeled processed feature CSV for one ticker."""
    normalized_ticker = validate_ticker(ticker)
    pattern = f"{normalized_ticker.replace('/', '-')}*_features_*_labeled_*.csv"
    matches = sorted(
        processed_data_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise LabeledDataNotFoundError(
            f"No labeled feature files found for {normalized_ticker} "
            f"in {processed_data_dir}."
        )
    return matches[0]


def load_labeled_tickers(
    tickers: Sequence[str],
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load and concatenate labeled processed CSVs for tickers."""
    frames = [
        pd.read_csv(find_labeled_feature_file(ticker, processed_data_dir))
        for ticker in tickers
    ]
    if not frames:
        raise LabeledDataNotFoundError("At least one ticker is required.")
    return pd.concat(frames, ignore_index=True)


def train_from_processed_files(
    tickers: Sequence[str],
    model_type: str,
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> TrainingResult:
    """Load labeled processed CSVs and train a model."""
    data = load_labeled_tickers(tickers, processed_data_dir=processed_data_dir)
    return train_model(
        data,
        model_type,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the model training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a baseline regime classifier from labeled features."
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
        help="Model type to train.",
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing labeled processed feature CSV files.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory for saved model artifacts.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for training reports and experiment metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for supervised model training."""
    parser = build_parser()
    args = parser.parse_args(argv)

    result = train_from_processed_files(
        args.tickers,
        args.model_type,
        processed_data_dir=args.processed_data_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )

    print(f"experiment_id: {result.experiment_id}")
    print(f"model_type: {result.model_type}")
    print(f"train_rows: {result.train_rows}")
    print(f"test_rows: {result.test_rows}")
    print(f"label_distribution: {result.label_distribution}")
    print(f"artifact_path: {result.artifact_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
