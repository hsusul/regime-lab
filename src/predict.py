"""Cached prediction helpers for the FastAPI API.

This module does not download live data. It reads local cached raw/processed
data and trained artifacts produced by the CLI milestones.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import RAW_DATA_DIR, validate_ticker
from src.evaluate import load_model_artifact, resolve_experiment_record
from src.features import PROCESSED_DATA_DIR, build_features
from src.features import find_raw_cache_file as find_cached_raw_file
from src.train import (
    EXPERIMENTS_FILENAME,
    REPORTS_DIR,
    TRAINING_TARGET_COLUMN,
    LabeledDataNotFoundError,
    TrainingError,
    find_labeled_feature_file,
    load_experiments,
)


TICKER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,19}$")
KEY_SIGNAL_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma_50_distance",
    "ma_200_distance",
    "drawdown_60d",
    "volume_change_20d",
]


class PredictionError(Exception):
    """Base exception for cached prediction failures."""


class InvalidTickerFormatError(PredictionError):
    """Raised when an API ticker path value is malformed."""


class NoModelAvailableError(PredictionError):
    """Raised when no valid completed model artifact exists."""


class CachedDataNotFoundError(PredictionError):
    """Raised when cached feature/raw/history data is unavailable."""


class PredictionInputError(PredictionError):
    """Raised when cached data cannot satisfy model feature requirements."""


def validate_api_ticker(ticker: str) -> str:
    """Validate ticker format for API paths."""
    normalized = validate_ticker(ticker)
    if not TICKER_PATTERN.fullmatch(normalized):
        raise InvalidTickerFormatError(f"Invalid ticker format: {ticker}")
    return normalized


def experiments_path(reports_dir: Path | None = None) -> Path:
    """Return the file-based experiments metadata path."""
    return (reports_dir or REPORTS_DIR) / EXPERIMENTS_FILENAME


def list_experiment_records(reports_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load experiment records, returning an empty list when absent."""
    try:
        return load_experiments(experiments_path(reports_dir))
    except (json.JSONDecodeError, TrainingError):
        return []


def latest_valid_experiment(reports_dir: Path | None = None) -> dict[str, Any]:
    """Resolve latest completed experiment with an existing artifact path."""
    records = list_experiment_records(reports_dir)
    candidates = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("artifact_path")
        and Path(record["artifact_path"]).exists()
    ]
    if not candidates:
        raise NoModelAvailableError("No completed experiment with a valid artifact.")
    return sorted(candidates, key=lambda item: item.get("created_at", ""))[-1]


def latest_model_artifact(reports_dir: Path | None = None) -> dict[str, Any]:
    """Load the latest valid model artifact."""
    record = latest_valid_experiment(reports_dir)
    return load_model_artifact(Path(record["artifact_path"]))


def load_metrics_payload(
    experiment_id: str | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Load full or minimal metrics for an experiment."""
    selected_reports_dir = reports_dir or REPORTS_DIR
    record = (
        latest_valid_experiment(selected_reports_dir)
        if experiment_id in (None, "latest")
        else resolve_experiment_record(experiment_id, reports_dir=selected_reports_dir)
    )
    selected_experiment_id = str(record["experiment_id"])
    metrics_path = selected_reports_dir / f"metrics_{selected_experiment_id}.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if "accuracy" in record and "macro_f1" in record:
        return {
            "experiment_id": selected_experiment_id,
            "model_type": record.get("model_type"),
            "tickers": record.get("tickers", []),
            "train_period": record.get("training_date_range"),
            "test_period": record.get("test_date_range"),
            "accuracy": record["accuracy"],
            "macro_f1": record["macro_f1"],
            "per_class": {},
            "confusion_matrix": [],
            "label_names": [],
            "feature_importance": [],
            "warnings": [
                f"Full metrics file not found at {metrics_path}; "
                "returning headline metrics from experiments.json."
            ],
        }

    raise CachedDataNotFoundError(
        f"No metrics available for experiment {selected_experiment_id}."
    )


def find_feature_or_labeled_file(
    ticker: str,
    processed_data_dir: Path | None = None,
) -> Path:
    """Find newest processed feature CSV, preferring labeled data."""
    selected_dir = processed_data_dir or PROCESSED_DATA_DIR
    normalized = validate_api_ticker(ticker)
    try:
        return find_labeled_feature_file(normalized, selected_dir)
    except LabeledDataNotFoundError:
        pattern = f"{normalized.replace('/', '-')}*_features_*.csv"
        matches = [
            path
            for path in selected_dir.glob(pattern)
            if "_labeled_" not in path.stem
        ]
        matches = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise CachedDataNotFoundError(
        f"No cached processed feature data available for {normalized}."
    )


def load_feature_data_for_ticker(
    ticker: str,
    *,
    processed_data_dir: Path | None = None,
    raw_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load cached processed features, or compute them from cached raw CSVs."""
    normalized = validate_api_ticker(ticker)
    try:
        feature_path = find_feature_or_labeled_file(normalized, processed_data_dir)
        return pd.read_csv(feature_path)
    except CachedDataNotFoundError:
        selected_raw_dir = raw_data_dir or RAW_DATA_DIR
        try:
            raw_path = find_cached_raw_file(normalized, raw_data_dir=selected_raw_dir)
        except Exception as exc:
            raise CachedDataNotFoundError(
                f"No cached raw or processed data available for {normalized}."
            ) from exc
        raw = pd.read_csv(raw_path)
        return build_features(raw, dropna=True)


def load_labeled_history_for_ticker(
    ticker: str,
    processed_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load cached labeled feature history for a ticker."""
    normalized = validate_api_ticker(ticker)
    selected_dir = processed_data_dir or PROCESSED_DATA_DIR
    try:
        path = find_labeled_feature_file(normalized, selected_dir)
    except LabeledDataNotFoundError as exc:
        raise CachedDataNotFoundError(
            f"No labeled history available for {normalized}."
        ) from exc
    return pd.read_csv(path)


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _probabilities_for_model(
    artifact: dict[str, Any],
    feature_row: pd.DataFrame,
) -> tuple[float | None, dict[str, float] | None]:
    model = artifact["model"]
    if not hasattr(model, "predict_proba"):
        return None, None

    probabilities = model.predict_proba(feature_row)[0]
    classes = [str(label) for label in model.classes_]
    mapped = {label: 0.0 for label in artifact["label_names"]}
    mapped.update(
        {label: float(probability) for label, probability in zip(classes, probabilities)}
    )
    return max(mapped.values()), mapped


def predict_latest_regime(
    ticker: str,
    *,
    reports_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    raw_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Predict the latest cached regime for one ticker."""
    normalized = validate_api_ticker(ticker)
    artifact = latest_model_artifact(reports_dir)
    features = load_feature_data_for_ticker(
        normalized,
        processed_data_dir=processed_data_dir,
        raw_data_dir=raw_data_dir,
    )
    feature_columns = list(artifact["feature_columns"])
    missing = sorted(set(feature_columns) - set(features.columns))
    if missing:
        raise PredictionInputError(
            f"Cached feature data missing model inputs: {', '.join(missing)}"
        )

    usable = features.dropna(subset=feature_columns).sort_values("date")
    if usable.empty:
        raise CachedDataNotFoundError(
            f"No rows with complete model features available for {normalized}."
        )
    latest = usable.iloc[-1]
    feature_row = latest[feature_columns].to_frame().T
    regime = str(artifact["model"].predict(feature_row)[0])
    confidence, probabilities = _probabilities_for_model(artifact, feature_row)

    return {
        "ticker": normalized,
        "as_of": str(latest["date"]),
        "regime": regime,
        "confidence": confidence,
        "probabilities": probabilities,
        "key_signals": {
            column: _safe_float(latest.get(column)) for column in KEY_SIGNAL_COLUMNS
        },
        "experiment_id": artifact["experiment_id"],
        "model_id": artifact["experiment_id"],
        "warnings": [],
    }


def history_for_ticker(
    ticker: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    reports_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Return cached label history and optional model predictions."""
    normalized = validate_api_ticker(ticker)
    history = load_labeled_history_for_ticker(normalized, processed_data_dir)
    history["date"] = pd.to_datetime(history["date"])

    if start_date and end_date and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError("start_date must be before or equal to end_date.")
    if start_date:
        history = history[history["date"] >= pd.to_datetime(start_date)]
    if end_date:
        history = history[history["date"] <= pd.to_datetime(end_date)]
    history = history.sort_values("date")
    if limit is not None:
        history = history.tail(limit)
    if history.empty:
        raise CachedDataNotFoundError(f"No history rows available for {normalized}.")

    warnings: list[str] = []
    predicted = pd.Series([None] * len(history), index=history.index, dtype="object")
    try:
        artifact = latest_model_artifact(reports_dir)
        feature_columns = list(artifact["feature_columns"])
        missing = sorted(set(feature_columns) - set(history.columns))
        if missing:
            warnings.append(
                f"prediction skipped; history missing features: {', '.join(missing)}"
            )
        else:
            complete = history[feature_columns].notna().all(axis=1)
            if complete.any():
                predicted.loc[complete] = artifact["model"].predict(
                    history.loc[complete, feature_columns]
                )
            if (~complete).any():
                warnings.append("some rows skipped for predictions due to missing features.")
    except NoModelAvailableError:
        warnings.append("no valid trained model available; predicted_regime is null.")

    rows = []
    for index, row in history.iterrows():
        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "rule_label": None
                if pd.isna(row.get(TRAINING_TARGET_COLUMN))
                else str(row.get(TRAINING_TARGET_COLUMN)),
                "predicted_regime": None
                if pd.isna(predicted.loc[index])
                else str(predicted.loc[index]),
                "close": _safe_float(row.get("close")),
                "return_20d": _safe_float(row.get("return_20d")),
                "volatility_20d": _safe_float(row.get("volatility_20d")),
                "drawdown_60d": _safe_float(row.get("drawdown_60d")),
            }
        )

    return {
        "ticker": normalized,
        "rows": rows,
        "count": len(rows),
        "warnings": warnings,
    }


def filter_experiments(
    *,
    reports_dir: Path | None = None,
    limit: int | None = None,
    model_type: str | None = None,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """List experiment records sorted newest first with optional filters."""
    records = list_experiment_records(reports_dir)
    if model_type:
        records = [record for record in records if record.get("model_type") == model_type]
    if ticker:
        normalized = validate_api_ticker(ticker)
        records = [
            record
            for record in records
            if normalized in {str(item).upper() for item in record.get("tickers", [])}
        ]
    records = sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)
    if limit is not None:
        records = records[:limit]
    return records
