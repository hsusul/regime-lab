"""Real-data diagnostics for the RegimeLab MVP.

Diagnostics are read-only QA helpers over cached labeled data, trained
artifacts, and metrics. They do not train models or create new ML features.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.features import PROCESSED_DATA_DIR
from src.evaluate import artifact_environment_metadata, load_model_artifact
from src.predict import (
    KEY_SIGNAL_COLUMNS,
    CachedDataNotFoundError,
    NoModelAvailableError,
    latest_valid_experiment,
    load_labeled_history_for_ticker,
    load_metrics_payload,
    validate_api_ticker,
)
from src.train import REPORTS_DIR, TRAINING_TARGET_COLUMN, current_environment_metadata


DOMINANCE_THRESHOLD = 0.70


def summarize_distribution(
    values: pd.Series,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return counts and percentages for a categorical series."""
    cleaned = values.dropna().astype(str)
    total = int(len(cleaned))
    label_order = list(labels or sorted(cleaned.unique().tolist()))
    counts: dict[str, dict[str, float | int]] = {}

    for label in label_order:
        count = int((cleaned == label).sum())
        counts[label] = {
            "count": count,
            "pct": 0.0 if total == 0 else count / total,
        }

    return {"total": total, "labels": counts}


def dominance_warnings(
    distribution: dict[str, Any],
    *,
    scope: str,
    column_name: str,
    threshold: float = DOMINANCE_THRESHOLD,
) -> list[str]:
    """Return warnings when one label dominates a distribution."""
    warnings = []
    total = int(distribution["total"])
    if total == 0:
        return warnings

    for label, stats in distribution["labels"].items():
        pct = float(stats["pct"])
        if pct > threshold:
            warnings.append(
                f"{scope}: {column_name} label {label!r} dominates "
                f"{pct:.1%} of {total} rows."
            )
    return warnings


def _predict_for_frame(
    frame: pd.DataFrame,
    artifact: dict[str, Any] | None,
) -> tuple[pd.Series, list[str]]:
    """Predict regimes for rows with complete artifact feature columns."""
    predictions = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    warnings: list[str] = []
    if artifact is None:
        warnings.append("no valid model artifact available; predictions skipped.")
        return predictions, warnings

    feature_columns = list(artifact["feature_columns"])
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        warnings.append(
            "predictions skipped; labeled data missing feature columns: "
            + ", ".join(missing)
        )
        return predictions, warnings

    complete = frame[feature_columns].notna().all(axis=1)
    if complete.any():
        predictions.loc[complete] = artifact["model"].predict(
            frame.loc[complete, feature_columns]
        )
    skipped = int((~complete).sum())
    if skipped:
        warnings.append(
            f"predictions skipped for {skipped} rows with missing feature values."
        )
    return predictions, warnings


def _latest_row_summary(frame: pd.DataFrame, predictions: pd.Series) -> dict[str, Any]:
    """Summarize latest row and latest predicted/rule regime."""
    latest_index = frame.sort_values("date").index[-1]
    latest = frame.loc[latest_index]
    return {
        "date": pd.to_datetime(latest["date"]).date().isoformat(),
        "rule_label": None
        if pd.isna(latest.get(TRAINING_TARGET_COLUMN))
        else str(latest.get(TRAINING_TARGET_COLUMN)),
        "predicted_regime": None
        if pd.isna(predictions.loc[latest_index])
        else str(predictions.loc[latest_index]),
        "key_signals": {
            column: None if pd.isna(latest.get(column)) else float(latest.get(column))
            for column in KEY_SIGNAL_COLUMNS
            if column in latest.index
        },
    }


def _model_summary(
    artifact: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return model and metrics summary for the report."""
    if artifact is None:
        return {
            "available": False,
            "experiment_id": None,
            "model_type": None,
            "train_period": None,
            "test_period": None,
            "accuracy": None,
            "macro_f1": None,
            "artifact_environment": None,
            "current_environment": current_environment_metadata(),
        }

    return {
        "available": True,
        "experiment_id": artifact["experiment_id"],
        "model_type": artifact["model_type"],
        "train_period": artifact["training_date_range"],
        "test_period": artifact["test_date_range"],
        "accuracy": None if metrics is None else metrics.get("accuracy"),
        "macro_f1": None if metrics is None else metrics.get("macro_f1"),
        "artifact_environment": artifact_environment_metadata(artifact),
        "current_environment": current_environment_metadata(),
    }


def load_latest_artifact_with_warnings(
    reports_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Load latest artifact and capture model persistence warnings."""
    record = latest_valid_experiment(reports_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = load_model_artifact(Path(record["artifact_path"]))
    captured = list(dict.fromkeys(str(warning.message) for warning in caught))
    return artifact, captured


def build_diagnostics_report(
    tickers: Sequence[str],
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    dominance_threshold: float = DOMINANCE_THRESHOLD,
) -> dict[str, Any]:
    """Build a diagnostics report for cached labeled data and latest model."""
    normalized_tickers = [validate_api_ticker(ticker) for ticker in tickers]
    warnings: list[str] = []

    try:
        artifact, artifact_warnings = load_latest_artifact_with_warnings(reports_dir)
        warnings.extend(artifact_warnings)
    except NoModelAvailableError:
        artifact = None
        warnings.append("no valid completed model artifact found.")

    try:
        metrics = load_metrics_payload("latest", reports_dir=reports_dir)
    except (CachedDataNotFoundError, NoModelAvailableError):
        metrics = None
        warnings.append("latest metrics are unavailable.")

    combined_frames: list[pd.DataFrame] = []
    combined_predictions: list[pd.Series] = []
    by_ticker: dict[str, Any] = {}
    labels = list(artifact["label_names"]) if artifact is not None else None

    for ticker in normalized_tickers:
        ticker_warnings: list[str] = []
        data = load_labeled_history_for_ticker(
            ticker,
            processed_data_dir=processed_data_dir,
        ).copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)

        predictions, prediction_warnings = _predict_for_frame(data, artifact)
        ticker_warnings.extend(prediction_warnings)

        rule_distribution = summarize_distribution(
            data[TRAINING_TARGET_COLUMN],
            labels=labels,
        )
        predicted_distribution = summarize_distribution(predictions, labels=labels)
        ticker_warnings.extend(
            dominance_warnings(
                rule_distribution,
                scope=ticker,
                column_name=TRAINING_TARGET_COLUMN,
                threshold=dominance_threshold,
            )
        )
        ticker_warnings.extend(
            dominance_warnings(
                predicted_distribution,
                scope=ticker,
                column_name="predicted_regime",
                threshold=dominance_threshold,
            )
        )

        by_ticker[ticker] = {
            "row_count": int(len(data)),
            "rule_label_distribution": rule_distribution,
            "predicted_regime_distribution": predicted_distribution,
            "latest": _latest_row_summary(data, predictions),
            "warnings": ticker_warnings,
        }
        combined_frames.append(data)
        combined_predictions.append(predictions.reset_index(drop=True))

    combined_data = pd.concat(combined_frames, ignore_index=True)
    combined_predicted = pd.concat(combined_predictions, ignore_index=True)
    combined_rule_distribution = summarize_distribution(
        combined_data[TRAINING_TARGET_COLUMN],
        labels=labels,
    )
    combined_predicted_distribution = summarize_distribution(
        combined_predicted,
        labels=labels,
    )
    combined_warnings = dominance_warnings(
        combined_rule_distribution,
        scope="combined",
        column_name=TRAINING_TARGET_COLUMN,
        threshold=dominance_threshold,
    )
    combined_warnings.extend(
        dominance_warnings(
            combined_predicted_distribution,
            scope="combined",
            column_name="predicted_regime",
            threshold=dominance_threshold,
        )
    )

    return {
        "tickers": normalized_tickers,
        "model": _model_summary(artifact, metrics),
        "by_ticker": by_ticker,
        "combined": {
            "row_count": int(len(combined_data)),
            "rule_label_distribution": combined_rule_distribution,
            "predicted_regime_distribution": combined_predicted_distribution,
            "warnings": combined_warnings,
        },
        "warnings": warnings + combined_warnings,
    }


def _format_distribution(distribution: dict[str, Any]) -> list[str]:
    lines = []
    for label, stats in distribution["labels"].items():
        lines.append(f"    {label}: {stats['count']} ({stats['pct']:.1%})")
    return lines


def format_text_report(report: dict[str, Any]) -> str:
    """Format diagnostics report as human-readable text."""
    lines = ["RegimeLab diagnostics", ""]
    model = report["model"]
    lines.extend(
        [
            f"Model available: {model['available']}",
            f"Experiment: {model['experiment_id']}",
            f"Model type: {model['model_type']}",
            f"Train period: {model['train_period']}",
            f"Test period: {model['test_period']}",
            f"Accuracy: {model['accuracy']}",
            f"Macro F1: {model['macro_f1']}",
            f"Artifact environment: {model['artifact_environment']}",
            f"Current environment: {model['current_environment']}",
            "",
        ]
    )

    for ticker, summary in report["by_ticker"].items():
        latest = summary["latest"]
        lines.extend(
            [
                f"{ticker}",
                f"  rows: {summary['row_count']}",
                f"  latest: {latest['date']} rule={latest['rule_label']} "
                f"predicted={latest['predicted_regime']}",
                "  rule_label:",
                *_format_distribution(summary["rule_label_distribution"]),
                "  predicted_regime:",
                *_format_distribution(summary["predicted_regime_distribution"]),
            ]
        )
        if summary["warnings"]:
            lines.append("  warnings:")
            lines.extend(f"    {warning}" for warning in summary["warnings"])
        lines.append("")

    combined = report["combined"]
    lines.extend(
        [
            "Combined",
            f"  rows: {combined['row_count']}",
            "  rule_label:",
            *_format_distribution(combined["rule_label_distribution"]),
            "  predicted_regime:",
            *_format_distribution(combined["predicted_regime_distribution"]),
        ]
    )
    if report["warnings"]:
        lines.append("Warnings")
        lines.extend(f"  {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build diagnostics CLI parser."""
    parser = argparse.ArgumentParser(
        description="Summarize cached RegimeLab labels, predictions, and metrics."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing labeled processed data.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory containing experiments and metrics.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for diagnostics."""
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_diagnostics_report(
        args.tickers,
        processed_data_dir=args.processed_data_dir,
        reports_dir=args.reports_dir,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_text_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
