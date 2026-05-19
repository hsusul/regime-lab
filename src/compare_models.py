"""Model comparison reports for RegimeLab.

This module trains/evaluates requested baseline model types using existing
pipeline functions, then writes a separate comparison report. It does not alter
model definitions, API behavior, or registry resolution logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

from src.evaluate import evaluate_artifact
from src.features import PROCESSED_DATA_DIR
from src.train import (
    MODELS_DIR,
    REPORTS_DIR,
    SUPPORTED_MODEL_TYPES,
    load_labeled_tickers,
    train_model,
)


MODEL_COMPARISON_PREFIX = "model_comparison"


class ModelComparisonError(Exception):
    """Base exception for model comparison failures."""


class InvalidComparisonModelError(ModelComparisonError):
    """Raised when an unsupported model type is requested."""


@dataclass(frozen=True)
class ModelComparisonResult:
    """Saved model comparison report metadata."""

    report_path: Path
    csv_path: Path | None
    report: dict[str, Any]


def validate_model_types(model_types: Sequence[str]) -> list[str]:
    """Validate and de-duplicate requested model types in user-provided order."""
    unique: list[str] = []
    for model_type in model_types:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise InvalidComparisonModelError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )
        if model_type not in unique:
            unique.append(model_type)
    if not unique:
        raise InvalidComparisonModelError("At least one model type is required.")
    return unique


def comparison_row(
    *,
    model_type: str,
    artifact_path: Path,
    evaluation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build one comparison row from an evaluated artifact."""
    artifact = joblib.load(artifact_path)
    return {
        "model_type": model_type,
        "experiment_id": evaluation_metrics["experiment_id"],
        "accuracy": evaluation_metrics["accuracy"],
        "macro_f1": evaluation_metrics["macro_f1"],
        "training_date_range": artifact["training_date_range"],
        "test_date_range": artifact["test_date_range"],
        "artifact_path": str(artifact_path),
        "feature_version": artifact["feature_version"],
        "labeling_version": artifact["labeling_version"],
    }


def build_model_comparison_report(
    tickers: Sequence[str],
    *,
    model_types: Sequence[str],
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and compare requested model types."""
    selected_model_types = validate_model_types(model_types)
    data = load_labeled_tickers(tickers, processed_data_dir=processed_data_dir)
    rows: list[dict[str, Any]] = []

    for model_type in selected_model_types:
        training = train_model(
            data,
            model_type,
            models_dir=models_dir,
            reports_dir=reports_dir,
        )
        evaluation = evaluate_artifact(
            training.artifact_path,
            processed_data_dir=processed_data_dir,
            reports_dir=reports_dir,
        )
        rows.append(
            comparison_row(
                model_type=model_type,
                artifact_path=training.artifact_path,
                evaluation_metrics=evaluation.metrics,
            )
        )

    timestamp = created_at or datetime.now(UTC).isoformat(timespec="microseconds")
    best_by_macro_f1 = max(rows, key=lambda row: float(row["macro_f1"]))
    return {
        "analysis_type": "model_comparison",
        "created_at": timestamp,
        "tickers": sorted({str(ticker).upper() for ticker in tickers}),
        "models": rows,
        "best_model_type": best_by_macro_f1["model_type"],
        "best_experiment_id": best_by_macro_f1["experiment_id"],
        "selection_metric": "macro_f1",
        "warnings": [
            "Comparison metrics measure agreement with heuristic labels, "
            "not trading profitability."
        ],
    }


def flatten_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested date ranges for CSV output."""
    return {
        "model_type": row["model_type"],
        "experiment_id": row["experiment_id"],
        "accuracy": row["accuracy"],
        "macro_f1": row["macro_f1"],
        "train_start": row["training_date_range"]["start_date"],
        "train_end": row["training_date_range"]["end_date"],
        "test_start": row["test_date_range"]["start_date"],
        "test_end": row["test_date_range"]["end_date"],
        "artifact_path": row["artifact_path"],
        "feature_version": row["feature_version"],
        "labeling_version": row["labeling_version"],
    }


def save_model_comparison_report(
    report: dict[str, Any],
    *,
    reports_dir: Path = REPORTS_DIR,
    save_csv: bool = False,
) -> ModelComparisonResult:
    """Persist comparison report JSON and optional CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
        report["created_at"]
        .replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    report_path = reports_dir / f"{MODEL_COMPARISON_PREFIX}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    csv_path = None
    if save_csv:
        csv_path = reports_dir / f"{MODEL_COMPARISON_PREFIX}_{timestamp}.csv"
        pd.DataFrame(
            [flatten_comparison_row(row) for row in report["models"]]
        ).to_csv(csv_path, index=False)

    return ModelComparisonResult(report_path=report_path, csv_path=csv_path, report=report)


def run_model_comparison(
    tickers: Sequence[str],
    *,
    model_types: Sequence[str],
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_csv: bool = False,
    created_at: str | None = None,
) -> ModelComparisonResult:
    """Build and save a model comparison report."""
    report = build_model_comparison_report(
        tickers,
        model_types=model_types,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        created_at=created_at,
    )
    return save_model_comparison_report(
        report,
        reports_dir=reports_dir,
        save_csv=save_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the model comparison CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and compare baseline model types."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
        help="Model types to compare.",
    )
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
        help="Directory for reports and experiment metadata.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save a flattened CSV comparison table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for model comparison."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_model_comparison(
            args.tickers,
            model_types=args.models,
            processed_data_dir=args.processed_data_dir,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            save_csv=args.save_csv,
        )
    except ModelComparisonError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print("model_comparison_report")
    print(f"tickers: {result.report['tickers']}")
    print(f"best_model_type: {result.report['best_model_type']}")
    print(f"selection_metric: {result.report['selection_metric']}")
    print(f"report_path: {result.report_path}")
    if result.csv_path is not None:
        print(f"csv_path: {result.csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
