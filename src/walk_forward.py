"""Optional walk-forward validation for RegimeLab.

This module evaluates model stability across chronological folds. It is a
separate reporting path and does not change the MVP train/evaluate/API behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.features import FEATURE_COLUMNS, FEATURE_VERSION, PROCESSED_DATA_DIR
from src.labeling import LABELING_VERSION, REGIME_LABELS
from src.train import (
    REPORTS_DIR,
    SUPPORTED_MODEL_TYPES,
    TRAINING_TARGET_COLUMN,
    create_model,
    load_labeled_tickers,
    prepare_training_data,
)


DEFAULT_TEST_WINDOW_YEARS = 1
WALK_FORWARD_REPORT_PREFIX = "walk_forward"


class WalkForwardError(Exception):
    """Base exception for walk-forward validation failures."""


class InvalidWalkForwardConfigError(WalkForwardError):
    """Raised when walk-forward settings are invalid."""


class InsufficientWalkForwardDataError(WalkForwardError):
    """Raised when data cannot produce any chronological folds."""


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window train/test fold."""

    fold_index: int
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardResult:
    """Paths and report from a walk-forward validation run."""

    report_path: Path
    csv_path: Path | None
    report: dict[str, Any]


def validate_walk_forward_config(
    *,
    model_type: str,
    start_year: int,
    test_window_years: int,
) -> None:
    """Validate walk-forward settings."""
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise InvalidWalkForwardConfigError(
            f"Unsupported model_type {model_type!r}. "
            f"Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
        )
    if start_year < 1900:
        raise InvalidWalkForwardConfigError("start_year must be 1900 or later.")
    if test_window_years <= 0:
        raise InvalidWalkForwardConfigError("test_window_years must be positive.")


def build_walk_forward_folds(
    data: pd.DataFrame,
    *,
    start_year: int,
    test_window_years: int = DEFAULT_TEST_WINDOW_YEARS,
) -> list[WalkForwardFold]:
    """Build expanding-window folds over a shared chronological ticker universe."""
    if test_window_years <= 0:
        raise InvalidWalkForwardConfigError("test_window_years must be positive.")

    cleaned = prepare_training_data(data)
    max_date = cleaned["date"].max()
    test_start = pd.Timestamp(year=start_year, month=1, day=1)
    folds: list[WalkForwardFold] = []

    while test_start <= max_date:
        next_test_start = test_start + pd.DateOffset(years=test_window_years)
        train = cleaned[cleaned["date"] < test_start].reset_index(drop=True)
        test = cleaned[
            (cleaned["date"] >= test_start) & (cleaned["date"] < next_test_start)
        ].reset_index(drop=True)

        if not train.empty and not test.empty:
            folds.append(
                WalkForwardFold(
                    fold_index=len(folds) + 1,
                    train=train,
                    test=test,
                )
            )
        test_start = next_test_start

    if not folds:
        raise InsufficientWalkForwardDataError(
            "No walk-forward folds could be built. Choose a later start_year or "
            "provide more labeled historical data."
        )
    return folds


def _date_range(data: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(data["date"])
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def per_class_f1(y_true: pd.Series, predictions: Sequence[str]) -> dict[str, float]:
    """Return per-class F1 scores in stable regime-label order."""
    report = classification_report(
        y_true,
        predictions,
        labels=REGIME_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {label: float(report[label]["f1-score"]) for label in REGIME_LABELS}


def evaluate_fold(fold: WalkForwardFold, *, model_type: str) -> dict[str, Any]:
    """Train and evaluate one chronological fold."""
    model = create_model(model_type)
    model.fit(fold.train[FEATURE_COLUMNS], fold.train[TRAINING_TARGET_COLUMN])
    predictions = model.predict(fold.test[FEATURE_COLUMNS])
    train_start, train_end = _date_range(fold.train)
    test_start, test_end = _date_range(fold.test)

    return {
        "fold_index": fold.fold_index,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_rows": int(len(fold.train)),
        "test_rows": int(len(fold.test)),
        "accuracy": float(accuracy_score(fold.test[TRAINING_TARGET_COLUMN], predictions)),
        "macro_f1": float(
            f1_score(
                fold.test[TRAINING_TARGET_COLUMN],
                predictions,
                labels=REGIME_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": per_class_f1(fold.test[TRAINING_TARGET_COLUMN], predictions),
    }


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Return aggregate metrics across folds."""
    if not folds:
        return {
            "fold_count": 0,
            "mean_accuracy": None,
            "mean_macro_f1": None,
            "min_macro_f1": None,
            "max_macro_f1": None,
        }
    macro_f1_values = [float(fold["macro_f1"]) for fold in folds]
    accuracy_values = [float(fold["accuracy"]) for fold in folds]
    return {
        "fold_count": len(folds),
        "mean_accuracy": float(sum(accuracy_values) / len(accuracy_values)),
        "mean_macro_f1": float(sum(macro_f1_values) / len(macro_f1_values)),
        "min_macro_f1": float(min(macro_f1_values)),
        "max_macro_f1": float(max(macro_f1_values)),
    }


def flatten_fold_for_csv(fold: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested per-class F1 values into stable CSV columns."""
    flattened = {
        key: value for key, value in fold.items() if key != "per_class_f1"
    }
    per_class = fold.get("per_class_f1", {})
    for label in REGIME_LABELS:
        flattened[f"f1_{label}"] = per_class.get(label)
    return flattened


def build_walk_forward_report(
    data: pd.DataFrame,
    *,
    model_type: str,
    start_year: int,
    test_window_years: int = DEFAULT_TEST_WINDOW_YEARS,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a walk-forward validation report from labeled data."""
    validate_walk_forward_config(
        model_type=model_type,
        start_year=start_year,
        test_window_years=test_window_years,
    )
    cleaned = prepare_training_data(data)
    folds = build_walk_forward_folds(
        cleaned,
        start_year=start_year,
        test_window_years=test_window_years,
    )
    evaluated_folds = [evaluate_fold(fold, model_type=model_type) for fold in folds]
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="microseconds")
    data_start, data_end = _date_range(cleaned)

    return {
        "analysis_type": "walk_forward_validation",
        "created_at": timestamp,
        "model_type": model_type,
        "tickers": sorted(cleaned["ticker"].unique().tolist()),
        "rows": int(len(cleaned)),
        "data_date_range": {
            "start_date": data_start,
            "end_date": data_end,
        },
        "start_year": start_year,
        "test_window_years": test_window_years,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TRAINING_TARGET_COLUMN,
        "feature_version": FEATURE_VERSION,
        "labeling_version": LABELING_VERSION,
        "folds": evaluated_folds,
        "summary": summarize_folds(evaluated_folds),
        "warnings": [
            "Walk-forward metrics measure agreement with heuristic labels, "
            "not trading profitability.",
            "Forward returns are not used as model inputs.",
        ],
    }


def save_walk_forward_outputs(
    report: dict[str, Any],
    *,
    reports_dir: Path = REPORTS_DIR,
    save_csv: bool = False,
) -> WalkForwardResult:
    """Persist walk-forward JSON report and optional CSV fold summary."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
        report["created_at"]
        .replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    report_path = reports_dir / f"{WALK_FORWARD_REPORT_PREFIX}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    csv_path = None
    if save_csv:
        csv_path = reports_dir / f"{WALK_FORWARD_REPORT_PREFIX}_{timestamp}_summary.csv"
        pd.DataFrame(
            [flatten_fold_for_csv(fold) for fold in report["folds"]]
        ).to_csv(csv_path, index=False)

    return WalkForwardResult(report_path=report_path, csv_path=csv_path, report=report)


def run_walk_forward_validation(
    tickers: Sequence[str],
    *,
    model_type: str,
    start_year: int,
    test_window_years: int = DEFAULT_TEST_WINDOW_YEARS,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_csv: bool = False,
    created_at: str | None = None,
) -> WalkForwardResult:
    """Load labeled ticker data, evaluate folds, and save report outputs."""
    data = load_labeled_tickers(tickers, processed_data_dir=processed_data_dir)
    report = build_walk_forward_report(
        data,
        model_type=model_type,
        start_year=start_year,
        test_window_years=test_window_years,
        created_at=created_at,
    )
    return save_walk_forward_outputs(
        report,
        reports_dir=reports_dir,
        save_csv=save_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the walk-forward validation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run expanding-window walk-forward validation."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
        help="Model type to evaluate.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="First test-window calendar year.",
    )
    parser.add_argument(
        "--test-window-years",
        type=int,
        default=DEFAULT_TEST_WINDOW_YEARS,
        help="Length of each chronological test window in years.",
    )
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
        help="Directory for walk-forward reports.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save a CSV fold summary under reports/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for walk-forward validation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_walk_forward_validation(
            args.tickers,
            model_type=args.model_type,
            start_year=args.start_year,
            test_window_years=args.test_window_years,
            processed_data_dir=args.processed_data_dir,
            reports_dir=args.reports_dir,
            save_csv=args.save_csv,
        )
    except WalkForwardError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    summary = result.report["summary"]
    print("walk_forward_report")
    print(f"model_type: {result.report['model_type']}")
    print(f"tickers: {result.report['tickers']}")
    print(f"fold_count: {summary['fold_count']}")
    print(f"mean_accuracy: {summary['mean_accuracy']}")
    print(f"mean_macro_f1: {summary['mean_macro_f1']}")
    print(f"report_path: {result.report_path}")
    if result.csv_path is not None:
        print(f"csv_path: {result.csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
