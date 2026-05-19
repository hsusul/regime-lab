"""Lightweight model explanation reports for RegimeLab.

This module is an optional reporting path. It does not modify training,
evaluation, API behavior, model artifacts, or experiment registry state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.features import PROCESSED_DATA_DIR
from src.predict import (
    KEY_SIGNAL_COLUMNS,
    CachedDataNotFoundError,
    latest_model_artifact,
    load_feature_data_for_ticker,
    validate_api_ticker,
)
from src.train import REPORTS_DIR


EXPLAIN_REPORT_PREFIX = "explain"


class ExplanationError(Exception):
    """Base exception for explanation report failures."""


class MissingExplanationFeaturesError(ExplanationError):
    """Raised when cached feature rows cannot satisfy artifact feature columns."""


@dataclass(frozen=True)
class ExplanationResult:
    """Saved explanation report metadata."""

    report_path: Path
    report: dict[str, Any]


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


def feature_importance_for_artifact(artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return global model explanation data and warnings."""
    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])
    model_type = str(artifact.get("model_type"))

    if hasattr(model, "feature_importances_"):
        importances = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(
                feature_columns,
                model.feature_importances_,
                strict=True,
            )
        ]
        return (
            sorted(importances, key=lambda item: item["importance"], reverse=True),
            [],
        )

    classifier = None
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
    if classifier is None:
        classifier = model

    if model_type == "logistic_regression" and hasattr(classifier, "coef_"):
        classes = [str(label) for label in classifier.classes_]
        coefficients = np.asarray(classifier.coef_)
        if len(classes) == 2 and coefficients.shape[0] == 1:
            classes = [classes[-1]]
        rows: list[dict[str, Any]] = []
        for class_label, class_coefficients in zip(classes, coefficients, strict=True):
            ranked = [
                {"feature": feature, "coefficient": float(coefficient)}
                for feature, coefficient in zip(
                    feature_columns,
                    class_coefficients,
                    strict=True,
                )
            ]
            rows.append(
                {
                    "class_label": class_label,
                    "coefficients": sorted(
                        ranked,
                        key=lambda item: abs(float(item["coefficient"])),
                        reverse=True,
                    ),
                }
            )
        return rows, [
            "logistic_regression coefficients are class-specific and should be "
            "interpreted with the fitted scaling pipeline."
        ]

    return [], [
        f"global feature importance unavailable for model_type={model_type}."
    ]


def latest_explanation_for_ticker(
    ticker: str,
    artifact: dict[str, Any],
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Any]:
    """Build one latest-row prediction explanation for a ticker."""
    normalized = validate_api_ticker(ticker)
    features = load_feature_data_for_ticker(
        normalized,
        processed_data_dir=processed_data_dir,
    )
    feature_columns = list(artifact["feature_columns"])
    missing = sorted(set(feature_columns) - set(features.columns))
    if missing:
        raise MissingExplanationFeaturesError(
            f"Cached feature data for {normalized} missing model inputs: "
            + ", ".join(missing)
        )

    usable = features.dropna(subset=feature_columns).sort_values("date")
    if usable.empty:
        raise CachedDataNotFoundError(
            f"No complete feature rows available for {normalized}."
        )
    latest = usable.iloc[-1]
    feature_row = latest[feature_columns].to_frame().T
    predicted_regime = str(artifact["model"].predict(feature_row)[0])
    confidence, probabilities = _probabilities_for_model(artifact, feature_row)

    return {
        "ticker": normalized,
        "as_of": str(latest["date"]),
        "predicted_regime": predicted_regime,
        "confidence": confidence,
        "probabilities": probabilities,
        "key_feature_values": {
            column: _safe_float(latest.get(column)) for column in KEY_SIGNAL_COLUMNS
        },
    }


def build_explanation_report(
    tickers: Sequence[str],
    *,
    reports_dir: Path = REPORTS_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build an explanation report for selected latest ticker rows."""
    artifact = latest_model_artifact(reports_dir)
    explanations = [
        latest_explanation_for_ticker(
            ticker,
            artifact,
            processed_data_dir=processed_data_dir,
        )
        for ticker in tickers
    ]
    global_importance, warnings = feature_importance_for_artifact(artifact)
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="microseconds")

    return {
        "analysis_type": "model_explanation",
        "created_at": timestamp,
        "experiment_id": artifact["experiment_id"],
        "model_type": artifact["model_type"],
        "tickers": [explanation["ticker"] for explanation in explanations],
        "feature_columns": list(artifact["feature_columns"]),
        "label_names": list(artifact["label_names"]),
        "predictions": explanations,
        "global_feature_importance": global_importance,
        "warnings": [
            *warnings,
            "Explanations are descriptive model diagnostics, not financial advice.",
            "SHAP explanations are intentionally not included in this lightweight report.",
        ],
    }


def save_explanation_report(
    report: dict[str, Any],
    *,
    reports_dir: Path = REPORTS_DIR,
) -> ExplanationResult:
    """Persist explanation report JSON under reports/."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
        report["created_at"]
        .replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    report_path = reports_dir / f"{EXPLAIN_REPORT_PREFIX}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return ExplanationResult(report_path=report_path, report=report)


def run_explanation_report(
    tickers: Sequence[str],
    *,
    reports_dir: Path = REPORTS_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    created_at: str | None = None,
) -> ExplanationResult:
    """Build and save an explanation report."""
    report = build_explanation_report(
        tickers,
        reports_dir=reports_dir,
        processed_data_dir=processed_data_dir,
        created_at=created_at,
    )
    return save_explanation_report(report, reports_dir=reports_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the explanation report CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a lightweight explanation report for latest predictions."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory containing experiments and model artifacts.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing cached processed feature data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for explanation reports."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_explanation_report(
            args.tickers,
            reports_dir=args.reports_dir,
            processed_data_dir=args.processed_data_dir,
        )
    except (ExplanationError, CachedDataNotFoundError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print("explanation_report")
    print(f"experiment_id: {result.report['experiment_id']}")
    print(f"model_type: {result.report['model_type']}")
    print(f"tickers: {result.report['tickers']}")
    print(f"report_path: {result.report_path}")
    if result.report["warnings"]:
        print("warnings:")
        for warning in result.report["warnings"]:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
