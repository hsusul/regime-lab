"""Retrospective forward return analysis for RegimeLab.

Forward returns are analysis-only outputs. They are not model features, are not
part of saved supervised artifacts, and must not be used for training or
prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.features import FEATURE_VERSION, PROCESSED_DATA_DIR
from src.predict import (
    NoModelAvailableError,
    latest_model_artifact,
)
from src.train import REPORTS_DIR, TRAINING_TARGET_COLUMN, find_labeled_feature_file


DEFAULT_FORWARD_HORIZONS = [5, 20, 60]
PRICE_COLUMN = "adj_close"
PREDICTED_REGIME_COLUMN = "predicted_regime"
FORWARD_REPORT_PREFIX = "forward_returns"
REQUIRED_FORWARD_COLUMNS = {"date", "ticker", PRICE_COLUMN, TRAINING_TARGET_COLUMN}


class ForwardReturnError(Exception):
    """Base exception for forward return analysis failures."""


class MissingForwardColumnsError(ForwardReturnError):
    """Raised when labeled processed data is missing required columns."""


class InvalidForwardHorizonError(ForwardReturnError):
    """Raised when a forward return horizon is invalid."""


class ForwardDataNotFoundError(ForwardReturnError):
    """Raised when labeled processed files cannot be found."""


@dataclass(frozen=True)
class ForwardReturnResult:
    """Paths and report metadata from a forward return analysis run."""

    report_path: Path
    csv_path: Path | None
    report: dict[str, Any]


def validate_horizons(horizons: Sequence[int]) -> list[int]:
    """Validate forward return horizons and return a sorted unique list."""
    cleaned = sorted({int(horizon) for horizon in horizons})
    if not cleaned or any(horizon <= 0 for horizon in cleaned):
        raise InvalidForwardHorizonError("Forward horizons must be positive integers.")
    return cleaned


def forward_return_column(horizon: int) -> str:
    """Return the analysis-only forward return column name for a horizon."""
    return f"forward_return_{horizon}d"


def validate_forward_columns(data: pd.DataFrame) -> None:
    """Validate required labeled processed columns."""
    missing = sorted(REQUIRED_FORWARD_COLUMNS - set(data.columns))
    if missing:
        raise MissingForwardColumnsError(
            f"Missing required forward analysis columns: {', '.join(missing)}"
        )


def prepare_forward_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and sort labeled data for retrospective analysis."""
    validate_forward_columns(data)
    prepared = data.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared[PRICE_COLUMN] = pd.to_numeric(prepared[PRICE_COLUMN], errors="coerce")
    prepared = prepared.dropna(subset=[PRICE_COLUMN])
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prepared


def add_forward_returns(data: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Add analysis-only forward returns by ticker using future prices."""
    validated_horizons = validate_horizons(horizons)
    prepared = prepare_forward_data(data)

    for horizon in validated_horizons:
        column = forward_return_column(horizon)
        prepared[column] = prepared.groupby("ticker", sort=False)[PRICE_COLUMN].transform(
            lambda prices: prices.shift(-horizon) / prices - 1
        )
    return prepared


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def summarize_forward_series(series: pd.Series) -> dict[str, float | int | None]:
    """Summarize one forward return series."""
    returns = pd.to_numeric(series, errors="coerce").dropna()
    valid_count = int(len(returns))
    return {
        "valid_count": valid_count,
        "average_forward_return": _safe_float(returns.mean()),
        "median_forward_return": _safe_float(returns.median()),
        "forward_volatility": _safe_float(returns.std()),
        "worst_forward_return": _safe_float(returns.min()),
        "best_forward_return": _safe_float(returns.max()),
        "hit_rate": None
        if valid_count == 0
        else float((returns > 0).sum() / valid_count),
    }


def summarize_by_regime(
    data: pd.DataFrame,
    *,
    group_column: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Summarize forward outcomes by rule or predicted regime."""
    if group_column not in data.columns:
        return {}

    summaries: dict[str, Any] = {}
    for regime, group in data.dropna(subset=[group_column]).groupby(group_column):
        regime_key = str(regime)
        summaries[regime_key] = {
            "observation_count": int(len(group)),
            "horizons": {
                str(horizon): summarize_forward_series(
                    group[forward_return_column(horizon)]
                )
                for horizon in horizons
            },
        }
    return summaries


def add_predicted_regimes(
    data: pd.DataFrame,
    *,
    reports_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any] | None, list[str]]:
    """Attach latest model predictions when a valid supervised artifact exists."""
    analyzed = data.copy()
    analyzed[PREDICTED_REGIME_COLUMN] = pd.NA
    warnings: list[str] = []

    try:
        artifact = latest_model_artifact(reports_dir)
    except NoModelAvailableError:
        warnings.append("no valid trained model available; predicted regimes skipped.")
        return analyzed, None, warnings

    feature_columns = list(artifact["feature_columns"])
    missing_features = sorted(set(feature_columns) - set(analyzed.columns))
    if missing_features:
        warnings.append(
            "predicted regimes skipped; data missing model feature columns: "
            + ", ".join(missing_features)
        )
        return analyzed, artifact, warnings

    complete = analyzed[feature_columns].notna().all(axis=1)
    if complete.any():
        analyzed.loc[complete, PREDICTED_REGIME_COLUMN] = artifact["model"].predict(
            analyzed.loc[complete, feature_columns]
        )
    skipped = int((~complete).sum())
    if skipped:
        warnings.append(
            f"predicted regimes skipped for {skipped} rows with missing features."
        )
    return analyzed, artifact, warnings


def _date_range(data: pd.DataFrame) -> dict[str, str] | None:
    if data.empty:
        return None
    dates = pd.to_datetime(data["date"])
    return {
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
    }


def build_group_summaries(
    data: pd.DataFrame,
    *,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Build combined and per-ticker summaries for rule and predicted regimes."""
    summaries: dict[str, Any] = {
        "combined": {
            TRAINING_TARGET_COLUMN: summarize_by_regime(
                data,
                group_column=TRAINING_TARGET_COLUMN,
                horizons=horizons,
            ),
            PREDICTED_REGIME_COLUMN: summarize_by_regime(
                data,
                group_column=PREDICTED_REGIME_COLUMN,
                horizons=horizons,
            ),
        },
        "by_ticker": {},
    }

    for ticker, group in data.groupby("ticker", sort=True):
        summaries["by_ticker"][str(ticker)] = {
            "row_count": int(len(group)),
            "date_range": _date_range(group),
            TRAINING_TARGET_COLUMN: summarize_by_regime(
                group,
                group_column=TRAINING_TARGET_COLUMN,
                horizons=horizons,
            ),
            PREDICTED_REGIME_COLUMN: summarize_by_regime(
                group,
                group_column=PREDICTED_REGIME_COLUMN,
                horizons=horizons,
            ),
        }
    return summaries


def _latest_model_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if artifact is None:
        return {
            "available": False,
            "experiment_id": None,
            "model_type": None,
            "feature_columns": None,
        }
    return {
        "available": True,
        "experiment_id": artifact.get("experiment_id"),
        "model_type": artifact.get("model_type"),
        "feature_columns": artifact.get("feature_columns"),
    }


def build_forward_return_report(
    data: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
    reports_dir: Path = REPORTS_DIR,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a retrospective forward return report from labeled feature data."""
    validated_horizons = validate_horizons(horizons)
    analyzed = add_forward_returns(data, validated_horizons)
    analyzed, artifact, warnings = add_predicted_regimes(
        analyzed,
        reports_dir=reports_dir,
    )
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="microseconds")

    report = {
        "analysis_type": "forward_return_regime_analysis",
        "created_at": timestamp,
        "tickers": sorted(analyzed["ticker"].unique().tolist()),
        "rows": int(len(analyzed)),
        "date_range": _date_range(analyzed),
        "horizons": validated_horizons,
        "price_column": PRICE_COLUMN,
        "forward_return_columns": [
            forward_return_column(horizon) for horizon in validated_horizons
        ],
        "feature_version": FEATURE_VERSION,
        "latest_model": _latest_model_summary(artifact),
        "summaries": build_group_summaries(analyzed, horizons=validated_horizons),
        "leakage_note": (
            "Forward returns are retrospective analysis-only columns and are not "
            "included in FEATURE_COLUMNS, model inputs, or model artifacts."
        ),
        "warnings": [
            *warnings,
            "Forward return analysis is retrospective research, not financial advice.",
        ],
    }
    return report


def flatten_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten report summaries into CSV-friendly rows."""
    rows: list[dict[str, Any]] = []

    def append_group_rows(
        *,
        scope: str,
        ticker: str | None,
        group_column: str,
        regime_summaries: dict[str, Any],
    ) -> None:
        for regime, regime_summary in regime_summaries.items():
            observation_count = regime_summary["observation_count"]
            for horizon, stats in regime_summary["horizons"].items():
                rows.append(
                    {
                        "scope": scope,
                        "ticker": ticker,
                        "group_column": group_column,
                        "regime": regime,
                        "horizon": int(horizon),
                        "observation_count": observation_count,
                        **stats,
                    }
                )

    combined = report["summaries"]["combined"]
    for group_column, regime_summaries in combined.items():
        append_group_rows(
            scope="combined",
            ticker=None,
            group_column=group_column,
            regime_summaries=regime_summaries,
        )

    for ticker, ticker_summary in report["summaries"]["by_ticker"].items():
        for group_column in [TRAINING_TARGET_COLUMN, PREDICTED_REGIME_COLUMN]:
            append_group_rows(
                scope="ticker",
                ticker=ticker,
                group_column=group_column,
                regime_summaries=ticker_summary[group_column],
            )
    return rows


def save_forward_return_outputs(
    report: dict[str, Any],
    *,
    reports_dir: Path = REPORTS_DIR,
    save_csv: bool = False,
) -> ForwardReturnResult:
    """Persist forward return report JSON and optional flattened CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
        report["created_at"]
        .replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    report_path = reports_dir / f"{FORWARD_REPORT_PREFIX}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    csv_path = None
    if save_csv:
        csv_path = reports_dir / f"{FORWARD_REPORT_PREFIX}_{timestamp}_summary.csv"
        pd.DataFrame(flatten_summary_rows(report)).to_csv(csv_path, index=False)

    return ForwardReturnResult(report_path=report_path, csv_path=csv_path, report=report)


def load_labeled_tickers(
    tickers: Sequence[str],
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load newest labeled processed feature files for selected tickers."""
    frames = []
    for ticker in tickers:
        try:
            path = find_labeled_feature_file(ticker, processed_data_dir)
        except Exception as exc:
            raise ForwardDataNotFoundError(
                f"No labeled processed data found for {ticker} in {processed_data_dir}."
            ) from exc
        frames.append(pd.read_csv(path))
    if not frames:
        raise ForwardDataNotFoundError("At least one ticker is required.")
    return pd.concat(frames, ignore_index=True)


def run_forward_return_analysis(
    tickers: Sequence[str],
    *,
    horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_csv: bool = False,
    created_at: str | None = None,
) -> ForwardReturnResult:
    """Load labeled data, build report, and save outputs."""
    data = load_labeled_tickers(tickers, processed_data_dir=processed_data_dir)
    report = build_forward_return_report(
        data,
        horizons=horizons,
        reports_dir=reports_dir,
        created_at=created_at,
    )
    return save_forward_return_outputs(
        report,
        reports_dir=reports_dir,
        save_csv=save_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the forward return analysis CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run retrospective forward return analysis by regime."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=DEFAULT_FORWARD_HORIZONS,
        help="Forward return horizons in trading days.",
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
        help="Directory for forward return reports.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save a flattened CSV summary table under reports/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for forward return analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_forward_return_analysis(
            args.tickers,
            horizons=args.horizons,
            processed_data_dir=args.processed_data_dir,
            reports_dir=args.reports_dir,
            save_csv=args.save_csv,
        )
    except ForwardReturnError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    report = result.report
    print("forward_return_report")
    print(f"tickers: {report['tickers']}")
    print(f"horizons: {report['horizons']}")
    print(f"rows: {report['rows']}")
    print(f"latest_model_available: {report['latest_model']['available']}")
    print(f"report_path: {result.report_path}")
    if result.csv_path is not None:
        print(f"csv_path: {result.csv_path}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
