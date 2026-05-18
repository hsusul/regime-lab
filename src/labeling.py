"""Rule-based market regime labeling.

This module implements Milestone 4 only. Labels are deterministic heuristics
used as supervised-learning targets later; they are not objective ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data import validate_ticker
from src.features import PROCESSED_DATA_DIR


REGIME_LABELS = [
    "stable_growth",
    "volatile_recovery",
    "sideways_defensive",
    "stress_selloff",
]
LABELING_VERSION = "v1"
REQUIRED_FEATURE_COLUMNS = {
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma_50_distance",
    "ma_200_distance",
    "drawdown_60d",
}


class LabelingError(Exception):
    """Base exception for labeling failures."""


class MissingFeatureColumnsError(LabelingError):
    """Raised when required feature columns are missing."""


class FeatureDataNotFoundError(LabelingError):
    """Raised when labeled CLI input feature data cannot be found."""


@dataclass(frozen=True)
class LabelingThresholds:
    """Configurable thresholds for rule-based regime labels."""

    stress_return_20d: float = -0.04
    stress_drawdown_60d: float = -0.08
    elevated_volatility_20d: float = 0.025
    recovery_return_5d: float = 0.015
    recovery_return_20d: float = 0.02
    recovery_drawdown_60d: float = -0.04
    stable_return_20d: float = 0.02
    near_ma_distance: float = -0.01
    stable_max_volatility_20d: float = 0.02
    stable_max_drawdown_60d: float = -0.03


@dataclass(frozen=True)
class LabelBuildResult:
    """Result metadata for one labeled feature file."""

    ticker: str
    input_path: Path
    output_path: Path
    rows: int
    label_counts: dict[str, int]


def validate_required_features(data: pd.DataFrame) -> None:
    """Validate that required feature columns exist."""
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(data.columns))
    if missing:
        raise MissingFeatureColumnsError(
            f"Missing required feature columns: {', '.join(missing)}"
        )


def add_rule_labels(
    data: pd.DataFrame,
    thresholds: LabelingThresholds | None = None,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """Add deterministic rule-based regime labels to a feature DataFrame."""
    validate_required_features(data)
    active_thresholds = thresholds or LabelingThresholds()
    result = data.copy()
    result["rule_label"] = pd.NA

    valid_features = result[list(REQUIRED_FEATURE_COLUMNS)].notna().all(axis=1)

    # Rule order is intentional. Stress is checked before recovery because a
    # severe selloff can also have short positive bounces.
    stress = (
        valid_features
        & (result["return_20d"] <= active_thresholds.stress_return_20d)
        & (
            (result["drawdown_60d"] <= active_thresholds.stress_drawdown_60d)
            | (result["ma_50_distance"] < 0)
            | (
                result["volatility_20d"]
                >= active_thresholds.elevated_volatility_20d
            )
        )
    )
    result.loc[stress, "rule_label"] = "stress_selloff"

    # Recovery is checked next: improving returns while still under drawdown
    # pressure or below the long-term moving average.
    unlabeled = valid_features & result["rule_label"].isna()
    volatile_recovery = (
        unlabeled
        & (
            (result["return_5d"] >= active_thresholds.recovery_return_5d)
            | (result["return_20d"] >= active_thresholds.recovery_return_20d)
        )
        & (result["drawdown_60d"] <= active_thresholds.recovery_drawdown_60d)
        & (
            (
                result["volatility_20d"]
                >= active_thresholds.elevated_volatility_20d
            )
            | (result["ma_200_distance"] < 0)
        )
    )
    result.loc[volatile_recovery, "rule_label"] = "volatile_recovery"

    # Stable growth requires positive trend, support from moving averages,
    # contained volatility, and limited drawdown.
    unlabeled = valid_features & result["rule_label"].isna()
    stable_growth = (
        unlabeled
        & (result["return_20d"] >= active_thresholds.stable_return_20d)
        & (result["ma_50_distance"] >= active_thresholds.near_ma_distance)
        & (result["ma_200_distance"] >= active_thresholds.near_ma_distance)
        & (
            result["volatility_20d"]
            <= active_thresholds.stable_max_volatility_20d
        )
        & (result["drawdown_60d"] >= active_thresholds.stable_max_drawdown_60d)
    )
    result.loc[stable_growth, "rule_label"] = "stable_growth"

    # Sideways/defensive is the default for valid rows that do not meet one of
    # the stronger directional regimes above.
    unlabeled = valid_features & result["rule_label"].isna()
    result.loc[unlabeled, "rule_label"] = "sideways_defensive"

    if drop_unlabeled:
        result = result.dropna(subset=["rule_label"]).reset_index(drop=True)
    return result


def find_feature_file(
    ticker: str,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Find the newest unlabeled processed feature CSV for a ticker."""
    normalized_ticker = validate_ticker(ticker)
    pattern = f"{normalized_ticker.replace('/', '-')}*_features_*.csv"
    matches = [
        path
        for path in processed_data_dir.glob(pattern)
        if "_labeled_" not in path.stem
    ]
    matches = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FeatureDataNotFoundError(
            f"No processed feature files found for {normalized_ticker} "
            f"in {processed_data_dir}."
        )
    return matches[0]


def labeled_path_for_features(
    feature_path: Path,
    labeling_version: str = LABELING_VERSION,
) -> Path:
    """Return the labeled output path for a feature CSV."""
    return feature_path.with_name(f"{feature_path.stem}_labeled_{labeling_version}.csv")


def label_features_for_ticker(
    ticker: str,
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    thresholds: LabelingThresholds | None = None,
    drop_unlabeled: bool = True,
) -> LabelBuildResult:
    """Load processed features for one ticker, add rule labels, and save CSV."""
    normalized_ticker = validate_ticker(ticker)
    feature_path = find_feature_file(normalized_ticker, processed_data_dir)
    features = pd.read_csv(feature_path)
    labeled = add_rule_labels(
        features,
        thresholds=thresholds,
        drop_unlabeled=drop_unlabeled,
    )
    output_path = labeled_path_for_features(feature_path)
    labeled.to_csv(output_path, index=False)
    counts = {
        label: int(count)
        for label, count in labeled["rule_label"].value_counts().sort_index().items()
    }
    return LabelBuildResult(
        ticker=normalized_ticker,
        input_path=feature_path,
        output_path=output_path,
        rows=len(labeled),
        label_counts=counts,
    )


def label_features_for_tickers(
    tickers: Sequence[str],
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    thresholds: LabelingThresholds | None = None,
    drop_unlabeled: bool = True,
) -> list[LabelBuildResult]:
    """Label feature CSVs for multiple tickers."""
    return [
        label_features_for_ticker(
            ticker,
            processed_data_dir=processed_data_dir,
            thresholds=thresholds,
            drop_unlabeled=drop_unlabeled,
        )
        for ticker in tickers
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the labeling CLI parser."""
    parser = argparse.ArgumentParser(
        description="Add deterministic rule-based regime labels to feature CSVs."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing processed feature CSV files.",
    )
    parser.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep rows with missing required feature values.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for rule-based labeling."""
    parser = build_parser()
    args = parser.parse_args(argv)

    results = label_features_for_tickers(
        args.tickers,
        processed_data_dir=args.processed_data_dir,
        drop_unlabeled=not args.keep_unlabeled,
    )

    for result in results:
        counts = ", ".join(
            f"{label}={count}" for label, count in result.label_counts.items()
        )
        print(
            f"{result.ticker}: labeled {result.rows} rows -> "
            f"{result.output_path} ({counts})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
