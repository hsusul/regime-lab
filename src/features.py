"""Feature engineering for normalized OHLCV data.

This module implements Milestone 3 only. It computes leakage-aware features
from current and prior observations and does not create labels, train models,
or generate predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import cli_or_config, cli_or_config_path, load_config
from src.data import NORMALIZED_COLUMNS, RAW_DATA_DIR, cache_path_for_ticker, validate_ticker


PROCESSED_DATA_DIR = Path("data/processed")
FEATURE_VERSION = "v1"
IDENTIFIER_COLUMNS = ["date", "ticker"]
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma_50_distance",
    "ma_200_distance",
    "drawdown_60d",
    "volume_change_20d",
]
REQUIRED_INPUT_COLUMNS = set(NORMALIZED_COLUMNS)


class FeatureEngineeringError(Exception):
    """Base exception for feature engineering failures."""


class MissingInputColumnsError(FeatureEngineeringError):
    """Raised when normalized OHLCV input columns are missing."""


class RawDataNotFoundError(FeatureEngineeringError):
    """Raised when the feature CLI cannot find cached raw data."""


@dataclass(frozen=True)
class FeatureBuildResult:
    """Result metadata for one processed feature file."""

    ticker: str
    input_path: Path
    output_path: Path
    rows: int


def validate_input_columns(data: pd.DataFrame) -> None:
    """Validate that input data has the normalized OHLCV schema."""
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(data.columns))
    if missing:
        raise MissingInputColumnsError(
            f"Missing required input columns: {', '.join(missing)}"
        )


def _compute_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute features for one ticker group using only current/prior rows."""
    result = group.sort_values("date").copy()
    adj_close = result["adj_close"].astype(float)
    volume = result["volume"].astype(float)

    daily_return = adj_close.pct_change(1)
    result["return_1d"] = daily_return
    result["return_5d"] = adj_close.pct_change(5)
    result["return_20d"] = adj_close.pct_change(20)
    result["volatility_20d"] = daily_return.rolling(window=20).std()
    result["ma_50_distance"] = adj_close / adj_close.rolling(window=50).mean() - 1
    result["ma_200_distance"] = adj_close / adj_close.rolling(window=200).mean() - 1
    result["drawdown_60d"] = adj_close / adj_close.rolling(window=60).max() - 1
    result["volume_change_20d"] = volume / volume.rolling(window=20).mean() - 1
    return result


def build_features(data: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Build leakage-aware time-series features from normalized OHLCV data."""
    validate_input_columns(data)
    frame = data.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_frames = [
        _compute_ticker_features(group)
        for _, group in frame.groupby("ticker", sort=False)
    ]
    featured = pd.concat(ticker_frames, ignore_index=True)

    if dropna:
        featured = featured.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    featured["date"] = featured["date"].dt.date.astype(str)
    output_columns = NORMALIZED_COLUMNS + FEATURE_COLUMNS
    return featured[output_columns]


def find_raw_cache_file(
    ticker: str,
    raw_data_dir: Path = RAW_DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """Find a cached raw CSV for a ticker."""
    normalized_ticker = validate_ticker(ticker)
    if start_date or end_date:
        path = cache_path_for_ticker(
            normalized_ticker,
            start_date,
            end_date,
            raw_data_dir,
        )
        if path.exists():
            return path
        raise RawDataNotFoundError(f"No raw cache file found at {path}.")

    pattern = f"{normalized_ticker.replace('/', '-')}*.csv"
    matches = sorted(
        raw_data_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise RawDataNotFoundError(
            f"No raw cache files found for {normalized_ticker} in {raw_data_dir}."
        )
    return matches[0]


def processed_path_for_raw(
    raw_path: Path,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    feature_version: str = FEATURE_VERSION,
) -> Path:
    """Return the processed feature CSV path for a raw cache file."""
    return processed_data_dir / f"{raw_path.stem}_features_{feature_version}.csv"


def build_features_for_ticker(
    ticker: str,
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
    dropna: bool = True,
) -> FeatureBuildResult:
    """Load cached raw data for one ticker and save processed features."""
    normalized_ticker = validate_ticker(ticker)
    raw_path = find_raw_cache_file(
        normalized_ticker,
        raw_data_dir=raw_data_dir,
        start_date=start_date,
        end_date=end_date,
    )
    raw = pd.read_csv(raw_path)
    features = build_features(raw, dropna=dropna)
    output_path = processed_path_for_raw(raw_path, processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return FeatureBuildResult(
        ticker=normalized_ticker,
        input_path=raw_path,
        output_path=output_path,
        rows=len(features),
    )


def build_features_for_tickers(
    tickers: Sequence[str],
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
    dropna: bool = True,
) -> list[FeatureBuildResult]:
    """Build and save features for multiple tickers."""
    return [
        build_features_for_ticker(
            ticker,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
        )
        for ticker in tickers
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the feature engineering CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build leakage-aware features from cached raw OHLCV data."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML config file.")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols.")
    parser.add_argument("--start-date", help="Optional raw cache start date.")
    parser.add_argument("--end-date", help="Optional raw cache end date.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Directory containing cached raw CSV files.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=None,
        help="Directory for processed feature CSV files.",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows without enough lookback history instead of dropping them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for feature engineering."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    tickers = args.tickers or cli_or_config(
        None,
        config,
        "project.supported_tickers",
        None,
    )
    if not tickers:
        parser.error("--tickers is required unless provided by --config")

    results = build_features_for_tickers(
        tickers,
        raw_data_dir=cli_or_config_path(
            args.raw_data_dir,
            config,
            "paths.raw_data_dir",
            RAW_DATA_DIR,
        ),
        processed_data_dir=cli_or_config_path(
            args.processed_data_dir,
            config,
            "paths.processed_data_dir",
            PROCESSED_DATA_DIR,
        ),
        start_date=cli_or_config(args.start_date, config, "data.start_date", None),
        end_date=cli_or_config(args.end_date, config, "data.end_date", None),
        dropna=not args.keep_na,
    )

    for result in results:
        print(
            f"{result.ticker}: built {result.rows} feature rows from "
            f"{result.input_path} -> {result.output_path}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
