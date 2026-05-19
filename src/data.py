"""Data ingestion utilities for RegimeLab.

This module implements the Milestone 2 data-loading boundary only. It does not
perform feature engineering, labeling, training, or prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from src.config import cli_or_config, cli_or_config_path, load_config


RAW_DATA_DIR = Path("data/raw")
NORMALIZED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "ticker",
]
REQUIRED_COLUMNS = set(NORMALIZED_COLUMNS)


class DataIngestionError(Exception):
    """Base exception for data ingestion failures."""


class InvalidTickerError(DataIngestionError):
    """Raised when a ticker symbol is empty or malformed."""


class EmptyDataError(DataIngestionError):
    """Raised when a provider returns no rows for a request."""


class MissingColumnsError(DataIngestionError):
    """Raised when required OHLCV columns are missing."""


class ProviderError(DataIngestionError):
    """Raised when the upstream data provider fails."""


class MarketDataProvider(Protocol):
    """Protocol implemented by market data providers."""

    def download(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Download daily OHLCV data for one ticker."""


@dataclass(frozen=True)
class DataLoadResult:
    """Result metadata for one loaded ticker dataset."""

    ticker: str
    data: pd.DataFrame
    cache_path: Path
    source: str

    @property
    def rows(self) -> int:
        """Number of rows in the returned dataset."""
        return len(self.data)


class YFinanceProvider:
    """Market data provider backed by yfinance."""

    def download(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf

            return yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # pragma: no cover - exercised with fake provider
            raise ProviderError(f"Provider failed for ticker {ticker}: {exc}") from exc


def validate_ticker(ticker: str) -> str:
    """Validate and normalize a ticker string."""
    normalized = ticker.strip().upper()
    if not normalized:
        raise InvalidTickerError("Ticker must not be empty.")
    return normalized


def cache_path_for_ticker(
    ticker: str,
    start_date: str | None,
    end_date: str | None,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Return the raw CSV cache path for a ticker/date range."""
    safe_ticker = validate_ticker(ticker).replace("/", "-")
    start = start_date or "start"
    end = end_date or "end"
    return raw_data_dir / f"{safe_ticker}_{start}_{end}.csv"


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance-style MultiIndex columns if present."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    flattened = data.copy()
    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in flattened.columns.to_flat_index()
    ]
    return flattened


def _normalize_column_name(column: object) -> str:
    """Normalize provider column names to lower snake case."""
    name = str(column).strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    if name in {"adj_close", "adjusted_close"}:
        return "adj_close"
    return name


def normalize_ohlcv(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize raw provider OHLCV data to the RegimeLab schema."""
    normalized_ticker = validate_ticker(ticker)

    if data.empty:
        raise EmptyDataError(f"No data returned for ticker {normalized_ticker}.")

    frame = _flatten_columns(data).copy()
    if isinstance(frame.index, pd.DatetimeIndex) and "date" not in frame.columns:
        frame = frame.reset_index()

    frame = frame.rename(columns={column: _normalize_column_name(column) for column in frame.columns})
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})

    # yfinance may return MultiIndex columns for one ticker in newer versions,
    # which flatten to names such as close_spy or spy_close.
    rename_map: dict[str, str] = {}
    for required in ["open", "high", "low", "close", "adj_close", "volume"]:
        if required in frame.columns:
            continue
        candidates = [
            column
            for column in frame.columns
            if column.startswith(f"{required}_") or column.endswith(f"_{required}")
        ]
        if candidates:
            rename_map[candidates[0]] = required
    if rename_map:
        frame = frame.rename(columns=rename_map)

    frame["ticker"] = normalized_ticker
    validate_required_columns(frame)

    result = frame[NORMALIZED_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    result = result.sort_values("date").reset_index(drop=True)
    return result


def validate_required_columns(data: pd.DataFrame) -> None:
    """Validate that normalized OHLCV data contains all required columns."""
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise MissingColumnsError(f"Missing required columns: {', '.join(missing)}")


def load_ticker_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    provider: MarketDataProvider | None = None,
    raw_data_dir: Path = RAW_DATA_DIR,
    force_refresh: bool = False,
) -> DataLoadResult:
    """Load one ticker from cache or provider and return normalized OHLCV data."""
    normalized_ticker = validate_ticker(ticker)
    cache_path = cache_path_for_ticker(
        normalized_ticker,
        start_date,
        end_date,
        raw_data_dir,
    )

    if cache_path.exists() and not force_refresh:
        cached = pd.read_csv(cache_path)
        normalized = normalize_ohlcv(cached, normalized_ticker)
        return DataLoadResult(
            ticker=normalized_ticker,
            data=normalized,
            cache_path=cache_path,
            source="cache",
        )

    selected_provider = provider or YFinanceProvider()
    try:
        raw = selected_provider.download(normalized_ticker, start_date, end_date)
    except DataIngestionError:
        raise
    except Exception as exc:
        raise ProviderError(
            f"Provider failed for ticker {normalized_ticker}: {exc}"
        ) from exc

    normalized = normalize_ohlcv(raw, normalized_ticker)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(cache_path, index=False)
    return DataLoadResult(
        ticker=normalized_ticker,
        data=normalized,
        cache_path=cache_path,
        source="download",
    )


def load_tickers(
    tickers: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    provider: MarketDataProvider | None = None,
    raw_data_dir: Path = RAW_DATA_DIR,
    force_refresh: bool = False,
) -> list[DataLoadResult]:
    """Load multiple tickers from cache or provider."""
    return [
        load_ticker_data(
            ticker,
            start_date,
            end_date,
            provider=provider,
            raw_data_dir=raw_data_dir,
            force_refresh=force_refresh,
        )
        for ticker in tickers
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the data ingestion CLI argument parser."""
    parser = argparse.ArgumentParser(description="Download and cache daily OHLCV data.")
    parser.add_argument("--config", type=Path, help="Optional YAML config file.")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Directory for cached raw CSV files.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Download even when a cache file already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for data ingestion."""
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

    results = load_tickers(
        tickers,
        cli_or_config(args.start_date, config, "data.start_date", None),
        cli_or_config(args.end_date, config, "data.end_date", None),
        raw_data_dir=cli_or_config_path(
            args.raw_data_dir,
            config,
            "paths.raw_data_dir",
            RAW_DATA_DIR,
        ),
        force_refresh=args.force_refresh,
    )

    for result in results:
        print(
            f"{result.ticker}: {result.source} {result.rows} rows -> "
            f"{result.cache_path}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
