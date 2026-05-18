"""Tests for data ingestion."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.data import (
    EmptyDataError,
    MissingColumnsError,
    ProviderError,
    build_parser,
    cache_path_for_ticker,
    load_ticker_data,
    load_tickers,
    normalize_ohlcv,
)
from src.features import RawDataNotFoundError, find_raw_cache_file


def sample_provider_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.5, 102.5],
            "Adj Close": [101.25, 102.25],
            "Volume": [1_000_000, 1_100_000],
        }
    )


@dataclass
class FakeProvider:
    data: pd.DataFrame
    calls: int = 0

    def download(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self.calls += 1
        return self.data.copy()


class FailingProvider:
    def download(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        raise RuntimeError("provider unavailable")


def test_normalize_ohlcv_columns() -> None:
    normalized = normalize_ohlcv(sample_provider_frame(), "spy")

    assert list(normalized.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ticker",
    ]
    assert normalized["ticker"].tolist() == ["SPY", "SPY"]
    assert normalized["date"].tolist() == ["2024-01-02", "2024-01-03"]


def test_normalize_ohlcv_handles_yfinance_ticker_suffixed_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open_SPY": [100.0, 101.0],
            "High_SPY": [102.0, 103.0],
            "Low_SPY": [99.0, 100.0],
            "Close_SPY": [101.5, 102.5],
            "Adj Close_SPY": [101.25, 102.25],
            "Volume_SPY": [1_000_000, 1_100_000],
        }
    )

    normalized = normalize_ohlcv(raw, "SPY")

    assert list(normalized.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ticker",
    ]
    assert normalized["close"].tolist() == [101.5, 102.5]


def test_validate_required_columns_rejects_missing_ohlcv_column() -> None:
    missing_volume = sample_provider_frame().drop(columns=["Volume"])

    with pytest.raises(MissingColumnsError, match="volume"):
        normalize_ohlcv(missing_volume, "SPY")


def test_load_ticker_data_creates_cache_then_loads_from_cache(tmp_path) -> None:
    provider = FakeProvider(sample_provider_frame())

    first = load_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-31",
        provider=provider,
        raw_data_dir=tmp_path,
    )
    second = load_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-31",
        provider=provider,
        raw_data_dir=tmp_path,
    )

    assert first.source == "download"
    assert second.source == "cache"
    assert provider.calls == 1
    assert first.cache_path == tmp_path / "SPY_2024-01-01_2024-01-31.csv"
    assert first.cache_path.exists()
    pd.testing.assert_frame_equal(first.data, second.data)


def test_force_refresh_ignores_existing_cache(tmp_path) -> None:
    provider = FakeProvider(sample_provider_frame())

    load_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-31",
        provider=provider,
        raw_data_dir=tmp_path,
    )
    refreshed = load_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-31",
        provider=provider,
        raw_data_dir=tmp_path,
        force_refresh=True,
    )

    assert refreshed.source == "download"
    assert provider.calls == 2


def test_empty_provider_data_raises_empty_data_error(tmp_path) -> None:
    provider = FakeProvider(pd.DataFrame())

    with pytest.raises(EmptyDataError, match="No data returned"):
        load_ticker_data("SPY", provider=provider, raw_data_dir=tmp_path)


def test_provider_failure_is_wrapped(tmp_path) -> None:
    with pytest.raises(ProviderError, match="Provider failed"):
        load_ticker_data("SPY", provider=FailingProvider(), raw_data_dir=tmp_path)


def test_cache_path_uses_safe_ticker_name(tmp_path) -> None:
    path = cache_path_for_ticker("BRK/B", "2024-01-01", "2024-01-31", tmp_path)

    assert path == tmp_path / "BRK-B_2024-01-01_2024-01-31.csv"


def test_cli_argument_parsing() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--tickers",
            "SPY",
            "QQQ",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
            "--force-refresh",
        ]
    )

    assert args.tickers == ["SPY", "QQQ"]
    assert args.start_date == "2024-01-01"
    assert args.end_date == "2024-01-31"
    assert args.force_refresh is True


def test_load_tickers_returns_one_result_per_ticker(tmp_path) -> None:
    provider = FakeProvider(sample_provider_frame())

    results = load_tickers(
        ["SPY", "QQQ"],
        "2024-01-01",
        "2024-01-31",
        provider=provider,
        raw_data_dir=tmp_path,
    )

    assert [result.ticker for result in results] == ["SPY", "QQQ"]
    assert [result.source for result in results] == ["download", "download"]
    assert provider.calls == 2


def test_unknown_ticker_raw_cache_file_not_found(tmp_path) -> None:
    with pytest.raises(RawDataNotFoundError):
        find_raw_cache_file("MSFT", raw_data_dir=tmp_path)
