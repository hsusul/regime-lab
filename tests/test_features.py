"""Tests for leakage-aware feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    MissingInputColumnsError,
    build_features,
    build_features_for_ticker,
)


def make_ohlcv(
    ticker: str = "SPY",
    rows: int = 220,
    *,
    start_price: float = 100.0,
    start_volume: int = 1_000,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    adj_close = np.arange(start_price, start_price + rows, dtype=float)
    volume = np.arange(start_volume, start_volume + rows, dtype=float)
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "open": adj_close - 0.5,
            "high": adj_close + 1.0,
            "low": adj_close - 1.0,
            "close": adj_close,
            "adj_close": adj_close,
            "volume": volume,
            "ticker": ticker,
        }
    )


def test_return_calculations() -> None:
    raw = make_ohlcv(rows=220)

    features = build_features(raw, dropna=False)
    row = features.iloc[20]

    assert row["return_1d"] == pytest.approx(120.0 / 119.0 - 1)
    assert row["return_5d"] == pytest.approx(120.0 / 115.0 - 1)
    assert row["return_20d"] == pytest.approx(120.0 / 100.0 - 1)


def test_rolling_volatility_uses_20_daily_returns() -> None:
    raw = make_ohlcv(rows=220)

    features = build_features(raw, dropna=False)
    expected_returns = raw["adj_close"].pct_change(1).iloc[1:21]

    assert features.loc[19, "volatility_20d"] != features.loc[19, "volatility_20d"]
    assert features.loc[20, "volatility_20d"] == pytest.approx(
        expected_returns.std()
    )


def test_moving_average_distance_calculation() -> None:
    raw = make_ohlcv(rows=220)

    features = build_features(raw, dropna=False)

    ma_50 = raw["adj_close"].iloc[0:50].mean()
    ma_200 = raw["adj_close"].iloc[0:200].mean()
    assert features.loc[49, "ma_50_distance"] == pytest.approx(149.0 / ma_50 - 1)
    assert features.loc[199, "ma_200_distance"] == pytest.approx(
        299.0 / ma_200 - 1
    )


def test_drawdown_60d_calculation() -> None:
    raw = make_ohlcv(rows=220)
    raw.loc[0:59, "adj_close"] = 100.0
    raw.loc[59, "adj_close"] = 120.0
    raw.loc[60, "adj_close"] = 90.0

    features = build_features(raw, dropna=False)

    assert features.loc[60, "drawdown_60d"] == pytest.approx(90.0 / 120.0 - 1)


def test_volume_change_20d_calculation() -> None:
    raw = make_ohlcv(rows=220)

    features = build_features(raw, dropna=False)

    expected_average_volume = raw["volume"].iloc[0:20].mean()
    assert features.loc[19, "volume_change_20d"] == pytest.approx(
        raw.loc[19, "volume"] / expected_average_volume - 1
    )


def test_multi_ticker_features_do_not_cross_ticker_boundaries() -> None:
    spy = make_ohlcv("SPY", rows=220, start_price=100.0)
    qqq = make_ohlcv("QQQ", rows=220, start_price=1_000.0)
    combined = pd.concat([spy, qqq], ignore_index=True)

    features = build_features(combined, dropna=False)
    first_qqq = features[features["ticker"] == "QQQ"].iloc[0]
    qqq_row_20 = features[features["ticker"] == "QQQ"].iloc[20]

    assert pd.isna(first_qqq["return_1d"])
    assert qqq_row_20["return_20d"] == pytest.approx(1020.0 / 1000.0 - 1)


def test_required_input_column_validation() -> None:
    raw = make_ohlcv().drop(columns=["adj_close"])

    with pytest.raises(MissingInputColumnsError, match="adj_close"):
        build_features(raw)


def test_dropna_behavior() -> None:
    raw = make_ohlcv(rows=220)

    kept = build_features(raw, dropna=False)
    dropped = build_features(raw, dropna=True)

    assert len(kept) == 220
    assert len(dropped) == 21
    assert dropped[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_no_forward_looking_columns_are_created() -> None:
    raw = make_ohlcv(rows=220)

    features = build_features(raw, dropna=False)

    forbidden_fragments = ("forward", "future", "next")
    assert not any(
        fragment in column
        for column in features.columns
        for fragment in forbidden_fragments
    )


def test_build_features_for_ticker_saves_processed_csv(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    raw_path = raw_dir / "SPY_2024-01-01_2024-12-31.csv"
    make_ohlcv(rows=220).to_csv(raw_path, index=False)

    result = build_features_for_ticker(
        "SPY",
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert result.rows == 21
    assert result.output_path.exists()
    saved = pd.read_csv(result.output_path)
    assert list(saved.columns)[-len(FEATURE_COLUMNS) :] == FEATURE_COLUMNS


def test_unsorted_duplicate_dates_are_sorted_before_features() -> None:
    raw = make_ohlcv(rows=205)
    duplicate = raw.iloc[[5]].copy()
    raw = pd.concat([raw, duplicate], ignore_index=True).sample(frac=1, random_state=3)

    features = build_features(raw, dropna=False)

    assert features["date"].is_monotonic_increasing
    assert len(features) == 206


def test_multiple_tickers_with_uneven_date_ranges_are_grouped_independently() -> None:
    spy = make_ohlcv("SPY", rows=220, start_price=100.0)
    qqq = make_ohlcv("QQQ", rows=205, start_price=500.0)

    features = build_features(pd.concat([spy, qqq], ignore_index=True), dropna=True)

    assert len(features[features["ticker"] == "SPY"]) == 21
    assert len(features[features["ticker"] == "QQQ"]) == 6


def test_insufficient_history_for_200_day_moving_average_drops_all_rows() -> None:
    raw = make_ohlcv(rows=199)

    features = build_features(raw, dropna=True)

    assert features.empty


def test_missing_volume_values_are_handled_by_dropna() -> None:
    raw = make_ohlcv(rows=220)
    raw.loc[210, "volume"] = pd.NA

    kept = build_features(raw, dropna=False)
    dropped = build_features(raw, dropna=True)

    assert pd.isna(kept.loc[210, "volume_change_20d"])
    assert len(dropped) < 21
