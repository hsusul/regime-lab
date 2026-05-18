"""Tests for retrospective forward return analysis."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.forward_returns import (
    ForwardDataNotFoundError,
    InvalidForwardHorizonError,
    MissingForwardColumnsError,
    PREDICTED_REGIME_COLUMN,
    add_forward_returns,
    build_forward_return_report,
    build_parser,
    flatten_summary_rows,
    forward_return_column,
    load_labeled_tickers,
    run_forward_return_analysis,
    summarize_by_regime,
    validate_horizons,
)
from src.train import TRAINING_TARGET_COLUMN, train_model
from tests.test_api import write_labeled_files
from tests.test_train import make_labeled_data


def make_forward_data(
    *,
    tickers: tuple[str, ...] = ("SPY",),
    periods: int = 8,
) -> pd.DataFrame:
    data = make_labeled_data(tickers=tickers, periods=periods)
    data["adj_close"] = data.groupby("ticker").cumcount() + 100.0
    data["close"] = data["adj_close"]
    return data


def test_validate_horizons_returns_sorted_unique_positive_values() -> None:
    assert validate_horizons([20, 5, 5]) == [5, 20]

    with pytest.raises(InvalidForwardHorizonError):
        validate_horizons([0])


def test_add_forward_returns_computes_by_ticker_without_cross_ticker_leakage() -> None:
    data = make_forward_data(tickers=("SPY", "QQQ"), periods=4).sample(
        frac=1,
        random_state=11,
    )

    analyzed = add_forward_returns(data, [1, 2]).sort_values(["ticker", "date"])

    spy = analyzed[analyzed["ticker"] == "SPY"].reset_index(drop=True)
    qqq = analyzed[analyzed["ticker"] == "QQQ"].reset_index(drop=True)
    assert spy.loc[0, "forward_return_1d"] == pytest.approx(101 / 100 - 1)
    assert spy.loc[0, "forward_return_2d"] == pytest.approx(102 / 100 - 1)
    assert pd.isna(spy.loc[3, "forward_return_1d"])
    assert qqq.loc[0, "forward_return_1d"] == pytest.approx(101 / 100 - 1)


def test_forward_returns_are_not_feature_columns() -> None:
    add_forward_returns(make_forward_data(periods=8), [5])

    assert all(not column.startswith("forward_return_") for column in FEATURE_COLUMNS)


def test_add_forward_returns_requires_labeled_price_data() -> None:
    data = make_forward_data().drop(columns=["adj_close"])

    with pytest.raises(MissingForwardColumnsError):
        add_forward_returns(data, [5])


def test_summarize_by_regime_includes_required_metrics() -> None:
    analyzed = add_forward_returns(make_forward_data(periods=8), [1])

    summary = summarize_by_regime(
        analyzed,
        group_column=TRAINING_TARGET_COLUMN,
        horizons=[1],
    )

    stable = summary["stable_growth"]
    stats = stable["horizons"]["1"]
    assert stable["observation_count"] == 2
    assert set(stats) == {
        "valid_count",
        "average_forward_return",
        "median_forward_return",
        "forward_volatility",
        "worst_forward_return",
        "best_forward_return",
        "hit_rate",
    }


def test_build_forward_return_report_without_model_has_rule_summaries(tmp_path) -> None:
    report = build_forward_return_report(
        make_forward_data(tickers=("SPY", "QQQ"), periods=8),
        horizons=[1, 3],
        reports_dir=tmp_path / "reports",
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert report["analysis_type"] == "forward_return_regime_analysis"
    assert report["latest_model"]["available"] is False
    assert report["tickers"] == ["QQQ", "SPY"]
    assert "stable_growth" in report["summaries"]["combined"][TRAINING_TARGET_COLUMN]
    assert report["summaries"]["combined"][PREDICTED_REGIME_COLUMN] == {}
    assert "analysis-only" in report["leakage_note"]


def test_build_forward_return_report_with_model_adds_predicted_regime_group(
    tmp_path,
) -> None:
    data = make_forward_data(tickers=("SPY", "QQQ"), periods=24)
    reports_dir = tmp_path / "reports"
    train_model(
        data,
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=reports_dir,
        created_at="2024-05-01T00:00:00+00:00",
    )

    report = build_forward_return_report(
        data,
        horizons=[5],
        reports_dir=reports_dir,
        created_at="2024-05-02T00:00:00+00:00",
    )

    predicted = report["summaries"]["combined"][PREDICTED_REGIME_COLUMN]
    assert report["latest_model"]["available"] is True
    assert predicted


def test_run_forward_return_analysis_saves_json_and_optional_csv(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    write_labeled_files(make_forward_data(periods=12), processed_dir)

    result = run_forward_return_analysis(
        ["SPY"],
        horizons=[1, 5],
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
        save_csv=True,
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert result.report_path.exists()
    assert result.csv_path is not None
    assert result.csv_path.exists()
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["horizons"] == [1, 5]
    csv_text = result.csv_path.read_text(encoding="utf-8")
    assert "average_forward_return" in csv_text


def test_flatten_summary_rows_contains_combined_and_ticker_rows(tmp_path) -> None:
    report = build_forward_return_report(
        make_forward_data(periods=8),
        horizons=[1],
        reports_dir=tmp_path / "reports",
        created_at="2024-05-01T00:00:00+00:00",
    )

    rows = flatten_summary_rows(report)

    assert any(row["scope"] == "combined" for row in rows)
    assert any(row["scope"] == "ticker" and row["ticker"] == "SPY" for row in rows)


def test_load_labeled_tickers_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(ForwardDataNotFoundError):
        load_labeled_tickers(["SPY"], processed_data_dir=tmp_path)


def test_forward_return_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(
        ["--tickers", "SPY", "QQQ", "--horizons", "5", "20", "60", "--save-csv"]
    )

    assert args.tickers == ["SPY", "QQQ"]
    assert args.horizons == [5, 20, 60]
    assert args.save_csv is True


def test_forward_return_column_naming() -> None:
    assert forward_return_column(20) == "forward_return_20d"
