"""Tests for optional walk-forward validation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.train import TRAINING_TARGET_COLUMN
from src.walk_forward import (
    InsufficientWalkForwardDataError,
    InvalidWalkForwardConfigError,
    build_parser,
    build_walk_forward_folds,
    build_walk_forward_report,
    flatten_fold_for_csv,
    run_walk_forward_validation,
    validate_walk_forward_config,
)
from tests.test_api import write_labeled_files
from tests.test_train import make_labeled_data


def make_walk_forward_data(
    *,
    tickers: tuple[str, ...] = ("SPY",),
    start: str = "2016-01-01",
    periods: int = 72,
    freq: str = "MS",
) -> pd.DataFrame:
    data = make_labeled_data(tickers=tickers, periods=periods)
    dates = pd.date_range(start, periods=periods, freq=freq)
    repeated_dates = []
    for date in dates:
        repeated_dates.extend([date.date().isoformat()] * len(tickers))
    data["date"] = repeated_dates
    return data


def test_validate_walk_forward_config_rejects_invalid_values() -> None:
    validate_walk_forward_config(
        model_type="random_forest",
        start_year=2018,
        test_window_years=1,
    )

    with pytest.raises(InvalidWalkForwardConfigError):
        validate_walk_forward_config(
            model_type="xgboost",
            start_year=2018,
            test_window_years=1,
        )
    with pytest.raises(InvalidWalkForwardConfigError):
        validate_walk_forward_config(
            model_type="random_forest",
            start_year=1800,
            test_window_years=1,
        )
    with pytest.raises(InvalidWalkForwardConfigError):
        validate_walk_forward_config(
            model_type="random_forest",
            start_year=2018,
            test_window_years=0,
        )


def test_build_walk_forward_folds_use_expanding_chronological_windows() -> None:
    data = make_walk_forward_data(tickers=("SPY", "QQQ"), periods=60)

    folds = build_walk_forward_folds(
        data,
        start_year=2018,
        test_window_years=1,
    )

    first = folds[0]
    second = folds[1]
    assert first.fold_index == 1
    assert first.train["date"].max() < first.test["date"].min()
    assert first.test["date"].min().date().isoformat() == "2018-01-01"
    assert first.test["date"].max().date().isoformat() == "2018-12-01"
    assert second.train["date"].max().date().isoformat() == "2018-12-01"
    assert second.test["date"].min().date().isoformat() == "2019-01-01"
    assert set(first.test[first.test["date"] == "2018-01-01"]["ticker"]) == {
        "SPY",
        "QQQ",
    }


def test_build_walk_forward_folds_raises_when_no_valid_folds() -> None:
    data = make_walk_forward_data(periods=12)

    with pytest.raises(InsufficientWalkForwardDataError):
        build_walk_forward_folds(data, start_year=2030, test_window_years=1)


def test_build_walk_forward_report_contains_required_fold_metrics() -> None:
    data = make_walk_forward_data(tickers=("SPY", "QQQ"), periods=72)

    report = build_walk_forward_report(
        data,
        model_type="random_forest",
        start_year=2018,
        test_window_years=1,
        created_at="2024-05-01T00:00:00+00:00",
    )

    fold = report["folds"][0]
    assert report["analysis_type"] == "walk_forward_validation"
    assert report["model_type"] == "random_forest"
    assert report["tickers"] == ["QQQ", "SPY"]
    assert "forward_return_5d" not in report["feature_columns"]
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert {
        "fold_index",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "train_rows",
        "test_rows",
        "accuracy",
        "macro_f1",
        "per_class_f1",
    } <= set(fold)
    assert set(fold["per_class_f1"]) == {
        "stable_growth",
        "volatile_recovery",
        "sideways_defensive",
        "stress_selloff",
    }
    assert report["target_column"] == TRAINING_TARGET_COLUMN
    assert report["summary"]["fold_count"] == len(report["folds"])
    assert any("heuristic labels" in warning for warning in report["warnings"])


def test_logistic_regression_walk_forward_smoke() -> None:
    data = make_walk_forward_data(periods=72)

    report = build_walk_forward_report(
        data,
        model_type="logistic_regression",
        start_year=2018,
        test_window_years=1,
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert report["folds"]
    assert report["summary"]["mean_macro_f1"] is not None


def test_flatten_fold_for_csv_creates_per_class_columns() -> None:
    fold = {
        "fold_index": 1,
        "macro_f1": 0.9,
        "per_class_f1": {
            "stable_growth": 0.8,
            "volatile_recovery": 0.7,
            "sideways_defensive": 0.6,
            "stress_selloff": 0.5,
        },
    }

    flattened = flatten_fold_for_csv(fold)

    assert "per_class_f1" not in flattened
    assert flattened["f1_stable_growth"] == 0.8
    assert flattened["f1_volatile_recovery"] == 0.7
    assert flattened["f1_sideways_defensive"] == 0.6
    assert flattened["f1_stress_selloff"] == 0.5


def test_run_walk_forward_validation_saves_json_and_optional_csv(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    write_labeled_files(make_walk_forward_data(periods=72), processed_dir)

    result = run_walk_forward_validation(
        ["SPY"],
        model_type="random_forest",
        start_year=2018,
        test_window_years=1,
        processed_data_dir=processed_dir,
        reports_dir=reports_dir,
        save_csv=True,
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert result.report_path.exists()
    assert result.csv_path is not None
    assert result.csv_path.exists()
    saved = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert saved["model_type"] == "random_forest"
    csv_text = result.csv_path.read_text(encoding="utf-8")
    assert "macro_f1" in csv_text
    assert "per_class_f1" not in csv_text
    assert "f1_stable_growth" in csv_text
    assert "f1_volatile_recovery" in csv_text
    assert "f1_sideways_defensive" in csv_text
    assert "f1_stress_selloff" in csv_text


def test_walk_forward_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--tickers",
            "SPY",
            "QQQ",
            "--model-type",
            "random_forest",
            "--start-year",
            "2018",
            "--test-window-years",
            "1",
            "--save-csv",
        ]
    )

    assert args.tickers == ["SPY", "QQQ"]
    assert args.model_type == "random_forest"
    assert args.start_year == 2018
    assert args.test_window_years == 1
    assert args.save_csv is True
