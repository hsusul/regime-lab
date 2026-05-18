"""Tests for supervised model training."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from src.features import FEATURE_COLUMNS, FEATURE_VERSION
from src.labeling import LABELING_VERSION, REGIME_LABELS
from src.train import (
    EXPERIMENTS_FILENAME,
    MissingTrainingColumnsError,
    label_distribution_for,
    load_experiments,
    time_based_split,
    train_model,
)


def make_labeled_data(
    *,
    tickers: tuple[str, ...] = ("SPY",),
    periods: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            base = float(date_index + 1 + ticker_index)
            label = REGIME_LABELS[(date_index + ticker_index) % len(REGIME_LABELS)]
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "open": 100.0 + base,
                    "high": 101.0 + base,
                    "low": 99.0 + base,
                    "close": 100.5 + base,
                    "adj_close": 100.5 + base,
                    "volume": 1_000_000 + date_index,
                    "ticker": ticker,
                    "return_1d": base / 1_000,
                    "return_5d": base / 900,
                    "return_20d": base / 800,
                    "volatility_20d": 0.01 + base / 10_000,
                    "ma_50_distance": base / 700,
                    "ma_200_distance": base / 600,
                    "drawdown_60d": -base / 1_000,
                    "volume_change_20d": base / 500,
                    "rule_label": label,
                }
            )
    return pd.DataFrame(rows)


def test_time_based_split_uses_earliest_80_percent_dates() -> None:
    data = make_labeled_data(periods=10).sample(frac=1, random_state=42)

    split = time_based_split(data)

    assert split.cutoff_date == "2024-01-08"
    assert len(split.train) == 8
    assert len(split.test) == 2
    assert split.train["date"].max().date().isoformat() == "2024-01-08"
    assert split.test["date"].min().date().isoformat() == "2024-01-09"


def test_combined_ticker_split_uses_shared_chronological_cutoff() -> None:
    data = make_labeled_data(tickers=("SPY", "QQQ"), periods=10)

    split = time_based_split(data)

    assert split.cutoff_date == "2024-01-08"
    assert len(split.train) == 16
    assert len(split.test) == 4
    assert set(split.train[split.train["date"] == "2024-01-08"]["ticker"]) == {
        "SPY",
        "QQQ",
    }
    assert set(split.test[split.test["date"] == "2024-01-09"]["ticker"]) == {
        "SPY",
        "QQQ",
    }


def test_missing_required_training_columns_raise_validation_error() -> None:
    data = make_labeled_data().drop(columns=["rule_label"])

    with pytest.raises(MissingTrainingColumnsError, match="rule_label"):
        train_model(data, "random_forest")


def test_logistic_regression_training_smoke_test(tmp_path) -> None:
    result = train_model(
        make_labeled_data(periods=24),
        "logistic_regression",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        created_at="2024-01-01T00:00:00+00:00",
    )

    assert result.artifact_path.exists()
    assert result.train_rows == 19
    assert result.test_rows == 5


def test_random_forest_training_smoke_test(tmp_path) -> None:
    result = train_model(
        make_labeled_data(periods=24),
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        created_at="2024-01-02T00:00:00+00:00",
    )

    assert result.artifact_path.exists()
    artifact = joblib.load(result.artifact_path)
    predictions = artifact["model"].predict(make_labeled_data(periods=2)[FEATURE_COLUMNS])
    assert len(predictions) == 2


def test_artifact_can_be_loaded_and_contains_required_metadata(tmp_path) -> None:
    result = train_model(
        make_labeled_data(tickers=("SPY", "QQQ"), periods=24),
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        created_at="2024-01-03T00:00:00+00:00",
    )

    artifact = joblib.load(result.artifact_path)

    assert {
        "model",
        "feature_columns",
        "label_names",
        "model_type",
        "training_date_range",
        "test_date_range",
        "ticker_universe",
        "feature_version",
        "labeling_version",
        "created_at",
        "experiment_id",
    }.issubset(artifact)
    assert artifact["feature_columns"] == FEATURE_COLUMNS
    assert artifact["label_names"] == REGIME_LABELS
    assert artifact["ticker_universe"] == ["QQQ", "SPY"]
    assert artifact["feature_version"] == FEATURE_VERSION
    assert artifact["labeling_version"] == LABELING_VERSION
    assert artifact["sklearn_version"] == sklearn.__version__
    assert artifact["pandas_version"] == pd.__version__
    assert artifact["numpy_version"] == np.__version__
    assert artifact["python_version"]
    assert artifact["environment"] == {
        "python_version": artifact["python_version"],
        "sklearn_version": artifact["sklearn_version"],
        "pandas_version": artifact["pandas_version"],
        "numpy_version": artifact["numpy_version"],
    }


def test_experiments_json_is_created_and_appended(tmp_path) -> None:
    reports_dir = tmp_path / "reports"

    first = train_model(
        make_labeled_data(periods=24),
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=reports_dir,
        created_at="2024-01-04T00:00:00+00:00",
    )
    second = train_model(
        make_labeled_data(periods=24),
        "logistic_regression",
        models_dir=tmp_path / "models",
        reports_dir=reports_dir,
        created_at="2024-01-05T00:00:00+00:00",
    )

    experiments_path = reports_dir / EXPERIMENTS_FILENAME
    experiments = load_experiments(experiments_path)

    assert experiments_path.exists()
    assert [item["experiment_id"] for item in experiments] == [
        first.experiment_id,
        second.experiment_id,
    ]


def test_label_distribution_is_recorded(tmp_path) -> None:
    result = train_model(
        make_labeled_data(periods=20),
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        created_at="2024-01-06T00:00:00+00:00",
    )

    with result.summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert result.label_distribution == {
        "sideways_defensive": 5,
        "stable_growth": 5,
        "stress_selloff": 5,
        "volatile_recovery": 5,
    }
    assert summary["label_distribution"] == result.label_distribution


def test_split_has_no_leakage_prone_shuffle_behavior() -> None:
    data = make_labeled_data(tickers=("SPY", "QQQ"), periods=12).sample(
        frac=1,
        random_state=7,
    )

    split = time_based_split(data)

    assert split.train["date"].is_monotonic_increasing
    assert split.test["date"].is_monotonic_increasing
    assert split.train["date"].max() < split.test["date"].min()


def test_label_distribution_handles_absent_labels() -> None:
    data = make_labeled_data(periods=8)
    data = data[data["rule_label"].isin(["stable_growth", "volatile_recovery"])]

    distribution = label_distribution_for(data)

    assert distribution == {"stable_growth": 2, "volatile_recovery": 2}
