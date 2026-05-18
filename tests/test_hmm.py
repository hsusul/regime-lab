"""Tests for optional HMM regime analysis."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_VERSION
from src.hmm import (
    HMM_FEATURE_COLUMNS,
    HMMInputDataNotFoundError,
    InvalidStateCountError,
    MissingHMMColumnsError,
    build_parser,
    compare_states_to_rule_labels,
    load_processed_tickers,
    prepare_hmm_data,
    train_hmm_regimes,
    validate_n_states,
)


class FakeGaussianHMM:
    """Small estimator stub with the hmmlearn fit/predict surface."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.fit_lengths: list[int] | None = None
        self.predict_lengths: list[int] | None = None

    def fit(self, observations: np.ndarray, lengths: list[int] | None = None) -> "FakeGaussianHMM":
        self.fit_lengths = lengths
        self.n_features_in_ = observations.shape[1]
        return self

    def predict(
        self,
        observations: np.ndarray,
        lengths: list[int] | None = None,
    ) -> np.ndarray:
        self.predict_lengths = lengths
        return np.arange(len(observations)) % self.n_components


def fake_estimator_factory(n_states: int) -> FakeGaussianHMM:
    return FakeGaussianHMM(n_states)


def make_hmm_data(
    *,
    tickers: tuple[str, ...] = ("SPY",),
    periods: int = 12,
    include_rule_label: bool = True,
) -> pd.DataFrame:
    rows = []
    labels = [
        "stable_growth",
        "volatile_recovery",
        "sideways_defensive",
        "stress_selloff",
    ]
    for ticker in tickers:
        for index, date in enumerate(pd.date_range("2024-01-01", periods=periods)):
            row = {
                "date": date.date().isoformat(),
                "ticker": ticker,
                "return_1d": (index % 5 - 2) / 1_000,
                "return_5d": (index % 7 - 3) / 100,
                "return_20d": (index % 9 - 4) / 50,
                "volatility_20d": 0.01 + (index % 4) / 100,
                "ma_50_distance": 0.01,
                "ma_200_distance": 0.02,
                "drawdown_60d": -0.01 * (index % 6),
                "volume_change_20d": 0.05,
            }
            if include_rule_label:
                row["rule_label"] = labels[index % len(labels)]
            rows.append(row)
    return pd.DataFrame(rows)


def test_validate_n_states_accepts_only_supported_range() -> None:
    validate_n_states(3)
    validate_n_states(5)

    with pytest.raises(InvalidStateCountError):
        validate_n_states(2)
    with pytest.raises(InvalidStateCountError):
        validate_n_states(6)


def test_prepare_hmm_data_sorts_and_keeps_per_ticker_sequences() -> None:
    data = make_hmm_data(tickers=("QQQ", "SPY"), periods=5).sample(
        frac=1,
        random_state=7,
    )

    prepared = prepare_hmm_data(data, n_states=3)

    assert prepared["ticker"].tolist()[:5] == ["QQQ"] * 5
    assert prepared["ticker"].tolist()[5:] == ["SPY"] * 5
    assert prepared["date"].is_monotonic_increasing is False
    assert prepared.groupby("ticker")["date"].apply(lambda dates: dates.is_monotonic_increasing).all()


def test_prepare_hmm_data_requires_summary_columns() -> None:
    data = make_hmm_data().drop(columns=["drawdown_60d"])

    with pytest.raises(MissingHMMColumnsError):
        prepare_hmm_data(data, n_states=4)


def test_train_hmm_regimes_saves_artifact_and_report(tmp_path) -> None:
    models_dir = tmp_path / "models" / "hmm"
    reports_dir = tmp_path / "reports"
    data = make_hmm_data(tickers=("SPY", "QQQ"), periods=10)

    result = train_hmm_regimes(
        data,
        n_states=4,
        models_dir=models_dir,
        reports_dir=reports_dir,
        created_at="2024-05-01T00:00:00+00:00",
        estimator_factory=fake_estimator_factory,
    )

    assert result.rows == 20
    assert result.ticker_universe == ["QQQ", "SPY"]
    assert result.artifact_path.exists()
    assert result.report_path.exists()

    artifact = joblib.load(result.artifact_path)
    assert artifact["analysis_type"] == "hidden_markov_model"
    assert artifact["feature_columns"] == HMM_FEATURE_COLUMNS
    assert artifact["n_states"] == 4
    assert artifact["model"].fit_lengths == [10, 10]

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["hmm_run_id"] == result.hmm_run_id
    assert set(report["state_summary"]) == {"0", "1", "2", "3"}
    assert report["rule_label_comparison"]["0"]["labels"]["stable_growth"]["count"] > 0
    assert "not financial advice" in " ".join(report["warnings"])


def test_compare_states_to_rule_labels_returns_empty_without_labels() -> None:
    data = make_hmm_data(include_rule_label=False)
    data["hmm_state"] = np.arange(len(data)) % 4

    assert compare_states_to_rule_labels(data) == {}


def test_load_processed_tickers_reads_cached_feature_files(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    data = make_hmm_data(tickers=("SPY",), periods=6)
    output_path = processed_dir / f"SPY_20150101_20260515_features_{FEATURE_VERSION}.csv"
    data.to_csv(output_path, index=False)

    loaded = load_processed_tickers(["SPY"], processed_data_dir=processed_dir)

    assert len(loaded) == 6
    assert loaded["ticker"].unique().tolist() == ["SPY"]


def test_load_processed_tickers_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(HMMInputDataNotFoundError):
        load_processed_tickers(["SPY"], processed_data_dir=tmp_path)


def test_hmm_cli_parser_accepts_expected_arguments() -> None:
    args = build_parser().parse_args(["--tickers", "SPY", "QQQ", "--n-states", "3"])

    assert args.tickers == ["SPY", "QQQ"]
    assert args.n_states == 3
