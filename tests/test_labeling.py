"""Tests for rule-based regime labeling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.labeling import (
    REGIME_LABELS,
    MissingFeatureColumnsError,
    add_rule_labels,
    label_features_for_ticker,
)


BASE_ROW = {
    "date": "2024-01-01",
    "open": 100.0,
    "high": 101.0,
    "low": 99.0,
    "close": 100.0,
    "adj_close": 100.0,
    "volume": 1_000_000,
    "ticker": "SPY",
    "return_1d": 0.001,
    "return_5d": 0.002,
    "return_20d": 0.0,
    "volatility_20d": 0.01,
    "ma_50_distance": 0.0,
    "ma_200_distance": 0.0,
    "drawdown_60d": -0.01,
    "volume_change_20d": 0.0,
}


def feature_frame(*rows: dict[str, object], **single_row: object) -> pd.DataFrame:
    if single_row:
        rows = (single_row,)
    if not rows:
        rows = ({},)

    merged_rows = []
    for index, row in enumerate(rows):
        merged = BASE_ROW.copy()
        merged.update(row)
        merged["date"] = f"2024-01-{index + 1:02d}"
        merged_rows.append(merged)
    return pd.DataFrame(merged_rows)


def test_stable_growth_can_be_produced() -> None:
    labeled = add_rule_labels(
        feature_frame(
            return_20d=0.04,
            ma_50_distance=0.02,
            ma_200_distance=0.03,
            volatility_20d=0.012,
            drawdown_60d=-0.01,
        )
    )

    assert labeled.loc[0, "rule_label"] == "stable_growth"


def test_volatile_recovery_can_be_produced() -> None:
    labeled = add_rule_labels(
        feature_frame(
            return_5d=0.025,
            return_20d=0.015,
            volatility_20d=0.03,
            ma_200_distance=-0.04,
            drawdown_60d=-0.09,
        )
    )

    assert labeled.loc[0, "rule_label"] == "volatile_recovery"


def test_sideways_defensive_can_be_produced() -> None:
    labeled = add_rule_labels(
        feature_frame(
            return_5d=0.002,
            return_20d=0.004,
            volatility_20d=0.014,
            ma_50_distance=-0.005,
            ma_200_distance=0.001,
            drawdown_60d=-0.015,
        )
    )

    assert labeled.loc[0, "rule_label"] == "sideways_defensive"


def test_stress_selloff_can_be_produced() -> None:
    labeled = add_rule_labels(
        feature_frame(
            return_20d=-0.06,
            volatility_20d=0.03,
            ma_50_distance=-0.03,
            drawdown_60d=-0.11,
        )
    )

    assert labeled.loc[0, "rule_label"] == "stress_selloff"


def test_missing_required_columns_raise_validation_error() -> None:
    data = feature_frame().drop(columns=["return_20d"])

    with pytest.raises(MissingFeatureColumnsError, match="return_20d"):
        add_rule_labels(data)


def test_missing_feature_values_are_unlabeled_when_kept() -> None:
    labeled = add_rule_labels(
        feature_frame(return_20d=pd.NA),
        drop_unlabeled=False,
    )

    assert pd.isna(labeled.loc[0, "rule_label"])


def test_drop_unlabeled_behavior() -> None:
    data = feature_frame({}, {"return_20d": pd.NA})

    kept = add_rule_labels(data, drop_unlabeled=False)
    dropped = add_rule_labels(data, drop_unlabeled=True)

    assert len(kept) == 2
    assert len(dropped) == 1
    assert pd.isna(kept.loc[1, "rule_label"])


def test_label_values_are_constrained_to_regime_labels() -> None:
    labeled = add_rule_labels(
        feature_frame(
            {"return_20d": 0.04, "ma_50_distance": 0.02, "ma_200_distance": 0.02},
            {"return_20d": -0.06, "ma_50_distance": -0.02},
            {"return_5d": 0.02, "drawdown_60d": -0.06, "volatility_20d": 0.03},
            {"return_20d": 0.0},
        )
    )

    assert set(labeled["rule_label"]).issubset(set(REGIME_LABELS))


def test_stress_selloff_takes_precedence_over_volatile_recovery() -> None:
    labeled = add_rule_labels(
        feature_frame(
            return_5d=0.03,
            return_20d=-0.06,
            volatility_20d=0.035,
            ma_50_distance=-0.04,
            ma_200_distance=-0.05,
            drawdown_60d=-0.10,
        )
    )

    assert labeled.loc[0, "rule_label"] == "stress_selloff"


def test_label_features_for_ticker_saves_labeled_csv(tmp_path) -> None:
    feature_path = tmp_path / "SPY_2024-01-01_2024-12-31_features_v1.csv"
    feature_frame(
        return_20d=0.04,
        ma_50_distance=0.02,
        ma_200_distance=0.02,
        volatility_20d=0.01,
        drawdown_60d=-0.01,
    ).to_csv(feature_path, index=False)

    result = label_features_for_ticker("SPY", processed_data_dir=tmp_path)

    assert result.rows == 1
    assert result.output_path.exists()
    saved = pd.read_csv(result.output_path)
    assert saved.loc[0, "rule_label"] == "stable_growth"
    assert result.label_counts == {"stable_growth": 1}
