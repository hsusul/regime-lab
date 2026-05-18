"""Optional Hidden Markov Model regime analysis for RegimeLab.

This module implements Advanced Version 1 as a separate exploratory path. It
does not change the supervised classifier, evaluation code, or API behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import validate_ticker
from src.features import FEATURE_VERSION, PROCESSED_DATA_DIR
from src.labeling import REGIME_LABELS
from src.predict import find_feature_or_labeled_file
from src.train import MODELS_DIR, REPORTS_DIR, TRAINING_TARGET_COLUMN


DEFAULT_N_STATES = 4
MIN_HMM_STATES = 3
MAX_HMM_STATES = 5
RANDOM_SEED = 42
HMM_ARTIFACT_VERSION = "v1"
HMM_MODELS_DIR = MODELS_DIR / "hmm"
HMM_FEATURE_COLUMNS = ["return_1d", "volatility_20d"]
HMM_SUMMARY_COLUMNS = [
    "return_1d",
    "return_20d",
    "volatility_20d",
    "drawdown_60d",
]
HMM_REQUIRED_COLUMNS = ["date", "ticker", *HMM_SUMMARY_COLUMNS]


class HMMError(Exception):
    """Base exception for HMM analysis failures."""


class OptionalDependencyError(HMMError):
    """Raised when hmmlearn is required but not installed."""


class MissingHMMColumnsError(HMMError):
    """Raised when processed features cannot support HMM analysis."""


class InvalidStateCountError(HMMError):
    """Raised when an unsupported number of hidden states is requested."""


class InsufficientHMMDataError(HMMError):
    """Raised when there are not enough complete rows to fit the HMM."""


class HMMInputDataNotFoundError(HMMError):
    """Raised when cached processed data cannot be found for a ticker."""


@dataclass(frozen=True)
class HMMTrainingResult:
    """Paths and metadata from an HMM analysis run."""

    hmm_run_id: str
    artifact_path: Path
    report_path: Path
    rows: int
    n_states: int
    ticker_universe: list[str]
    report: dict[str, Any]


def validate_n_states(n_states: int) -> None:
    """Validate supported HMM state count."""
    if n_states < MIN_HMM_STATES or n_states > MAX_HMM_STATES:
        raise InvalidStateCountError(
            f"n_states must be between {MIN_HMM_STATES} and {MAX_HMM_STATES}."
        )


def validate_hmm_columns(data: pd.DataFrame) -> None:
    """Validate that processed feature data contains required HMM columns."""
    missing = sorted(set(HMM_REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise MissingHMMColumnsError(
            f"Missing required HMM columns: {', '.join(missing)}"
        )


def prepare_hmm_data(data: pd.DataFrame, n_states: int) -> pd.DataFrame:
    """Clean and sort feature rows for HMM fitting without future information."""
    validate_n_states(n_states)
    validate_hmm_columns(data)

    prepared = data.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["date"] = pd.to_datetime(prepared["date"])
    for column in HMM_SUMMARY_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=HMM_FEATURE_COLUMNS)
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    if len(prepared) < n_states:
        raise InsufficientHMMDataError(
            f"At least {n_states} complete rows are required to fit a {n_states}-state HMM."
        )
    return prepared


def sequence_lengths(data: pd.DataFrame) -> list[int]:
    """Return per-ticker sequence lengths for sorted HMM input."""
    return [int(count) for count in data.groupby("ticker", sort=False).size().tolist()]


def create_hmm_model(n_states: int) -> Any:
    """Create a Gaussian HMM, importing hmmlearn only when this path is used."""
    validate_n_states(n_states)
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise OptionalDependencyError(
            "HMM analysis requires the optional hmmlearn dependency. "
            'Install it with: python -m pip install -e ".[hmm]"'
        ) from exc

    return GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
    )


def make_hmm_run_id(n_states: int, created_at: str) -> str:
    """Create a filesystem-safe HMM run identifier."""
    timestamp = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    return f"{timestamp}_hmm_{n_states}_states"


def date_range_for(data: pd.DataFrame) -> dict[str, str]:
    """Return inclusive date range metadata for HMM input data."""
    dates = pd.to_datetime(data["date"])
    return {
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
    }


def _safe_mean(series: pd.Series) -> float | None:
    mean = series.mean()
    if pd.isna(mean):
        return None
    return float(mean)


def interpret_state(
    *,
    average_return_20d: float | None,
    average_volatility_20d: float | None,
    average_drawdown_60d: float | None,
    median_volatility_20d: float,
) -> str:
    """Assign a descriptive name to an inferred state after training."""
    return_20d = average_return_20d if average_return_20d is not None else 0.0
    volatility = (
        average_volatility_20d if average_volatility_20d is not None else 0.0
    )
    drawdown = average_drawdown_60d if average_drawdown_60d is not None else 0.0

    if return_20d < -0.02 and (drawdown < -0.04 or volatility > median_volatility_20d):
        return "stress_selloff_like"
    if return_20d > 0 and drawdown < -0.03 and volatility >= median_volatility_20d:
        return "volatile_recovery_like"
    if return_20d > 0 and volatility <= median_volatility_20d and drawdown > -0.03:
        return "stable_growth_like"
    return "sideways_defensive_like"


def summarize_hmm_states(
    data: pd.DataFrame,
    *,
    n_states: int,
) -> dict[str, dict[str, Any]]:
    """Summarize each inferred HMM state with interpretable statistics."""
    total_rows = len(data)
    median_volatility = float(data["volatility_20d"].median())
    summaries: dict[str, dict[str, Any]] = {}

    for state in range(n_states):
        state_data = data[data["hmm_state"] == state]
        frequency = int(len(state_data))
        average_return_1d = _safe_mean(state_data["return_1d"])
        average_return_20d = _safe_mean(state_data["return_20d"])
        average_volatility_20d = _safe_mean(state_data["volatility_20d"])
        average_drawdown_60d = _safe_mean(state_data["drawdown_60d"])
        summaries[str(state)] = {
            "state": state,
            "frequency": frequency,
            "pct": 0.0 if total_rows == 0 else frequency / total_rows,
            "average_return_1d": average_return_1d,
            "average_return_20d": average_return_20d,
            "average_volatility_20d": average_volatility_20d,
            "average_drawdown_60d": average_drawdown_60d,
            "interpreted_regime": interpret_state(
                average_return_20d=average_return_20d,
                average_volatility_20d=average_volatility_20d,
                average_drawdown_60d=average_drawdown_60d,
                median_volatility_20d=median_volatility,
            ),
        }
    return summaries


def compare_states_to_rule_labels(data: pd.DataFrame) -> dict[str, Any]:
    """Compare unsupervised HMM states against heuristic rule labels when present."""
    if TRAINING_TARGET_COLUMN not in data.columns:
        return {}

    comparison: dict[str, Any] = {}
    for state, state_data in data.groupby("hmm_state", sort=True):
        labels = state_data[TRAINING_TARGET_COLUMN].dropna().astype(str)
        total = int(len(labels))
        counts = labels.value_counts()
        comparison[str(int(state))] = {
            "total": total,
            "labels": {
                label: {
                    "count": int(counts.get(label, 0)),
                    "pct": 0.0 if total == 0 else int(counts.get(label, 0)) / total,
                }
                for label in REGIME_LABELS
            },
        }
    return comparison


def build_hmm_artifact(
    *,
    model: Any,
    scaler: StandardScaler,
    n_states: int,
    ticker_universe: list[str],
    date_range: dict[str, str],
    created_at: str,
    hmm_run_id: str,
) -> dict[str, Any]:
    """Build the persisted HMM artifact dictionary."""
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(HMM_FEATURE_COLUMNS),
        "n_states": n_states,
        "ticker_universe": ticker_universe,
        "date_range": date_range,
        "feature_version": FEATURE_VERSION,
        "created_at": created_at,
        "hmm_run_id": hmm_run_id,
        "artifact_version": HMM_ARTIFACT_VERSION,
        "analysis_type": "hidden_markov_model",
    }


def train_hmm_regimes(
    data: pd.DataFrame,
    *,
    n_states: int = DEFAULT_N_STATES,
    models_dir: Path = HMM_MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    created_at: str | None = None,
    estimator_factory: Callable[[int], Any] | None = None,
) -> HMMTrainingResult:
    """Fit an HMM on return/volatility features and persist artifact/report."""
    prepared = prepare_hmm_data(data, n_states)
    lengths = sequence_lengths(prepared)
    scaler = StandardScaler()
    observations = scaler.fit_transform(prepared[HMM_FEATURE_COLUMNS])

    model = (
        estimator_factory(n_states)
        if estimator_factory is not None
        else create_hmm_model(n_states)
    )
    model.fit(observations, lengths)
    states = model.predict(observations, lengths)

    analyzed = prepared.copy()
    analyzed["hmm_state"] = np.asarray(states, dtype=int)
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="microseconds")
    hmm_run_id = make_hmm_run_id(n_states, timestamp)
    ticker_universe = sorted(analyzed["ticker"].unique().tolist())
    run_date_range = date_range_for(analyzed)

    artifact = build_hmm_artifact(
        model=model,
        scaler=scaler,
        n_states=n_states,
        ticker_universe=ticker_universe,
        date_range=run_date_range,
        created_at=timestamp,
        hmm_run_id=hmm_run_id,
    )
    state_summary = summarize_hmm_states(analyzed, n_states=n_states)
    rule_label_comparison = compare_states_to_rule_labels(analyzed)
    warnings = [
        "HMM state names are interpreted after training and are not known ahead of time.",
        "HMM output is exploratory analysis, not financial advice or a trading signal.",
    ]

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"hmm_{hmm_run_id}.joblib"
    report_path = reports_dir / f"hmm_{hmm_run_id}_report.json"
    joblib.dump(artifact, artifact_path)

    report = {
        "hmm_run_id": hmm_run_id,
        "created_at": timestamp,
        "analysis_type": "hidden_markov_model",
        "n_states": n_states,
        "tickers": ticker_universe,
        "rows": int(len(analyzed)),
        "date_range": run_date_range,
        "feature_columns": list(HMM_FEATURE_COLUMNS),
        "feature_version": FEATURE_VERSION,
        "artifact_path": str(artifact_path),
        "state_summary": state_summary,
        "rule_label_comparison": rule_label_comparison,
        "warnings": warnings,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    return HMMTrainingResult(
        hmm_run_id=hmm_run_id,
        artifact_path=artifact_path,
        report_path=report_path,
        rows=len(analyzed),
        n_states=n_states,
        ticker_universe=ticker_universe,
        report=report,
    )


def load_processed_tickers(
    tickers: Sequence[str],
    *,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load newest cached processed or labeled feature files for HMM analysis."""
    frames = []
    for ticker in tickers:
        normalized = validate_ticker(ticker)
        try:
            path = find_feature_or_labeled_file(normalized, processed_data_dir)
        except Exception as exc:
            raise HMMInputDataNotFoundError(
                f"No processed feature data found for {normalized} in {processed_data_dir}."
            ) from exc
        frames.append(pd.read_csv(path))

    if not frames:
        raise HMMInputDataNotFoundError("At least one ticker is required.")
    return pd.concat(frames, ignore_index=True)


def train_hmm_from_processed_files(
    tickers: Sequence[str],
    *,
    n_states: int = DEFAULT_N_STATES,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = HMM_MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> HMMTrainingResult:
    """Load cached processed features and run HMM analysis."""
    data = load_processed_tickers(tickers, processed_data_dir=processed_data_dir)
    return train_hmm_regimes(
        data,
        n_states=n_states,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the HMM CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fit an optional HMM for exploratory latent regime analysis."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    parser.add_argument(
        "--n-states",
        type=int,
        default=DEFAULT_N_STATES,
        help="Number of hidden states to fit, from 3 to 5.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing cached processed or labeled feature CSV files.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=HMM_MODELS_DIR,
        help="Directory for saved HMM artifacts.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for HMM reports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for HMM regime analysis."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = train_hmm_from_processed_files(
            args.tickers,
            n_states=args.n_states,
            processed_data_dir=args.processed_data_dir,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
        )
    except HMMError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(f"hmm_run_id: {result.hmm_run_id}")
    print(f"n_states: {result.n_states}")
    print(f"rows: {result.rows}")
    print(f"tickers: {result.ticker_universe}")
    print(f"artifact_path: {result.artifact_path}")
    print(f"report_path: {result.report_path}")
    print("state_summary:")
    for state, summary in result.report["state_summary"].items():
        print(
            f"  state {state}: {summary['frequency']} rows "
            f"({summary['pct']:.1%}), interpreted={summary['interpreted_regime']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
