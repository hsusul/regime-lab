"""API tests for cached RegimeLab backend endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd

import src.predict as predict_service
from app.main import app
from src.predict import DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT
from src.train import train_model
from tests.test_train import make_labeled_data


def configure_api_dirs(monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    raw_dir = tmp_path / "raw"
    processed_dir.mkdir()
    reports_dir.mkdir()
    raw_dir.mkdir()
    monkeypatch.setattr(predict_service, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(predict_service, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(predict_service, "RAW_DATA_DIR", raw_dir)
    return processed_dir, reports_dir, raw_dir


def write_labeled_files(data, processed_dir: Path) -> None:
    for ticker, group in data.groupby("ticker"):
        group.to_csv(
            processed_dir / f"{ticker}_synthetic_features_v1_labeled_v1.csv",
            index=False,
        )


def create_trained_api_fixture(
    tmp_path: Path,
    processed_dir: Path,
    reports_dir: Path,
    *,
    tickers: tuple[str, ...] = ("SPY",),
    model_type: str = "random_forest",
):
    data = make_labeled_data(tickers=tickers, periods=24)
    write_labeled_files(data, processed_dir)
    result = train_model(
        data,
        model_type,
        models_dir=tmp_path / "models",
        reports_dir=reports_dir,
        created_at="2024-03-01T00:00:00+00:00",
    )
    return result


def test_health_works_without_model(monkeypatch, tmp_path) -> None:
    configure_api_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert response.json()["latest_experiment_id"] is None


def test_health_reports_model_when_valid_artifact_exists(monkeypatch, tmp_path) -> None:
    processed_dir, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    result = create_trained_api_fixture(tmp_path, processed_dir, reports_dir)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["latest_experiment_id"] == result.experiment_id


def test_experiments_returns_empty_list_when_missing(monkeypatch, tmp_path) -> None:
    configure_api_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/experiments")

    assert response.status_code == 200
    assert response.json() == {"experiments": [], "count": 0}


def test_experiments_returns_empty_list_for_empty_or_malformed_file(
    monkeypatch,
    tmp_path,
) -> None:
    _, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    (reports_dir / "experiments.json").write_text("", encoding="utf-8")
    empty_response = client.get("/experiments")
    (reports_dir / "experiments.json").write_text("{bad json", encoding="utf-8")
    malformed_response = client.get("/experiments")

    assert empty_response.status_code == 200
    assert empty_response.json() == {"experiments": [], "count": 0}
    assert malformed_response.status_code == 200
    assert malformed_response.json() == {"experiments": [], "count": 0}


def test_experiments_filtering_by_model_type_and_ticker(monkeypatch, tmp_path) -> None:
    _, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    records = [
        {
            "experiment_id": "rf",
            "created_at": "2024-01-02T00:00:00+00:00",
            "status": "completed",
            "model_type": "random_forest",
            "tickers": ["SPY", "QQQ"],
            "artifact_path": "missing-rf.joblib",
        },
        {
            "experiment_id": "lr",
            "created_at": "2024-01-01T00:00:00+00:00",
            "status": "completed",
            "model_type": "logistic_regression",
            "tickers": ["AAPL"],
            "artifact_path": "missing-lr.joblib",
        },
    ]
    (reports_dir / "experiments.json").write_text(json.dumps(records), encoding="utf-8")
    client = TestClient(app)

    response = client.get("/experiments?model_type=random_forest&ticker=SPY")

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["experiments"][0]["experiment_id"] == "rf"


def test_metrics_latest_loads_metrics_json(monkeypatch, tmp_path) -> None:
    _, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    artifact_path = tmp_path / "models" / "model.joblib"
    artifact_path.parent.mkdir()
    artifact_path.write_text("exists", encoding="utf-8")
    experiment_id = "exp123"
    (reports_dir / "experiments.json").write_text(
        json.dumps(
            [
                {
                    "experiment_id": experiment_id,
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "status": "completed",
                    "model_type": "random_forest",
                    "tickers": ["SPY"],
                    "artifact_path": str(artifact_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    (reports_dir / f"metrics_{experiment_id}.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "model_type": "random_forest",
                "tickers": ["SPY"],
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "per_class": {},
                "confusion_matrix": [],
                "label_names": [],
                "feature_importance": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["experiment_id"] == experiment_id
    assert response.json()["accuracy"] == 0.5


def test_metrics_returns_404_when_unavailable(monkeypatch, tmp_path) -> None:
    configure_api_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 404


def test_metrics_returns_minimal_payload_from_partial_experiment_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    _, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    artifact_path = tmp_path / "models" / "model.joblib"
    artifact_path.parent.mkdir()
    artifact_path.write_text("exists", encoding="utf-8")
    (reports_dir / "experiments.json").write_text(
        json.dumps(
            [
                {
                    "experiment_id": "partial",
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "status": "completed",
                    "model_type": "random_forest",
                    "tickers": ["SPY"],
                    "artifact_path": str(artifact_path),
                    "accuracy": 0.7,
                    "macro_f1": 0.6,
                }
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["accuracy"] == 0.7
    assert response.json()["macro_f1"] == 0.6
    assert response.json()["warnings"]


def test_regime_returns_409_when_no_model_exists(monkeypatch, tmp_path) -> None:
    configure_api_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/regime/SPY")

    assert response.status_code == 409


def test_regime_returns_404_when_cached_data_is_missing(monkeypatch, tmp_path) -> None:
    processed_dir, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    create_trained_api_fixture(tmp_path, processed_dir, reports_dir)
    for path in processed_dir.glob("*"):
        path.unlink()
    client = TestClient(app)

    response = client.get("/regime/SPY")

    assert response.status_code == 404


def test_regime_returns_409_when_required_feature_columns_are_missing(
    monkeypatch,
    tmp_path,
) -> None:
    processed_dir, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    create_trained_api_fixture(tmp_path, processed_dir, reports_dir)
    feature_path = next(processed_dir.glob("*_labeled_*.csv"))
    data = pd.read_csv(feature_path).drop(columns=["return_20d"])
    data.to_csv(feature_path, index=False)
    client = TestClient(app)

    response = client.get("/regime/SPY")

    assert response.status_code == 409


def test_regime_returns_prediction_response(monkeypatch, tmp_path) -> None:
    processed_dir, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    result = create_trained_api_fixture(tmp_path, processed_dir, reports_dir)
    client = TestClient(app)

    response = client.get("/regime/SPY")

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "SPY"
    assert body["experiment_id"] == result.experiment_id
    assert body["regime"] in {
        "stable_growth",
        "volatile_recovery",
        "sideways_defensive",
        "stress_selloff",
    }
    assert set(body["key_signals"]) == {
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "ma_50_distance",
        "ma_200_distance",
        "drawdown_60d",
        "volume_change_20d",
    }


def test_history_returns_rule_labels_without_model_warning(monkeypatch, tmp_path) -> None:
    processed_dir, _, _ = configure_api_dirs(monkeypatch, tmp_path)
    write_labeled_files(make_labeled_data(periods=4), processed_dir)
    client = TestClient(app)

    response = client.get("/history/SPY?limit=2")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["total_available"] == 4
    assert body["rows"][0]["rule_label"] is not None
    assert body["rows"][0]["predicted_regime"] is None
    assert body["warnings"]


def test_history_returns_predicted_regime_when_model_exists(monkeypatch, tmp_path) -> None:
    processed_dir, reports_dir, _ = configure_api_dirs(monkeypatch, tmp_path)
    create_trained_api_fixture(tmp_path, processed_dir, reports_dir)
    client = TestClient(app)

    response = client.get("/history/SPY?limit=3")

    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 3
    assert all(row["predicted_regime"] is not None for row in rows)


def test_history_default_limit_returns_latest_100_rows_in_chronological_order(
    monkeypatch,
    tmp_path,
) -> None:
    processed_dir, _, _ = configure_api_dirs(monkeypatch, tmp_path)
    write_labeled_files(make_labeled_data(periods=150), processed_dir)
    client = TestClient(app)

    response = client.get("/history/SPY")

    assert response.status_code == 200
    body = response.json()
    dates = [row["date"] for row in body["rows"]]
    assert body["count"] == DEFAULT_HISTORY_LIMIT
    assert body["total_available"] == 150
    assert dates[0] == "2024-02-20"
    assert dates[-1] == "2024-05-29"
    assert dates == sorted(dates)


def test_history_overly_large_limit_is_capped(monkeypatch, tmp_path) -> None:
    processed_dir, _, _ = configure_api_dirs(monkeypatch, tmp_path)
    write_labeled_files(make_labeled_data(periods=1_200), processed_dir)
    client = TestClient(app)

    response = client.get("/history/SPY?limit=5000")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == MAX_HISTORY_LIMIT
    assert body["total_available"] == 1_200
    assert any("capped" in warning for warning in body["warnings"])


def test_history_supports_limit_start_date_and_end_date_filters(
    monkeypatch,
    tmp_path,
) -> None:
    processed_dir, _, _ = configure_api_dirs(monkeypatch, tmp_path)
    write_labeled_files(make_labeled_data(periods=10), processed_dir)
    client = TestClient(app)

    response = client.get(
        "/history/SPY?start_date=2024-01-03&end_date=2024-01-08&limit=3"
    )

    assert response.status_code == 200
    body = response.json()
    dates = [row["date"] for row in body["rows"]]
    assert body["count"] == 3
    assert body["total_available"] == 6
    assert dates == ["2024-01-06", "2024-01-07", "2024-01-08"]


def test_invalid_ticker_and_date_behavior(monkeypatch, tmp_path) -> None:
    processed_dir, _, _ = configure_api_dirs(monkeypatch, tmp_path)
    write_labeled_files(make_labeled_data(periods=4), processed_dir)
    client = TestClient(app)

    invalid_ticker = client.get("/history/BAD!")
    invalid_dates = client.get("/history/SPY?start_date=2024-02-01&end_date=2024-01-01")

    assert invalid_ticker.status_code == 422
    assert invalid_dates.status_code == 400
