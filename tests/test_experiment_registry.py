"""Tests for the lightweight SQLite experiment registry."""

from __future__ import annotations

import json

import pytest

from src.experiment_registry import (
    RegistryExperimentNotFoundError,
    activate_experiment,
    active_experiment,
    get_experiment,
    list_experiments,
    main,
    registry_exists,
    update_experiment_metrics,
    upsert_experiment,
)
from src.predict import latest_valid_experiment
from src.train import train_model
from tests.test_train import make_labeled_data


def make_registry_record(tmp_path, experiment_id: str = "exp1") -> dict[str, object]:
    artifact_path = tmp_path / "models" / f"{experiment_id}.joblib"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text("placeholder", encoding="utf-8")
    return {
        "experiment_id": experiment_id,
        "created_at": "2024-01-01T00:00:00+00:00",
        "status": "completed",
        "model_type": "random_forest",
        "tickers": ["SPY", "QQQ"],
        "artifact_path": str(artifact_path),
        "feature_version": "v1",
        "labeling_version": "v1",
        "training_date_range": {
            "start_date": "2020-01-01",
            "end_date": "2023-01-01",
        },
        "test_date_range": {
            "start_date": "2023-01-02",
            "end_date": "2024-01-01",
        },
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "environment": {"sklearn_version": "1.0.0"},
    }


def test_upsert_and_get_experiment_round_trip(tmp_path) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"
    record = make_registry_record(tmp_path)

    upsert_experiment(record, db_path=db_path)
    loaded = get_experiment("exp1", db_path=db_path)

    assert registry_exists(db_path)
    assert loaded["experiment_id"] == "exp1"
    assert loaded["tickers"] == ["SPY", "QQQ"]
    assert loaded["metrics"]["accuracy"] == 0.9
    assert loaded["environment"]["sklearn_version"] == "1.0.0"
    assert loaded["active"] is False


def test_activate_experiment_sets_single_active_record(tmp_path) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"
    upsert_experiment(make_registry_record(tmp_path, "old"), db_path=db_path)
    upsert_experiment(make_registry_record(tmp_path, "new"), db_path=db_path)

    activated = activate_experiment("new", db_path=db_path)
    old = get_experiment("old", db_path=db_path)
    current = active_experiment(db_path=db_path)

    assert activated["experiment_id"] == "new"
    assert old["active"] is False
    assert current is not None
    assert current["experiment_id"] == "new"


def test_active_experiment_ignores_missing_artifact(tmp_path) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"
    record = make_registry_record(tmp_path)
    record["artifact_path"] = str(tmp_path / "missing.joblib")
    upsert_experiment(record, db_path=db_path, active=True)

    assert active_experiment(db_path=db_path) is None
    assert active_experiment(db_path=db_path, require_valid_artifact=False) is not None


def test_list_experiments_filters_by_model_type_and_ticker(tmp_path) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"
    upsert_experiment(make_registry_record(tmp_path, "rf"), db_path=db_path)
    lr = make_registry_record(tmp_path, "lr")
    lr["model_type"] = "logistic_regression"
    lr["tickers"] = ["AAPL"]
    upsert_experiment(lr, db_path=db_path)

    records = list_experiments(
        db_path=db_path,
        model_type="logistic_regression",
        ticker="AAPL",
    )

    assert [record["experiment_id"] for record in records] == ["lr"]


def test_update_experiment_metrics_merges_metrics(tmp_path) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"
    upsert_experiment(make_registry_record(tmp_path), db_path=db_path)

    update_experiment_metrics(
        "exp1",
        {"accuracy": 0.95, "macro_f1": 0.85},
        metrics_path=tmp_path / "reports" / "metrics_exp1.json",
        db_path=db_path,
    )
    loaded = get_experiment("exp1", db_path=db_path)

    assert loaded["metrics"]["accuracy"] == 0.95
    assert loaded["metrics"]["macro_f1"] == 0.85
    assert loaded["metrics_path"].endswith("metrics_exp1.json")


def test_latest_valid_experiment_prefers_active_sqlite_record(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    json_artifact = tmp_path / "json.joblib"
    json_artifact.write_text("placeholder", encoding="utf-8")
    (reports_dir / "experiments.json").write_text(
        json.dumps(
            [
                {
                    "experiment_id": "json-latest",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "status": "completed",
                    "model_type": "random_forest",
                    "tickers": ["SPY"],
                    "artifact_path": str(json_artifact),
                }
            ]
        ),
        encoding="utf-8",
    )
    upsert_experiment(
        make_registry_record(tmp_path, "sqlite-active"),
        db_path=reports_dir / "regimelab.db",
        active=True,
    )

    selected = latest_valid_experiment(reports_dir)

    assert selected["experiment_id"] == "sqlite-active"


def test_latest_valid_experiment_falls_back_to_json_when_active_invalid(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    json_artifact = tmp_path / "json.joblib"
    json_artifact.write_text("placeholder", encoding="utf-8")
    (reports_dir / "experiments.json").write_text(
        json.dumps(
            [
                {
                    "experiment_id": "json-latest",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "status": "completed",
                    "model_type": "random_forest",
                    "tickers": ["SPY"],
                    "artifact_path": str(json_artifact),
                }
            ]
        ),
        encoding="utf-8",
    )
    invalid = make_registry_record(tmp_path, "sqlite-active")
    invalid["artifact_path"] = str(tmp_path / "missing.joblib")
    upsert_experiment(invalid, db_path=reports_dir / "regimelab.db", active=True)

    selected = latest_valid_experiment(reports_dir)

    assert selected["experiment_id"] == "json-latest"


def test_training_writes_file_and_sqlite_registry(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    result = train_model(
        make_labeled_data(periods=24),
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=reports_dir,
        created_at="2024-05-01T00:00:00+00:00",
    )

    assert (reports_dir / "experiments.json").exists()
    loaded = get_experiment(result.experiment_id, db_path=reports_dir / "regimelab.db")
    assert loaded["experiment_id"] == result.experiment_id
    assert loaded["artifact_path"] == str(result.artifact_path)


def test_registry_cli_list_show_and_activate(tmp_path, capsys) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"
    upsert_experiment(make_registry_record(tmp_path), db_path=db_path)

    list_code = main(["--db-path", str(db_path), "list"])
    listed = json.loads(capsys.readouterr().out)
    show_code = main(["--db-path", str(db_path), "show", "exp1"])
    shown = json.loads(capsys.readouterr().out)
    activate_code = main(["--db-path", str(db_path), "activate", "exp1"])
    activated = capsys.readouterr().out

    assert list_code == 0
    assert listed[0]["experiment_id"] == "exp1"
    assert show_code == 0
    assert shown["experiment_id"] == "exp1"
    assert activate_code == 0
    assert "activated: exp1" in activated


def test_get_missing_experiment_raises_clear_error(tmp_path) -> None:
    db_path = tmp_path / "reports" / "regimelab.db"

    with pytest.raises(RegistryExperimentNotFoundError):
        get_experiment("missing", db_path=db_path)
