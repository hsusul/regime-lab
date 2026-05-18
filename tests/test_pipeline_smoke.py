"""End-to-end synthetic smoke test for the local MVP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import src.data as data_module
import src.predict as predict_service
from app.main import app
from src.data import main as data_main
from src.evaluate import main as evaluate_main
from src.features import main as features_main
from src.labeling import main as labeling_main
from src.train import main as train_main


@dataclass
class SmokeProvider:
    frame: pd.DataFrame

    def download(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.frame.copy()


def make_smoke_provider_frame(rows: int = 230) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    adj_close = [100 + index * 0.5 for index in range(rows)]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": adj_close,
            "High": [value + 1 for value in adj_close],
            "Low": [value - 1 for value in adj_close],
            "Close": adj_close,
            "Adj Close": adj_close,
            "Volume": [1_000_000 + index for index in range(rows)],
        }
    )


def test_synthetic_cli_pipeline_smoke(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    provider = SmokeProvider(make_smoke_provider_frame())
    monkeypatch.setattr(data_module, "YFinanceProvider", lambda: provider)

    assert (
        data_main(
            [
                "--tickers",
                "SPY",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-08-17",
                "--raw-data-dir",
                str(raw_dir),
            ]
        )
        == 0
    )
    assert (
        features_main(
            [
                "--tickers",
                "SPY",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-08-17",
                "--raw-data-dir",
                str(raw_dir),
                "--processed-data-dir",
                str(processed_dir),
            ]
        )
        == 0
    )
    assert (
        labeling_main(
            [
                "--tickers",
                "SPY",
                "--processed-data-dir",
                str(processed_dir),
            ]
        )
        == 0
    )
    assert (
        train_main(
            [
                "--model-type",
                "random_forest",
                "--tickers",
                "SPY",
                "--processed-data-dir",
                str(processed_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
            ]
        )
        == 0
    )
    assert (
        evaluate_main(
            [
                "--experiment-id",
                "latest",
                "--processed-data-dir",
                str(processed_dir),
                "--reports-dir",
                str(reports_dir),
            ]
        )
        == 0
    )

    monkeypatch.setattr(predict_service, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(predict_service, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(predict_service, "RAW_DATA_DIR", raw_dir)
    client = TestClient(app)

    regime = client.get("/regime/SPY")
    metrics = client.get("/metrics")

    assert regime.status_code == 200
    assert regime.json()["ticker"] == "SPY"
    assert metrics.status_code == 200
    assert metrics.json()["experiment_id"] == regime.json()["experiment_id"]
