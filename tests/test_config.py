"""Tests for lightweight configuration support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.data as data_module
import src.train as train_module
from src.config import (
    ConfigNotFoundError,
    cli_or_config,
    cli_or_config_path,
    get_config_value,
    load_config,
)
from src.data import DataLoadResult
from src.labeling import LabelingThresholds, thresholds_from_config
from src.train import TrainingResult


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
project:
  supported_tickers: [AAA, BBB]
data:
  start_date: "2020-01-01"
  end_date: "2021-01-01"
labeling:
  thresholds:
    stress_return_20d: -0.05
paths:
  raw_data_dir: "custom/raw"
  processed_data_dir: "custom/processed"
  models_dir: "custom/models"
  reports_dir: "custom/reports"
model:
  default_model_type: "random_forest"
""",
        encoding="utf-8",
    )
    return path


def test_load_config_reads_yaml_mapping(tmp_path) -> None:
    config = load_config(write_config(tmp_path))

    assert get_config_value(config, "project.supported_tickers") == ["AAA", "BBB"]
    assert get_config_value(config, "data.start_date") == "2020-01-01"
    assert get_config_value(config, "missing.value", "fallback") == "fallback"


def test_load_config_returns_empty_when_omitted() -> None:
    assert load_config(None) == {}


def test_load_config_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_cli_or_config_prefers_cli_value(tmp_path) -> None:
    config = load_config(write_config(tmp_path))

    assert cli_or_config("CLI", config, "data.start_date", "DEFAULT") == "CLI"
    assert cli_or_config(None, config, "data.start_date", "DEFAULT") == "2020-01-01"
    assert cli_or_config(None, config, "missing", "DEFAULT") == "DEFAULT"


def test_cli_or_config_path_resolves_config_path(tmp_path) -> None:
    config = load_config(write_config(tmp_path))

    assert cli_or_config_path(
        None,
        config,
        "paths.raw_data_dir",
        Path("data/raw"),
    ) == Path("custom/raw")
    assert cli_or_config_path(
        Path("cli/raw"),
        config,
        "paths.raw_data_dir",
        Path("data/raw"),
    ) == Path("cli/raw")


def test_thresholds_from_config_uses_configured_values(tmp_path) -> None:
    thresholds = thresholds_from_config(load_config(write_config(tmp_path)))

    assert isinstance(thresholds, LabelingThresholds)
    assert thresholds.stress_return_20d == -0.05
    assert thresholds.stress_drawdown_60d == LabelingThresholds().stress_drawdown_60d


def test_data_cli_uses_config_and_cli_overrides(monkeypatch, tmp_path, capsys) -> None:
    config_path = write_config(tmp_path)
    captured = {}

    def fake_load_tickers(
        tickers,
        start_date=None,
        end_date=None,
        *,
        provider=None,
        raw_data_dir=Path("data/raw"),
        force_refresh=False,
    ):
        captured.update(
            {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "raw_data_dir": raw_data_dir,
                "force_refresh": force_refresh,
            }
        )
        frame = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "adj_close": [1.0],
                "volume": [100],
                "ticker": ["ZZZ"],
            }
        )
        return [
            DataLoadResult(
                ticker="ZZZ",
                data=frame,
                cache_path=Path("custom/raw/ZZZ.csv"),
                source="cache",
            )
        ]

    monkeypatch.setattr(data_module, "load_tickers", fake_load_tickers)

    exit_code = data_module.main(
        [
            "--config",
            str(config_path),
            "--tickers",
            "ZZZ",
            "--start-date",
            "2019-01-01",
        ]
    )

    assert exit_code == 0
    assert captured["tickers"] == ["ZZZ"]
    assert captured["start_date"] == "2019-01-01"
    assert captured["end_date"] == "2021-01-01"
    assert captured["raw_data_dir"] == Path("custom/raw")
    assert "ZZZ" in capsys.readouterr().out


def test_train_cli_uses_config_and_cli_model_override(monkeypatch, tmp_path, capsys) -> None:
    config_path = write_config(tmp_path)
    captured = {}

    def fake_train_from_processed_files(
        tickers,
        model_type,
        *,
        processed_data_dir=Path("data/processed"),
        models_dir=Path("models"),
        reports_dir=Path("reports"),
    ):
        captured.update(
            {
                "tickers": tickers,
                "model_type": model_type,
                "processed_data_dir": processed_data_dir,
                "models_dir": models_dir,
                "reports_dir": reports_dir,
            }
        )
        return TrainingResult(
            experiment_id="exp",
            model_type=model_type,
            artifact_path=Path("custom/models/exp.joblib"),
            summary_path=Path("custom/reports/summary.json"),
            train_rows=10,
            test_rows=2,
            label_distribution={},
            metadata={},
        )

    monkeypatch.setattr(
        train_module,
        "train_from_processed_files",
        fake_train_from_processed_files,
    )

    exit_code = train_module.main(
        [
            "--config",
            str(config_path),
            "--model-type",
            "logistic_regression",
        ]
    )

    assert exit_code == 0
    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["model_type"] == "logistic_regression"
    assert captured["processed_data_dir"] == Path("custom/processed")
    assert captured["models_dir"] == Path("custom/models")
    assert captured["reports_dir"] == Path("custom/reports")
    assert "experiment_id: exp" in capsys.readouterr().out
