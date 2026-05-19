"""Tests for /reports/* read-only endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import src.report_loader as report_loader
from app.main import app


def _configure_reports_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point report_loader at a temporary reports directory."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(report_loader, "REPORTS_DIR", reports_dir)
    return reports_dir


# ── Endpoint → (report_type, file prefix/pattern) mapping ──────────────

_ENDPOINTS = {
    "/reports/hmm/latest": ("hmm", "hmm_{ts}_report.json"),
    "/reports/forward-returns/latest": ("forward_returns", "forward_returns_{ts}.json"),
    "/reports/walk-forward/latest": ("walk_forward", "walk_forward_{ts}.json"),
    "/reports/model-comparison/latest": ("model_comparison", "model_comparison_{ts}.json"),
    "/reports/explain/latest": ("explain", "explain_{ts}.json"),
}


# ── 404 when no report exists ──────────────────────────────────────────


def test_hmm_returns_404_when_missing(monkeypatch, tmp_path) -> None:
    _configure_reports_dir(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/reports/hmm/latest")

    assert response.status_code == 404
    assert "hmm" in response.json()["detail"].lower()


def test_forward_returns_returns_404_when_missing(monkeypatch, tmp_path) -> None:
    _configure_reports_dir(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/reports/forward-returns/latest")

    assert response.status_code == 404


def test_walk_forward_returns_404_when_missing(monkeypatch, tmp_path) -> None:
    _configure_reports_dir(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/reports/walk-forward/latest")

    assert response.status_code == 404


def test_model_comparison_returns_404_when_missing(monkeypatch, tmp_path) -> None:
    _configure_reports_dir(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/reports/model-comparison/latest")

    assert response.status_code == 404


def test_explain_returns_404_when_missing(monkeypatch, tmp_path) -> None:
    _configure_reports_dir(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/reports/explain/latest")

    assert response.status_code == 404


# ── Success: returns the newest matching report ────────────────────────


def _write_report(reports_dir: Path, filename: str, payload: dict) -> Path:
    path = reports_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def test_hmm_returns_latest_report(monkeypatch, tmp_path) -> None:
    reports_dir = _configure_reports_dir(monkeypatch, tmp_path)
    old = {"analysis_type": "hidden_markov_model", "n_states": 3, "version": "old"}
    new = {"analysis_type": "hidden_markov_model", "n_states": 4, "version": "new"}
    _write_report(reports_dir, "hmm_20240101T000000_hmm_3_states_report.json", old)
    _write_report(reports_dir, "hmm_20240201T000000_hmm_4_states_report.json", new)
    client = TestClient(app)

    response = client.get("/reports/hmm/latest")

    assert response.status_code == 200
    assert response.json()["version"] == "new"
    assert response.json()["n_states"] == 4


def test_forward_returns_returns_latest_report(monkeypatch, tmp_path) -> None:
    reports_dir = _configure_reports_dir(monkeypatch, tmp_path)
    payload = {"analysis_type": "forward_return_regime_analysis", "rows": 1000}
    _write_report(reports_dir, "forward_returns_20240301T000000.json", payload)
    client = TestClient(app)

    response = client.get("/reports/forward-returns/latest")

    assert response.status_code == 200
    assert response.json()["rows"] == 1000


def test_walk_forward_returns_latest_report(monkeypatch, tmp_path) -> None:
    reports_dir = _configure_reports_dir(monkeypatch, tmp_path)
    payload = {"analysis_type": "walk_forward_validation", "model_type": "random_forest"}
    _write_report(reports_dir, "walk_forward_20240401T000000.json", payload)
    client = TestClient(app)

    response = client.get("/reports/walk-forward/latest")

    assert response.status_code == 200
    assert response.json()["model_type"] == "random_forest"


def test_model_comparison_returns_latest_report(monkeypatch, tmp_path) -> None:
    reports_dir = _configure_reports_dir(monkeypatch, tmp_path)
    payload = {"analysis_type": "model_comparison", "best_model_type": "logistic_regression"}
    _write_report(reports_dir, "model_comparison_20240501T000000.json", payload)
    client = TestClient(app)

    response = client.get("/reports/model-comparison/latest")

    assert response.status_code == 200
    assert response.json()["best_model_type"] == "logistic_regression"


def test_explain_returns_latest_report(monkeypatch, tmp_path) -> None:
    reports_dir = _configure_reports_dir(monkeypatch, tmp_path)
    payload = {"analysis_type": "model_explanation", "experiment_id": "exp_123"}
    _write_report(reports_dir, "explain_20240601T000000.json", payload)
    client = TestClient(app)

    response = client.get("/reports/explain/latest")

    assert response.status_code == 200
    assert response.json()["experiment_id"] == "exp_123"


# ── Edge: picks newest when multiple reports exist ─────────────────────


def test_returns_newest_of_multiple_reports(monkeypatch, tmp_path) -> None:
    reports_dir = _configure_reports_dir(monkeypatch, tmp_path)
    _write_report(
        reports_dir,
        "forward_returns_20240101T000000.json",
        {"version": "old"},
    )
    _write_report(
        reports_dir,
        "forward_returns_20240601T000000.json",
        {"version": "newest"},
    )
    _write_report(
        reports_dir,
        "forward_returns_20240301T000000.json",
        {"version": "middle"},
    )
    client = TestClient(app)

    response = client.get("/reports/forward-returns/latest")

    assert response.status_code == 200
    assert response.json()["version"] == "newest"


# ── Unit: report_loader.latest_report raises for unknown types ─────────


def test_latest_report_raises_for_unknown_type(tmp_path) -> None:
    import pytest

    with pytest.raises(report_loader.ReportNotFoundError, match="Unknown report type"):
        report_loader.latest_report("nonexistent", reports_dir=tmp_path)
