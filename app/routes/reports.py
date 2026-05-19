"""Report routes — read-only access to generated analysis reports."""

from typing import Any

from fastapi import APIRouter, HTTPException

import src.report_loader as report_loader

router = APIRouter(prefix="/reports", tags=["reports"])


def _latest(report_type: str) -> dict[str, Any]:
    """Shared handler: load latest report or raise 404."""
    try:
        return report_loader.latest_report(report_type)
    except report_loader.ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/hmm/latest")
def get_latest_hmm_report() -> dict[str, Any]:
    """Return the latest HMM analysis report."""
    return _latest("hmm")


@router.get("/forward-returns/latest")
def get_latest_forward_returns_report() -> dict[str, Any]:
    """Return the latest forward-returns analysis report."""
    return _latest("forward_returns")


@router.get("/walk-forward/latest")
def get_latest_walk_forward_report() -> dict[str, Any]:
    """Return the latest walk-forward validation report."""
    return _latest("walk_forward")


@router.get("/model-comparison/latest")
def get_latest_model_comparison_report() -> dict[str, Any]:
    """Return the latest model comparison report."""
    return _latest("model_comparison")


@router.get("/explain/latest")
def get_latest_explain_report() -> dict[str, Any]:
    """Return the latest model explanation report."""
    return _latest("explain")
