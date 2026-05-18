"""Metrics route."""

from fastapi import APIRouter, HTTPException

from app.schemas import MetricsResponse
from src.evaluate import ExperimentNotFoundError
from src.predict import (
    CachedDataNotFoundError,
    NoModelAvailableError,
    load_metrics_payload,
)

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(experiment_id: str = "latest") -> MetricsResponse:
    """Return latest or requested evaluation metrics."""
    try:
        return MetricsResponse(**load_metrics_payload(experiment_id))
    except (CachedDataNotFoundError, ExperimentNotFoundError, NoModelAvailableError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
