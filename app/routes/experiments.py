"""Experiment metadata routes."""

from fastapi import APIRouter, HTTPException, Query

from app.schemas import ExperimentsResponse
from src.predict import InvalidTickerFormatError, filter_experiments

router = APIRouter(tags=["experiments"])


@router.get("/experiments", response_model=ExperimentsResponse)
def get_experiments(
    limit: int | None = Query(default=None, ge=1, le=1_000),
    model_type: str | None = None,
    ticker: str | None = None,
) -> ExperimentsResponse:
    """List recorded training runs."""
    try:
        experiments = filter_experiments(
            limit=limit,
            model_type=model_type,
            ticker=ticker,
        )
    except InvalidTickerFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ExperimentsResponse(experiments=experiments, count=len(experiments))
