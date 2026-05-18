"""Regime prediction and history routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas import HistoryResponse, RegimeResponse
from src.predict import (
    CachedDataNotFoundError,
    DEFAULT_HISTORY_LIMIT,
    InvalidTickerFormatError,
    NoModelAvailableError,
    PredictionInputError,
    history_for_ticker,
    predict_latest_regime,
)

router = APIRouter(tags=["regime"])


@router.get("/regime/{ticker}", response_model=RegimeResponse)
def get_regime(ticker: str) -> RegimeResponse:
    """Return the latest cached model regime prediction for a ticker."""
    try:
        return RegimeResponse(**predict_latest_regime(ticker))
    except InvalidTickerFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NoModelAvailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CachedDataNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PredictionInputError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/history/{ticker}", response_model=HistoryResponse)
def get_history(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=5_000),
) -> HistoryResponse:
    """Return historical rule labels and optional predictions for a ticker."""
    try:
        return HistoryResponse(
            **history_for_ticker(
                ticker,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        )
    except InvalidTickerFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CachedDataNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
