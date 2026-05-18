"""Health check route."""

from fastapi import APIRouter

from app.schemas import HealthResponse
from src.predict import NoModelAvailableError, latest_valid_experiment

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic service status."""
    model_loaded = False
    latest_experiment_id = None
    try:
        experiment = latest_valid_experiment()
        model_loaded = True
        latest_experiment_id = str(experiment["experiment_id"])
    except NoModelAvailableError:
        pass

    return HealthResponse(
        status="ok",
        service="regimelab-api",
        model_loaded=model_loaded,
        version="0.1.0",
        latest_experiment_id=latest_experiment_id,
    )
