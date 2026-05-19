"""RegimeLab FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.experiments import router as experiments_router
from app.routes.health import router as health_router
from app.routes.metrics import router as metrics_router
from app.routes.regime import router as regime_router
from app.routes.reports import router as reports_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RegimeLab API",
        description="Backend API for market regime classification workflows.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(regime_router)
    app.include_router(metrics_router)
    app.include_router(experiments_router)
    app.include_router(reports_router)
    return app


app = create_app()
