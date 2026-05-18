"""Shared API schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    model_config = ConfigDict(frozen=True)

    status: str
    service: str
    model_loaded: bool
    version: str
    latest_experiment_id: str | None = None


class RegimeResponse(BaseModel):
    """Latest regime prediction response."""

    ticker: str
    as_of: str
    regime: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    key_signals: dict[str, float | None]
    experiment_id: str
    model_id: str
    warnings: list[str] = Field(default_factory=list)


class HistoryRow(BaseModel):
    """One historical regime row."""

    date: str
    rule_label: str | None = None
    predicted_regime: str | None = None
    close: float | None = None
    return_20d: float | None = None
    volatility_20d: float | None = None
    drawdown_60d: float | None = None


class HistoryResponse(BaseModel):
    """Historical labels and predictions response."""

    ticker: str
    rows: list[HistoryRow]
    count: int
    warnings: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """Model evaluation metrics response."""

    experiment_id: str
    model_type: str | None = None
    tickers: list[str] = Field(default_factory=list)
    train_period: dict[str, str] | None = None
    test_period: dict[str, str] | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    per_class: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    label_names: list[str] = Field(default_factory=list)
    feature_importance: list[dict[str, float | str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExperimentsResponse(BaseModel):
    """Experiment list response."""

    experiments: list[dict[str, Any]]
    count: int
