"""Esquemas de entrada y salida de la API (Pydantic). FastAPI los publica en /docs (OpenAPI)."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MAX_INSTANCES = 1000


class PredictionInstance(BaseModel):
    """Una combinación estación-hora a predecir."""

    model_config = ConfigDict(extra="forbid")

    station_id: int = Field(..., gt=0, description="Id de la estación de MiBici (columna Origen_Id)", examples=[2])
    fecha_hora: datetime = Field(
        ..., description="Hora a predecir, hora local de Guadalajara (se redondea a la hora). "
                         "Si incluye zona horaria se convierte a UTC-06:00.",
        examples=["2026-01-01T08:00:00"])
    demand_lag_24h: Optional[float] = Field(None, ge=0, description="Opcional: demanda de la estación 24 h antes")
    demand_lag_168h: Optional[float] = Field(None, ge=0, description="Opcional: demanda de la estación 168 h antes")
    demand_same_hour_mean_7d: Optional[float] = Field(
        None, ge=0, description="Opcional: promedio de la demanda a la misma hora en los 7 días previos")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "instances": [
            {"station_id": 2, "fecha_hora": "2025-12-31T18:00:00"},
            {"station_id": 3, "fecha_hora": "2026-01-01T08:00:00",
             "demand_lag_24h": 2, "demand_lag_168h": 1, "demand_same_hour_mean_7d": 1.4},
        ]}]})

    instances: List[PredictionInstance] = Field(..., min_length=1, max_length=MAX_INSTANCES)


class Prediction(BaseModel):
    station_id: int
    fecha_hora: str
    demand_pred: float = Field(..., description="Extracciones de bicicletas estimadas en esa hora")
    lags_source: Literal["request", "feature_store"]
    demand_lag_24h: float
    demand_lag_168h: float
    demand_same_hour_mean_7d: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: Optional[str]
    model_version: Optional[str]
    model_uri: Optional[str]
    n_predictions: int
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "error"]
    model_loaded: bool
    model_version: Optional[str] = None
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    registered_model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_uri: Optional[str] = None
    model_sha256: Optional[str] = None
    predictable_window: Dict[str, str]
