"""Esquemas de entrada y salida (Pydantic v2) del servicio de predicción de demanda.

Los nombres de los rezagos (`demand_lag_24h`, `demand_lag_168h`, `demand_same_hour_mean_7d`) son los que
genera `src.features.lag_columns` con la configuración actual (`features.lag_hours: [24, 168]` y
`features.same_hour_mean_days: 7`). Si la configuración o el modelo cambian, `src.serving.app` lo
detecta al arrancar (ver `LAG_FIELDS`) y no carga el modelo, con un mensaje explícito.

Desde la versión 1.1.0 los tres rezagos son opcionales: o se envían los tres o ninguno. Sin rezagos, el servicio
los calcula con el feature store del modelo (`models/champion/feature_store.csv.gz`); con los tres, la predicción
es la misma que en la versión 1.0.0.

Los ejemplos de los esquemas son reales: la primera fila de `models/reference_predictions.csv` (con rezagos) y
una hora de la ventana del feature store (sin rezagos), con sus respuestas.
"""

from __future__ import annotations

from datetime import datetime as dt
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

# Rezagos que el cliente envía en /predict (deben coincidir con src.features.lag_columns(config)).
LAG_FIELDS = ("demand_lag_24h", "demand_lag_168h", "demand_same_hour_mean_7d")

# Primera fila de models/reference_predictions.csv (modelo models:/mibici-demand-forecaster/1).
EXAMPLE_INSTANCE = {
    "station_id": 24,
    "datetime": "2025-11-08T10:00:00",
    "demand_lag_24h": 5.0,
    "demand_lag_168h": 0.0,
    "demand_same_hour_mean_7d": 3.5714285714285716,
}
EXAMPLE_RAW_ROW = {
    "sin_hour": 0.49999999999999994,
    "cos_hour": -0.8660254037844387,
    "sin_dayofweek": -0.9749279121818236,
    "cos_dayofweek": -0.2225209339563146,
    "latitude": 20.679389,
    "longitude": -103.35913,
    "capacity": 15,
    "demand_lag_24h": 5.0,
    "demand_lag_168h": 0.0,
    "demand_same_hour_mean_7d": 3.5714285714285716,
    "Origen_Id": 24,
    "is_weekend": 1,
    "is_holiday": 0,
}
EXAMPLE_PREDICTION = 2.6064224732591033
EXAMPLE_MODEL = {
    "name": "mibici-demand-forecaster",
    "version": 1,
    "uri": "models:/mibici-demand-forecaster/1",
    "alias": "champion",
    "algorithm": "hist_gradient_boosting",
}
# Sin rezagos: el servicio los toma del feature store (ventana 2025-12-24 00:00 a 2026-01-01 23:00).
EXAMPLE_INSTANCE_FEATURE_STORE = {"station_id": 24, "datetime": "2025-12-31T18:00:00"}
EXAMPLE_OUTPUT_REQUEST = {
    "station_id": 24, "datetime": "2025-11-08T10:00:00", "demand_pred": EXAMPLE_PREDICTION,
    "lags_source": "request", **{k: EXAMPLE_INSTANCE[k] for k in LAG_FIELDS},
}
EXAMPLE_OUTPUT_FEATURE_STORE = {
    "station_id": 24, "datetime": "2025-12-31T18:00:00", "demand_pred": 3.1631559452523517,
    "lags_source": "feature_store", "demand_lag_24h": 0.0, "demand_lag_168h": 1.0,
    "demand_same_hour_mean_7d": 1.8571428571428572,
}

_LAG_KWARGS: Dict[str, Any] = {"ge": 0, "allow_inf_nan": False, "strict": True}


# --------------------------------------------------------------------------- entrada
class PredictionInstance(BaseModel):
    """Una estación y la hora a predecir, con la demanda histórica que el cliente ya conoce."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_INSTANCE})

    station_id: int = Field(
        ..., strict=True, ge=1,
        description="Identificador de la estación de MiBici (`Origen_Id`); debe ser una de las estaciones "
                    "conocidas por el modelo (ver /model-info, `num_stations`).",
    )
    datetime: dt = Field(
        ...,
        description="Hora a predecir en ISO-8601, en hora local de Guadalajara. Sin zona horaria se interpreta "
                    "como hora local; con zona horaria (p. ej. `Z` o `-06:00`) se convierte a la hora local "
                    "(UTC-06:00, sin horario de verano). Se trunca a la hora: 10:45 predice el bloque 10:00-10:59.",
    )
    demand_lag_24h: Optional[float] = Field(
        None, **_LAG_KWARGS,
        description="Opcional. Viajes que salieron de la estación en la misma hora del día anterior (t - 24 h). "
                    "Se envían los tres rezagos o ninguno; sin rezagos se calculan con el feature store.",
    )
    demand_lag_168h: Optional[float] = Field(
        None, **_LAG_KWARGS,
        description="Opcional. Viajes que salieron de la estación en la misma hora de la semana anterior "
                    "(t - 168 h).",
    )
    demand_same_hour_mean_7d: Optional[float] = Field(
        None, **_LAG_KWARGS,
        description="Opcional. Promedio de viajes a la misma hora en los 7 días previos (t - 24 h, ..., t - 168 h).",
    )

    @field_validator("datetime", mode="before")
    @classmethod
    def _datetime_as_text(cls, value: Any) -> Any:
        """Solo se aceptan textos ISO-8601: un número (época Unix) sería ambiguo respecto a la zona horaria."""
        if not isinstance(value, (str, dt)):
            raise ValueError("debe ser un texto ISO-8601, p. ej. 2025-11-08T10:00:00")
        return value

    @model_validator(mode="after")
    def _all_or_no_lags(self) -> "PredictionInstance":
        """Los tres rezagos o ninguno: mezclar valores del cliente con los del feature store no es claro."""
        sent = [lag for lag in LAG_FIELDS if getattr(self, lag) is not None]
        if 0 < len(sent) < len(LAG_FIELDS):
            missing = [lag for lag in LAG_FIELDS if lag not in sent]
            raise PydanticCustomError(
                "lags_incomplete",
                "Se envían los tres rezagos o ninguno. Faltan: {missing}. Sin rezagos el servicio los calcula con "
                "el feature store.", {"missing": ", ".join(missing)})
        return self

    @property
    def has_lags(self) -> bool:
        return all(getattr(self, lag) is not None for lag in LAG_FIELDS)


class PredictRequest(BaseModel):
    """Lote de instancias a predecir (el límite superior está en `serving.max_batch_size`)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"instances": [EXAMPLE_INSTANCE, EXAMPLE_INSTANCE_FEATURE_STORE]}})

    instances: List[PredictionInstance] = Field(
        ..., min_length=1, description="Instancias a predecir (al menos una). Cada una con sus tres rezagos o sin "
                                       "ninguno; se pueden mezclar ambos tipos en el mismo lote.",
    )


class RawPredictRequest(BaseModel):
    """Filas con las columnas de entrada del modelo ya construidas (mismo formato que el entrenamiento)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"rows": [EXAMPLE_RAW_ROW]}})

    rows: List[Dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)]]] = Field(
        ..., min_length=1,
        description="Cada fila trae exactamente las columnas `input_columns` de /model-info. Las columnas "
                    "enteras (`capacity`, `Origen_Id`, `is_weekend`, `is_holiday`) deben ser valores enteros.",
    )


# --------------------------------------------------------------------------- salida
class ModelRef(BaseModel):
    """Versión del modelo del Model Registry que generó las predicciones."""

    model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"example": EXAMPLE_MODEL})

    name: str = Field(..., description="Nombre del modelo registrado en el MLflow Model Registry.")
    version: int = Field(..., description="Versión registrada del modelo.")
    uri: str = Field(..., description="URI del artefacto en el Registry (`models:/<nombre>/<versión>`).")
    alias: Optional[str] = Field(None, description="Alias con el que se exportó (p. ej. `champion`).")
    algorithm: Optional[str] = Field(None, description="Algoritmo del modelo.")


class Prediction(BaseModel):
    """Predicción para una instancia, con los rezagos que se usaron y su origen."""

    station_id: int = Field(..., description="Identificador de la estación.")
    datetime: str = Field(..., description="Hora local predicha, truncada a la hora (ISO-8601 sin zona horaria).")
    demand_pred: float = Field(..., description="Viajes esperados que salen de la estación en esa hora (>= 0).")
    lags_source: Literal["request", "feature_store"] = Field(
        ..., description="`request`: rezagos enviados por el cliente; `feature_store`: calculados por el servicio.")
    demand_lag_24h: float = Field(..., description="Rezago de 24 h usado en la predicción.")
    demand_lag_168h: float = Field(..., description="Rezago de 168 h usado en la predicción.")
    demand_same_hour_mean_7d: float = Field(..., description="Media de 7 días a la misma hora usada en la predicción.")


class PredictResponse(BaseModel):
    """Respuesta de /predict: una predicción por instancia, en el mismo orden de la petición."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "predictions": [EXAMPLE_OUTPUT_REQUEST, EXAMPLE_OUTPUT_FEATURE_STORE],
        "model": EXAMPLE_MODEL,
        "n_instances": 2,
    }})

    predictions: List[Prediction]
    model: ModelRef
    n_instances: int = Field(..., description="Número de instancias predichas.")


class RawPredictResponse(BaseModel):
    """Respuesta de /predict/raw: una predicción por fila, en el mismo orden de la petición."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "predictions": [EXAMPLE_PREDICTION], "model": EXAMPLE_MODEL, "n_instances": 1,
    }})

    predictions: List[float] = Field(..., description="Viajes esperados por fila (>= 0).")
    model: ModelRef
    n_instances: int = Field(..., description="Número de filas predichas.")


class HealthResponse(BaseModel):
    """Estado del servicio."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="`ok` si el modelo está cargado; `unavailable` en otro caso.")
    model_loaded: bool = Field(..., description="Indica si el modelo está listo para predecir.")
    model_uri: Optional[str] = Field(None, description="URI del modelo cargado.")
    model_sha256_verified: Optional[bool] = Field(
        None, description="True si el SHA-256 del archivo del modelo coincide con los metadatos.")
    service_version: str = Field(..., description="Versión del servicio (`serving.service_version`).")
    detail: Optional[str] = Field(None, description="Motivo por el que el modelo no está cargado, si aplica.")


class ErrorDetail(BaseModel):
    """Un error de validación concreto."""

    loc: List[Any] = Field(..., description="Ubicación del campo con error (p. ej. `[body, instances, 0, ...]`).")
    msg: str = Field(..., description="Descripción del error.")
    type: str = Field(..., description="Tipo de error de Pydantic o del servicio.")


class ErrorResponse(BaseModel):
    """Cuerpo común de todos los errores del servicio."""

    error: str = Field(..., description="Código corto: validation_error, unknown_station, model_unavailable, "
                                        "not_found, internal_error, ...")
    message: str = Field(..., description="Mensaje legible en español.")
    details: Optional[List[ErrorDetail]] = Field(None, description="Errores de validación (solo en 422).")
    unknown_station_ids: Optional[List[int]] = Field(None, description="Estaciones desconocidas (solo en 404).")
    predictable_window: Optional[Dict[str, str]] = Field(
        None, description="Horas que se pueden predecir sin rezagos (`start`, `end`); solo en el 422 "
                          "`outside_feature_store_window`.")
