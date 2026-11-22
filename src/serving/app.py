"""API de predicción de demanda MiBici con FastAPI.

Endpoints:
    GET  /health       estado del servicio y versión del modelo cargado
    GET  /model-info   origen del modelo (URI del Model Registry, run, hash, métricas)
    POST /predict      demanda estimada por estación y hora
    GET  /docs         documentación interactiva (Swagger / OpenAPI)

Ejecutar localmente:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Variables de entorno:
    MODEL_DIR        carpeta del paquete de despliegue (por defecto models/champion)
    PREDICTION_LOG   archivo JSONL donde se registra cada solicitud (opcional, para monitoreo)
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.config import PROJ_ROOT
from src.serving.schemas import (ErrorResponse, HealthResponse, ModelInfoResponse, PredictRequest,
                                 PredictResponse)
from src.serving.service import ModelService, PredictionError

logger = logging.getLogger("mibici.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

API_VERSION = "1.0.0"
state: dict = {"service": None, "error": None}


def model_dir() -> Path:
    path = Path(os.getenv("MODEL_DIR", "models/champion"))
    return path if path.is_absolute() else PROJ_ROOT / path


def load_service() -> Optional[ModelService]:
    try:
        state["service"] = ModelService(model_dir())
        state["error"] = None
        card = state["service"].card
        logger.info("Modelo cargado: %s (sha256 %s)", card.get("model_uri"), card.get("model_sha256", "")[:12])
    except Exception as exc:  # el servicio arranca y /health reporta el error
        state["service"], state["error"] = None, str(exc)
        logger.error("No se pudo cargar el modelo: %s", exc)
    return state["service"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_service()
    yield


app = FastAPI(
    title="MiBici Demand API (Equipo 19)",
    version=API_VERSION,
    description="Predicción de extracciones de bicicletas por estación y hora en MiBici Guadalajara. "
                "Sirve el modelo champion exportado desde el MLflow Model Registry.",
    lifespan=lifespan,
)


def get_service() -> ModelService:
    if state["service"] is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {state['error']}")
    return state["service"]


def log_prediction(record: dict) -> None:
    """Registro de cada solicitud (entradas, salidas, versión y latencia) para auditoría y monitoreo."""
    logger.info("predict %s", json.dumps({k: record[k] for k in ("model_version", "n", "latency_ms")}))
    path = os.getenv("PREDICTION_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Operación"])
def health() -> HealthResponse:
    service = state["service"]
    if service is None:
        return HealthResponse(status="error", model_loaded=False, detail=state["error"])
    return HealthResponse(status="ok", model_loaded=True, model_version=service.card.get("model_version"))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Operación"])
def model_info() -> dict:
    return get_service().info()


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"],
          responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def predict(request: PredictRequest) -> PredictResponse:
    service = get_service()
    start = time.perf_counter()
    instances = [inst.model_dump() for inst in request.instances]
    try:
        predictions = service.predict(instances)
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    card = service.card
    log_prediction({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model_uri": card.get("model_uri"), "model_version": card.get("model_version"),
                    "n": len(predictions), "latency_ms": latency_ms, "predictions": predictions})
    return PredictResponse(model_name=card.get("registered_model_name"), model_version=card.get("model_version"),
                           model_uri=card.get("model_uri"), n_predictions=len(predictions),
                           predictions=predictions)
