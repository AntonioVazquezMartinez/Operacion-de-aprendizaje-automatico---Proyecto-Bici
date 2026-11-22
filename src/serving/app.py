"""API de predicción de demanda horaria por estación (FastAPI) con el modelo champion exportado.

El modelo se carga desde una carpeta MLflow autocontenida (`serving.model_dir`, por defecto
`models/champion`) con `mlflow.sklearn.load_model`, sin la base `mlruns.db`. Las variables de calendario
y de estación se derivan con las mismas funciones de `src.features` que se usaron en el entrenamiento.

Uso:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
    MODEL_DIR=/ruta/al/modelo MODEL_METADATA=/ruta/metadata.json uvicorn src.serving.app:app

Variables de entorno (tienen prioridad sobre config.yaml; una ruta relativa se resuelve contra la
carpeta de trabajo):
    MODEL_DIR        carpeta del modelo MLflow (con stations.csv dentro).
    MODEL_METADATA   archivo champion_metadata.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import PROJ_ROOT, Config, load_config
from src.features import MiBiciFeatureEngineer, lag_columns
from src.serving.schemas import (
    LAG_FIELDS,
    ErrorResponse,
    HealthResponse,
    PredictionInstance,
    PredictRequest,
    PredictResponse,
    RawPredictRequest,
    RawPredictResponse,
)

logger = logging.getLogger("mibici.serving")

STATION_FEATURES = ("latitude", "longitude", "capacity")
INTEGER_SIGNATURE_TYPES = {"long", "integer"}

TAGS = [
    {"name": "Predicción", "description": "Predicción de la demanda horaria (viajes que salen de una estación)."},
    {"name": "Operación", "description": "Estado del servicio y metadatos del modelo cargado."},
]

ERROR_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    422: {"model": ErrorResponse,
          "description": "Entrada inválida (tipos, negativos, NaN/inf, fechas, tamaño del lote)."},
    503: {"model": ErrorResponse, "description": "El modelo no está cargado."},
    500: {"model": ErrorResponse, "description": "Error interno (sin detalles de la traza)."},
}


# --------------------------------------------------------------------------- errores
class ServingError(Exception):
    """Error controlado del servicio que se traduce a una respuesta JSON con el formato de ErrorResponse."""

    def __init__(self, status_code: int, error: str, message: str, **extra: Any):
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.extra = extra

    def body(self) -> Dict[str, Any]:
        return {"error": self.error, "message": self.message, **self.extra}


def _clean_loc(loc: Sequence[Any]) -> List[Any]:
    return [part if isinstance(part, (int, str)) else str(part) for part in loc]


# Traducción de los errores de Pydantic más comunes; los demás conservan el mensaje original.
SPANISH_MESSAGES = {
    "missing": "Campo obligatorio.",
    "extra_forbidden": "Campo no permitido.",
    "greater_than_equal": "Debe ser mayor o igual que {ge}.",
    "finite_number": "Debe ser un número finito (no se aceptan NaN ni infinito).",
    "float_type": "Debe ser un número.",
    "int_type": "Debe ser un número entero.",
    "too_short": "La lista debe tener al menos {min_length} elemento(s).",
    "list_type": "Debe ser una lista.",
    "dict_type": "Debe ser un objeto JSON.",
    "model_type": "Debe ser un objeto JSON.",
    "datetime_from_date_parsing": "Fecha y hora inválidas; usa ISO-8601, p. ej. 2025-11-08T10:00:00.",
    "datetime_parsing": "Fecha y hora inválidas; usa ISO-8601, p. ej. 2025-11-08T10:00:00.",
    "datetime_type": "Fecha y hora inválidas; usa ISO-8601, p. ej. 2025-11-08T10:00:00.",
    "json_invalid": "El cuerpo no es un JSON válido.",
}


def _error_message(error: Dict[str, Any]) -> str:
    template = SPANISH_MESSAGES.get(str(error.get("type", "")))
    if template is None:
        return str(error.get("msg", ""))
    try:
        return template.format(**(error.get("ctx") or {}))
    except (KeyError, IndexError):
        return str(error.get("msg", ""))


def _validation_details(errors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Errores de Pydantic sin `input` ni `ctx` (pueden ser grandes o no serializables)."""
    return [{"loc": _clean_loc(e.get("loc", ())), "msg": _error_message(e), "type": str(e.get("type", ""))}
            for e in errors]


# --------------------------------------------------------------------------- estado del modelo
@dataclass
class ModelState:
    """Modelo cargado y todo lo necesario para construir sus entradas."""

    model_dir: Path
    metadata_path: Path
    model_dir_label: str
    model: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stations: Optional[pd.DataFrame] = None
    station_ids: Set[int] = field(default_factory=set)
    input_columns: List[str] = field(default_factory=list)
    int_columns: Set[str] = field(default_factory=set)
    model_sha256: Optional[str] = None
    model_sha256_verified: Optional[bool] = None
    stations_sha256_verified: Optional[bool] = None
    loaded_at: Optional[str] = None
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.error is None

    def model_ref(self) -> Dict[str, Any]:
        md = self.metadata
        return {
            "name": md.get("registered_model_name"),
            "version": md.get("version"),
            "uri": md.get("model_uri"),
            "alias": md.get("alias"),
            "algorithm": md.get("algorithm"),
        }


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJ_ROOT / path


def resolve_paths(config: Config) -> Dict[str, Any]:
    """Rutas del modelo y de los metadatos: variable de entorno si existe; si no, `serving` de config.yaml."""
    serving = config["serving"]
    env_dir, env_md = os.environ.get("MODEL_DIR"), os.environ.get("MODEL_METADATA")
    model_dir = Path(env_dir).resolve() if env_dir else _resolve_config_path(serving["model_dir"])
    metadata = Path(env_md).resolve() if env_md else _resolve_config_path(serving["metadata_file"])
    return {
        "model_dir": model_dir,
        "metadata": metadata,
        "stations": model_dir / serving.get("stations_file", "stations.csv"),
        "model_dir_label": env_dir or serving["model_dir"],
    }


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 de un archivo (mismo algoritmo que src.tracking.calculate_sha256, sin depender de MLflow)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mlmodel(model_dir: Path) -> Dict[str, Any]:
    with open(model_dir / "MLmodel", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _signature_types(mlmodel: Dict[str, Any]) -> Dict[str, str]:
    """Tipo de cada columna según la firma del modelo MLflow (vacío si el modelo no tiene firma)."""
    inputs = (mlmodel.get("signature") or {}).get("inputs")
    if not inputs:
        return {}
    return {col["name"]: col.get("type", "") for col in json.loads(inputs) if "name" in col}


def calendar_feature_columns() -> List[str]:
    """Columnas que agrega `MiBiciFeatureEngineer.add_calendar_features` (se obtienen ejecutándola)."""
    probe = pd.DataFrame({"fecha_hora": pd.to_datetime(["2025-01-01 00:00:00"])})
    return [c for c in MiBiciFeatureEngineer.add_calendar_features(probe).columns if c != "fecha_hora"]


def check_schema_compatibility(config: Config, input_columns: Sequence[str], station_key: str) -> None:
    """Comprueba que el esquema de /predict alcanza para construir todas las entradas del modelo.

    Lanza ValueError con un mensaje explícito si los rezagos del esquema Pydantic no coinciden con los de
    la configuración o con los de los metadatos del modelo, o si el modelo pide una columna que el servicio
    no sabe construir.
    """
    expected_lags = lag_columns(config)
    if tuple(expected_lags) != LAG_FIELDS:
        raise ValueError(f"Los rezagos de config.yaml {expected_lags} no coinciden con los campos del esquema "
                         f"de /predict {list(LAG_FIELDS)} (src/serving/schemas.py: LAG_FIELDS).")
    derived = set(calendar_feature_columns()) | set(STATION_FEATURES) | {station_key}
    model_lags = [c for c in input_columns if c not in derived]
    if set(model_lags) != set(LAG_FIELDS):
        raise ValueError(f"Las columnas de entrada del modelo que no deriva el servicio {model_lags} no coinciden "
                         f"con los rezagos del esquema de /predict {list(LAG_FIELDS)}.")


def load_model_state(config: Config) -> ModelState:
    """Carga modelo, metadatos y catálogo de estaciones; cualquier problema queda en `state.error`."""
    paths = resolve_paths(config)
    state = ModelState(model_dir=paths["model_dir"], metadata_path=paths["metadata"],
                       model_dir_label=str(paths["model_dir_label"]))
    serving = config["serving"]
    try:
        if not state.metadata_path.exists():
            raise FileNotFoundError(f"No existe el archivo de metadatos del modelo: {state.metadata_path.name}")
        if not (state.model_dir / "MLmodel").exists():
            raise FileNotFoundError(f"No existe un modelo MLflow (archivo MLmodel) en {state.model_dir_label}")
        state.metadata = json.loads(state.metadata_path.read_text(encoding="utf-8"))
        state.input_columns = list(state.metadata["input_columns"])
        station_key = config["features"]["categorical"][0]
        check_schema_compatibility(config, state.input_columns, station_key)

        # Integridad: SHA-256 del modelo serializado y del catálogo de estaciones contra los metadatos.
        mlmodel = _read_mlmodel(state.model_dir)
        model_file = state.model_dir / mlmodel["flavors"]["sklearn"]["pickled_model"]
        state.model_sha256 = sha256_file(model_file)
        expected_sha = state.metadata.get("model_sha256")
        state.model_sha256_verified = expected_sha is not None and state.model_sha256 == expected_sha
        if not state.model_sha256_verified:
            msg = f"El SHA-256 del modelo ({state.model_sha256}) no coincide con los metadatos ({expected_sha})."
            if serving.get("require_sha256_match", True):
                raise ValueError(msg)
            logger.warning(msg)
        stations_path = paths["stations"]
        if not stations_path.exists():
            raise FileNotFoundError(f"No existe el catálogo de estaciones {stations_path.name} en "
                                    f"{state.model_dir_label}")
        expected_st = state.metadata.get("stations_sha256")
        state.stations_sha256_verified = expected_st is not None and sha256_file(stations_path) == expected_st
        if not state.stations_sha256_verified:
            logger.warning("El SHA-256 del catálogo de estaciones no coincide con los metadatos.")

        stations = pd.read_csv(stations_path)
        missing = {"id", *STATION_FEATURES} - set(stations.columns)
        if missing:
            raise ValueError(f"Al catálogo de estaciones le faltan columnas: {sorted(missing)}")
        state.stations = stations[["id", *STATION_FEATURES]]
        state.station_ids = {int(i) for i in stations["id"]}

        sig = _signature_types(mlmodel)
        if sig:
            state.int_columns = {c for c in state.input_columns if sig.get(c) in INTEGER_SIGNATURE_TYPES}
        else:  # sin firma: estación, binarias y atributos enteros del catálogo
            int_station = {c for c in STATION_FEATURES if pd.api.types.is_integer_dtype(stations[c])}
            feats = config["features"]
            state.int_columns = set(feats["categorical"]) | set(feats["binary"]) | int_station

        import mlflow.sklearn  # importación diferida: solo se necesita para cargar el modelo

        state.model = mlflow.sklearn.load_model(str(state.model_dir))
        names = getattr(state.model, "feature_names_in_", None)
        if names is not None and list(names) != state.input_columns:
            raise ValueError(f"Las columnas con que se ajustó el modelo {list(names)} no coinciden con "
                             f"input_columns de los metadatos {state.input_columns}.")
        state.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info("Modelo cargado: %s (SHA-256 verificado: %s, %d estaciones)",
                    state.metadata.get("model_uri"), state.model_sha256_verified, len(state.station_ids))
    except Exception as exc:  # el servicio arranca y responde 503 con el motivo
        state.model = None
        state.error = f"{type(exc).__name__}: {exc}"
        logger.error("No se pudo cargar el modelo: %s", state.error)
    return state


# --------------------------------------------------------------------------- variables de entrada
def parse_utc_offset(text: str) -> timezone:
    """Convierte un desfase '-06:00' en un objeto timezone."""
    sign = -1 if text.strip().startswith("-") else 1
    hours, minutes = text.strip().lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))


def to_local_hour(value: datetime, local_tz: timezone) -> datetime:
    """Hora local naive truncada a la hora; las fechas con zona horaria se convierten a `local_tz`."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(local_tz).replace(tzinfo=None)
    return value.replace(minute=0, second=0, microsecond=0)


def build_model_inputs(instances: Sequence[PredictionInstance], state: ModelState, local_tz: timezone,
                       station_key: str) -> tuple[pd.DataFrame, List[datetime]]:
    """Construye las columnas del modelo con las funciones de src.features (mismo cálculo que en entrenamiento)."""
    hours = [to_local_hour(inst.datetime, local_tz) for inst in instances]
    frame = pd.DataFrame({
        "_row": np.arange(len(instances)),
        station_key: np.array([inst.station_id for inst in instances], dtype=np.int64),
        "fecha_hora": pd.to_datetime(hours),
        **{lag: np.array([getattr(inst, lag) for inst in instances], dtype=np.float64) for lag in LAG_FIELDS},
    })
    frame = MiBiciFeatureEngineer.add_calendar_features(frame)
    frame = MiBiciFeatureEngineer.add_station_features(frame, state.stations)
    frame = frame.sort_values("_row").reset_index(drop=True)  # add_station_features reordena las filas
    return cast_inputs(frame[state.input_columns], state.int_columns), hours


def cast_inputs(X: pd.DataFrame, int_columns: Set[str]) -> pd.DataFrame:
    """Tipos de la firma del modelo: enteros (int64) para las columnas `long`, float64 para el resto."""
    return X.astype({c: (np.int64 if c in int_columns else np.float64) for c in X.columns})


def predict_clipped(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predicción con el mismo posprocesamiento del entrenamiento y la exportación (sin negativos)."""
    return np.clip(np.asarray(model.predict(X), dtype=float), 0, None)


# --------------------------------------------------------------------------- aplicación
def _configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def _runtime_versions() -> Dict[str, str]:
    import sklearn

    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform()}
    try:
        import mlflow

        versions["mlflow"] = mlflow.__version__
    except ImportError:  # pragma: no cover - mlflow-skinny siempre está en requirements-serve.txt
        pass
    return versions


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Fábrica de la aplicación; `config` permite usar rutas temporales en las pruebas."""
    config = config or load_config()
    serving = config["serving"]
    service_version = str(serving.get("service_version", "1.0.0"))
    max_batch = int(serving.get("max_batch_size", 1000))
    local_tz = parse_utc_offset(str(serving.get("local_utc_offset", "-06:00")))
    station_key = config["features"]["categorical"][0]
    _configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = load_model_state(config)
        if state.error and serving.get("strict_startup", False):
            raise RuntimeError(f"Arranque cancelado (serving.strict_startup): {state.error}")
        app.state.model_state = state
        yield

    app = FastAPI(
        title="MiBici - API de demanda horaria por estación",
        version=service_version,
        description=(
            "Predice cuántos viajes saldrán de una estación de MiBici (Guadalajara) en una hora dada, con el "
            "modelo champion del MLflow Model Registry (`models:/mibici-demand-forecaster/<versión>`, alias "
            "`champion`), exportado a `models/champion`.\n\n"
            "El cliente envía la estación, la hora (hora local de Guadalajara) y tres rezagos de demanda; el "
            "servicio deriva las variables de calendario (seno/coseno de hora y día, fin de semana, festivo) y "
            "de estación (latitud, longitud, capacidad) con las mismas funciones de `src/features.py` usadas en "
            f"el entrenamiento. Máximo {max_batch} instancias por petición. Equipo 19."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.model_state = ModelState(model_dir=Path("."), metadata_path=Path("."), model_dir_label="",
                                       error="El modelo aún no se ha cargado.")

    # ----------------------------------------------------------------- manejo de errores
    @app.exception_handler(ServingError)
    async def _serving_error(request: Request, exc: ServingError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = _validation_details(exc.errors())
        logger.info("422 %s %s: %d errores de validación", request.method, request.url.path, len(details))
        return JSONResponse(status_code=422, content={
            "error": "validation_error", "message": "La petición no cumple el esquema de entrada.",
            "details": details})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        known = {404: ("not_found", "Recurso no encontrado."), 405: ("method_not_allowed", "Método no permitido.")}
        error, message = known.get(exc.status_code, ("http_error", str(exc.detail)))
        return JSONResponse(status_code=exc.status_code, content={"error": error, "message": message},
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error interno en %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error",
                                                      "message": "Error interno del servidor."})

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
        logger.info("%s %s -> %d (%.2f ms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    # ----------------------------------------------------------------- utilidades de las rutas
    def _ready_state(request: Request) -> ModelState:
        state: ModelState = request.app.state.model_state
        if not state.loaded:
            raise ServingError(503, "model_unavailable", f"El modelo no está cargado: {state.error}")
        return state

    def _check_batch(n: int, field_name: str) -> None:
        if n > max_batch:
            raise ServingError(422, "validation_error", "La petición no cumple el esquema de entrada.", details=[{
                "loc": ["body", field_name], "type": "too_long",
                "msg": f"El lote tiene {n} elementos; el máximo es {max_batch} (serving.max_batch_size)."}])

    # ----------------------------------------------------------------- rutas
    @app.get("/", tags=["Operación"], summary="Resumen del servicio")
    def root(request: Request) -> Dict[str, Any]:
        state: ModelState = request.app.state.model_state
        return {"service": app.title, "service_version": service_version, "model_loaded": state.loaded,
                "model_uri": state.metadata.get("model_uri"), "docs": "/docs", "openapi": "/openapi.json",
                "endpoints": ["GET /health", "GET /model-info", "POST /predict", "POST /predict/raw"]}

    @app.get("/health", tags=["Operación"], summary="Estado del servicio", response_model=HealthResponse,
             responses={503: {"model": HealthResponse, "description": "El modelo no está cargado."}})
    def health(request: Request) -> JSONResponse:
        state: ModelState = request.app.state.model_state
        body = HealthResponse(
            status="ok" if state.loaded else "unavailable", model_loaded=state.loaded,
            model_uri=state.metadata.get("model_uri"), model_sha256_verified=state.model_sha256_verified,
            service_version=service_version, detail=state.error)
        return JSONResponse(status_code=200 if state.loaded else 503, content=body.model_dump())

    @app.get("/model-info", tags=["Operación"], summary="Metadatos del modelo cargado",
             responses={503: ERROR_RESPONSES[503]})
    def model_info(request: Request) -> Dict[str, Any]:
        state = _ready_state(request)
        md = state.metadata
        return {
            "model": state.model_ref(),
            "alias_uri": md.get("alias_uri"),
            "aliases": md.get("aliases"),
            "run_id": md.get("run_id"),
            "estimator_class": md.get("estimator_class"),
            "hyperparameters": md.get("hyperparameters"),
            "model_dir": state.model_dir_label,
            "model_file": md.get("model_file"),
            "model_sha256": md.get("model_sha256"),
            "model_sha256_computed": state.model_sha256,
            "model_sha256_verified": state.model_sha256_verified,
            "stations_sha256_verified": state.stations_sha256_verified,
            "dataset_sha256": md.get("dataset_sha256"),
            "num_stations": len(state.station_ids),
            "metrics": md.get("metrics"),
            "input_columns": state.input_columns,
            "request_fields": ["station_id", "datetime", *LAG_FIELDS],
            "target": md.get("target"),
            "prediction_postprocessing": md.get("prediction_postprocessing"),
            "data_period": md.get("data_period"),
            "training_library_versions": md.get("library_versions"),
            "runtime_library_versions": _runtime_versions(),
            "exported_at": md.get("exported_at"),
            "loaded_at": state.loaded_at,
            "service_version": service_version,
            "local_utc_offset": serving.get("local_utc_offset", "-06:00"),
            "max_batch_size": max_batch,
        }

    @app.post("/predict", tags=["Predicción"], response_model=PredictResponse,
              summary="Predice la demanda a partir de estación, hora y rezagos",
              responses={404: {"model": ErrorResponse, "description": "Estación desconocida para el modelo."},
                         **ERROR_RESPONSES})
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        start = time.perf_counter()
        state = _ready_state(request)
        _check_batch(len(payload.instances), "instances")
        unknown = sorted({inst.station_id for inst in payload.instances} - state.station_ids)
        if unknown:
            raise ServingError(404, "unknown_station",
                               f"Estaciones desconocidas para el modelo: {unknown}. Consulta /model-info.",
                               unknown_station_ids=unknown)
        X, hours = build_model_inputs(payload.instances, state, local_tz, station_key)
        preds = predict_clipped(state.model, X)
        logger.info("predict: %d instancias en %.2f ms", len(preds), (time.perf_counter() - start) * 1000)
        return PredictResponse(
            predictions=[{"station_id": inst.station_id, "datetime": h.isoformat(), "demand_pred": float(p)}
                         for inst, h, p in zip(payload.instances, hours, preds)],
            model=state.model_ref(), n_instances=len(preds))

    @app.post("/predict/raw", tags=["Predicción"], response_model=RawPredictResponse,
              summary="Predice a partir de las columnas del modelo ya construidas",
              description="Recibe filas con exactamente las columnas `input_columns` del modelo (las de "
                          "`models/reference_predictions.csv`) y regresa la predicción sin derivar variables. "
                          "Sirve para verificar la reproducibilidad contra la referencia.",
              responses=ERROR_RESPONSES)
    def predict_raw(payload: RawPredictRequest, request: Request) -> RawPredictResponse:
        start = time.perf_counter()
        state = _ready_state(request)
        _check_batch(len(payload.rows), "rows")
        expected = set(state.input_columns)
        details: List[Dict[str, Any]] = []
        for i, row in enumerate(payload.rows):
            missing, extra = sorted(expected - row.keys()), sorted(row.keys() - expected)
            if missing or extra:
                details.append({"loc": ["body", "rows", i], "type": "columns_mismatch",
                                "msg": f"Columnas faltantes: {missing}; columnas no esperadas: {extra}."})
            for col in state.int_columns & row.keys():
                if not float(row[col]).is_integer():
                    details.append({"loc": ["body", "rows", i, col], "type": "int_expected",
                                    "msg": f"La columna {col} debe ser un valor entero."})
        if details:
            raise ServingError(422, "validation_error", "La petición no cumple el esquema de entrada.",
                               details=details)
        X = cast_inputs(pd.DataFrame(payload.rows)[state.input_columns], state.int_columns)
        preds = predict_clipped(state.model, X)
        logger.info("predict/raw: %d filas en %.2f ms", len(preds), (time.perf_counter() - start) * 1000)
        return RawPredictResponse(predictions=[float(p) for p in preds], model=state.model_ref(),
                                  n_instances=len(preds))

    return app


app = create_app()
