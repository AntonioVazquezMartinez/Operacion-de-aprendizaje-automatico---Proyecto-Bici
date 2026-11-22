# syntax=docker/dockerfile:1
# =============================================================================
# Servicio de predicción de demanda horaria por estación de MiBici (FastAPI) - Equipo 19
#
# Construir y correr (comandos de la tarea):
#   docker build -t ml-service:latest .
#   docker run -p 8000:8000 ml-service:latest
# Imagen publicada: gusrivasmna/mibici-demand-service:1.0.0 (ver docs/docker.md).
#
# Requisito del build: la carpeta models/champion/ (modelo MLflow exportado + stations.csv) NO está en Git
# (.gitignore). Se obtiene con `python -m src.modeling.export` (desde el Model Registry de mlruns.db) o con
# `dvc pull`. Si falta, el build se detiene con un mensaje explícito (etapa final, paso "Verifica el modelo").
# =============================================================================

# Imagen base fijada por versión exacta (misma Python 3.13.9 del entrenamiento) y por digest del índice
# multiplataforma: un rebuild usa exactamente la misma base aunque la etiqueta se vuelva a publicar.
ARG PYTHON_IMAGE=python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86

# -----------------------------------------------------------------------------
# Etapa 1: instala las dependencias de serving en un venv aislado (/opt/venv)
# -----------------------------------------------------------------------------
# La etiqueta y el digest de la base vienen de ARG PYTHON_IMAGE (hadolint no los ve en la instrucción FROM).
# hadolint ignore=DL3006
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Solo el archivo de dependencias: esta capa se reutiliza mientras requirements-serve.txt no cambie.
COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serve.txt \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "tests" -path "*/site-packages/*/tests" -prune -exec rm -rf {} +

# -----------------------------------------------------------------------------
# Etapa 2: imagen final (sin pip cache ni herramientas de construcción)
# -----------------------------------------------------------------------------
# La etiqueta y el digest de la base vienen de ARG PYTHON_IMAGE (hadolint no los ve en la instrucción FROM).
# hadolint ignore=DL3006
FROM ${PYTHON_IMAGE} AS runtime

ARG SERVICE_VERSION=1.0.0
ARG MODEL_VERSION=1
ARG MODEL_NAME=mibici-demand-forecaster
ARG VCS_REF=unknown

ARG IMAGE_DESCRIPTION="API FastAPI de demanda horaria por estación de MiBici Guadalajara (modelo champion, Equipo 19)"
ARG REPO_URL=https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici

LABEL org.opencontainers.image.title="mibici-demand-service" \
      org.opencontainers.image.description="${IMAGE_DESCRIPTION}" \
      org.opencontainers.image.source="${REPO_URL}" \
      org.opencontainers.image.version="${SERVICE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.13.9-slim" \
      mibici.service.version="${SERVICE_VERSION}" \
      mibici.model.name="${MODEL_NAME}" \
      mibici.model.version="${MODEL_VERSION}" \
      mibici.model.uri="models:/${MODEL_NAME}/${MODEL_VERSION}"

# PYTHONHASHSEED fija el hash de str (mismo valor que `seed` en config.yaml); MLFLOW_DISABLE_AGENT_HINT
# silencia el aviso de MLflow al importarse. MODEL_DIR / MODEL_METADATA quedan sin definir: el servicio usa
# serving.model_dir y serving.metadata_file de config.yaml (se pueden sobrescribir con `docker run -e`).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH="/opt/venv/bin:${PATH}" \
    SERVICE_VERSION=${SERVICE_VERSION} \
    MODEL_VERSION=${MODEL_VERSION}

# Usuario sin privilegios (UID/GID fijos para que los permisos sean predecibles en cualquier host).
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

# Misma estructura que el repositorio: src/config.py resuelve las rutas contra PROJ_ROOT (= /app).
WORKDIR /app

# Modelo y metadatos (cambian con cada versión del modelo, menos seguido que el código).
COPY models/ ./models/

# Verifica el modelo: falla con un mensaje claro si falta models/champion/ o si el SHA-256 de model.pkl no
# coincide con models/champion_metadata.json (solo biblioteca estándar).
RUN python - <<'EOF'
import hashlib, json, pathlib, sys
root = pathlib.Path("/app/models")
needed = [root / "champion" / "MLmodel", root / "champion" / "stations.csv", root / "champion_metadata.json"]
missing = [str(p.relative_to("/app")) for p in needed if not p.exists()]
if missing:
    sys.exit("ERROR: faltan archivos del modelo en el contexto de build: " + ", ".join(missing) + ". "
             "La carpeta models/champion/ no se versiona en Git: genérala con `python -m src.modeling.export` "
             "o descárgala con `dvc pull` antes de `docker build`.")
meta = json.loads((root / "champion_metadata.json").read_text(encoding="utf-8"))
model_file = root / "champion" / meta.get("model_file", "model.pkl")
digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
if digest != meta.get("model_sha256"):
    sys.exit(f"ERROR: SHA-256 de {model_file.name} = {digest}; "
             f"champion_metadata.json espera {meta.get('model_sha256')}.")
print(f"Modelo verificado: {meta['model_uri']} sha256={digest}")
EOF

# Configuración y código (solo los módulos que importa el servicio: src.config, src.features, src.serving).
COPY config.yaml ./config.yaml
COPY src/__init__.py src/config.py src/features.py ./src/
COPY src/serving/ ./src/serving/

USER app

EXPOSE 8000

# /health responde 200 solo con el modelo cargado (503 si no); urlopen lanza excepción con 503 -> unhealthy.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
