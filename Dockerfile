# =============================================================================
# Imagen del servicio de predicción MiBici (Equipo 19)
# Construir:  podman build --format docker -t mibici-demand-api:1.0.0 .   (o docker build -t ...)
#             --format docker conserva el HEALTHCHECK (el formato OCI de Podman lo ignora)
# Ejecutar:   podman run --rm -p 8000:8000 mibici-demand-api:1.0.0
# Documentación interactiva: http://localhost:8000/docs
# =============================================================================
# Misma versión menor de Python con la que se entrenó el modelo (3.12).
FROM python:3.12-slim

LABEL org.opencontainers.image.title="mibici-demand-api" \
      org.opencontainers.image.description="API FastAPI que sirve el modelo champion de demanda MiBici (models:/mibici-demand-forecaster/1)" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici" \
      org.opencontainers.image.authors="Equipo 19 TC5061"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MODEL_DIR=/app/models/champion

WORKDIR /app

# 1. Dependencias primero (capa en caché mientras requirements-api.txt no cambie)
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2. Código y paquete del modelo (modelo + feature store + catálogo + predicciones de referencia)
COPY config.yaml ./
COPY src/ src/
COPY models/champion/ models/champion/

# 3. Usuario sin privilegios
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
