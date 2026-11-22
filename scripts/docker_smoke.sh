#!/usr/bin/env bash
# Prueba de humo del contenedor en un solo comando: build + run + espera healthy + verificación + limpieza.
#
# Uso (desde la raíz del repositorio):
#   bash scripts/docker_smoke.sh                                   # construye ml-service:latest y lo verifica
#   SKIP_BUILD=1 IMAGE=gusrivasmna/mibici-demand-service:1.0.0 bash scripts/docker_smoke.sh   # imagen publicada
# Variables: IMAGE (ml-service:latest), PORT (8000), LABEL (contenedor-local), OUTPUT (JSON de resultados),
#            SKIP_BUILD (1 = no construir), PYTHON (python3 o python).
set -euo pipefail

IMAGE=${IMAGE:-ml-service:latest}
PORT=${PORT:-8000}
LABEL=${LABEL:-contenedor-local}
OUTPUT=${OUTPUT:-reports/fase_final/reproducibilidad/docker_smoke_${LABEL}.json}
NAME=mibici-smoke-$$
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "== docker build -t $IMAGE ."
    docker build -t "$IMAGE" .
fi

echo "== docker run -d -p $PORT:8000 $IMAGE"
docker run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null

echo "== esperando HEALTHCHECK"
status=starting
for _ in $(seq 1 40); do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$NAME")
    [ "$status" = "healthy" ] && break
    [ "$status" = "unhealthy" ] && break
    sleep 3
done
echo "estado: $status"
if [ "$status" != "healthy" ]; then
    docker logs "$NAME"
    exit 1
fi

echo "== verificación de reproducibilidad"
"$PYTHON" scripts/verify_reproducibility.py --url "http://localhost:$PORT" --label "$LABEL" --output "$OUTPUT"
