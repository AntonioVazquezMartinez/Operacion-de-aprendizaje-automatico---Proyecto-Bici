# Contenedor Docker del servicio y verificación de reproducibilidad

La imagen empaqueta la API FastAPI (`src/serving/`), sus dependencias (`requirements-serve.txt`) y el modelo
champion exportado (`models/champion/` + `models/champion_metadata.json`, `models:/mibici-demand-forecaster/1`).
También incluye `models/reference_predictions.csv` para que cualquiera pueda verificar la imagen.

## Construir y correr (comandos de la tarea)

Requisito: `models/champion/` no está en Git (se versiona con DVC). Antes del build se genera con
`python -m src.modeling.export` (desde `mlruns.db`) o se descarga con `dvc pull`. Si falta, el build se detiene
con el mensaje `ERROR: faltan archivos del modelo en el contexto de build: models/champion/MLmodel, ...`
(evidencia: `reports/fase_final/reproducibilidad/docker_build_sin_modelo.log`).

```bash
docker build -t ml-service:latest .
docker run -p 8000:8000 ml-service:latest
```

- Documentación interactiva: <http://localhost:8000/docs>; estado: `GET /health` (ver [`docs/api.md`](api.md)).
- En segundo plano: `docker run -d --name mibici-api -p 8000:8000 ml-service:latest` y
  `docker inspect --format '{{.State.Health.Status}}' mibici-api` (llega a `healthy` en ~8 s).
- Con Docker Compose: `docker compose up -d` (imagen publicada) o `docker compose up -d --build` (build local).
- Prueba de humo completa (build + run + healthy + verificación + limpieza) en un solo comando:
  `bash scripts/docker_smoke.sh` o `powershell -ExecutionPolicy Bypass -File scripts\docker_smoke.ps1`.

## Usar la imagen publicada (DockerHub)

```bash
docker pull gusrivasmna/mibici-demand-service:1.0.0
docker run -d --name mibici-api -p 8000:8000 gusrivasmna/mibici-demand-service:1.0.0
python scripts/verify_reproducibility.py --url http://localhost:8000 --label mi-maquina \
    --output reports/fase_final/reproducibilidad/mi_maquina.json
docker rm -f mibici-api
```

## Tags versionados

| Tag | Significado | ¿Mutable? |
|---|---|---|
| `1.0.0` | Versión del servicio (`serving.service_version` en `config.yaml`, SemVer) con el modelo con que se publicó | No: nunca se vuelve a publicar |
| `1.0.0-model-v1` | Servicio 1.0.0 + versión 1 del modelo en el MLflow Model Registry (`models:/mibici-demand-forecaster/1`) | No |
| `latest` | Última imagen publicada | Sí: se mueve en cada publicación |

Política: un cambio de código o dependencias sube la versión del servicio (MAJOR si rompe el esquema de la API,
MINOR si agrega funcionalidad compatible, PATCH si corrige). Un modelo nuevo promovido a `champion` con el mismo
código publica `<servicio>-model-v<N>` (p. ej. `1.0.0-model-v2`) y, si se quiere un tag de servicio nuevo, sube
PATCH. Para producción se fija el digest (`imagen@sha256:...`), no el tag. Las etiquetas de la imagen lo
documentan: `docker image inspect gusrivasmna/mibici-demand-service:1.0.0 --format '{{json .Config.Labels}}'`
(`org.opencontainers.image.version`, `mibici.model.version`, `mibici.model.uri`, `org.opencontainers.image.source`).

## Detalles de la imagen

| Concepto | Valor |
|---|---|
| Base | `python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86` (misma Python 3.13.9 del entrenamiento, fijada por digest) |
| Construcción | Multi-etapa: `pip install --no-cache-dir -r requirements-serve.txt` en un venv (`/opt/venv`) que se copia a la etapa final |
| Contenido de `/app` | `config.yaml`, `src/{__init__,config,features}.py`, `src/serving/`, `models/champion/`, `models/champion_metadata.json`, `models/reference_predictions.csv` |
| Verificación en el build | El SHA-256 de `models/champion/model.pkl` debe coincidir con `champion_metadata.json` |
| Usuario | `app` (UID/GID 10001, no root) |
| Variables fijas | `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=42`, `MLFLOW_DISABLE_AGENT_HINT=1` |
| Puerto / comando | `EXPOSE 8000`; `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000` (forma exec) |
| HEALTHCHECK | `GET /health` con `urllib` de Python cada 15 s (503 sin modelo → `unhealthy`) |
| Tamaño | 558 MB en disco (127 MB comprimido); 293 MB son el venv; contexto de build 6.3 MB |
| Tiempo de build | 72 s sin caché de capas (base ya descargada); 5 s con caché |
| ID local | `sha256:d9244f3d66ec25c5d1676744ae2bd66d8cabc0bece881acde9e16eba1a1f3bd6` |
| Digest publicado en DockerHub | PENDIENTE (se llena después del `docker push`) |

Variables de entorno opcionales (`docker run -e ...`): `MODEL_DIR` (carpeta del modelo MLflow con
`stations.csv`) y `MODEL_METADATA` (archivo de metadatos). Sin ellas se usan `serving.model_dir` y
`serving.metadata_file` de `config.yaml` (`/app/models/champion` y `/app/models/champion_metadata.json`). Para
probar otro modelo sin reconstruir: `docker run -v /ruta/modelo:/modelo:ro -e MODEL_DIR=/modelo -e
MODEL_METADATA=/modelo/metadata.json -p 8000:8000 ml-service:latest`.

Las dependencias de `requirements-serve.txt` están fijadas con `==`; la única dependencia transitiva que quedó
distinta del entorno de entrenamiento es `python-dateutil` (4.0.0 en la imagen vs 2.9.0.post0 en
`requirements-lock.txt`; ver `reports/fase_final/reproducibilidad/imagen_pip_freeze.txt`). No cambia ninguna
predicción.

## Verificación de reproducibilidad

`scripts/verify_reproducibility.py` usa solo la biblioteca estándar (Python 3.10+): corre en cualquier máquina o
runner sin instalar nada. Espera a `/health`, lee `/model-info`, compara el SHA-256 del modelo servido con
`models/champion_metadata.json`, envía las 500 filas de `models/reference_predictions.csv` por `/predict/raw` (13
columnas del modelo) y por `/predict` (estación + hora + rezagos), y compara con `demand_pred`: diferencia máxima y
media, filas idénticas bit a bit y MAE/RMSE contra el valor real. Escribe un JSON (entorno del cliente, entorno del
servidor, veredicto) y sale con código ≠ 0 si falla.

```bash
python scripts/verify_reproducibility.py --url http://localhost:8000 --label contenedor-local \
    --output reports/fase_final/reproducibilidad/contenedor_local.json [--tolerance 1e-9]
```

### Resultados por entorno

El modelo es el mismo binario en todos los casos (SHA-256 `ca4331a49948c99b29580463d89e8aa1dc8d14451f866c029588f3de67ce2ed5`).
Referencia: MAE 0.752896241060654 y RMSE 1.1785947698217105 sobre las 500 filas.

| Entorno | Servidor | Dif. máx. `/predict/raw` | Dif. máx. `/predict` | Idénticas bit a bit | MAE / RMSE de la muestra | Veredicto (tol. 1e-9) | Evidencia |
|---|---|---|---|---|---|---|---|
| Windows 11, venv de entrenamiento | Python 3.13.9, uvicorn local | 0.0 | 0.0 | 500/500 | iguales a la referencia | PASS | `reproducibilidad/windows_venv.json` |
| Contenedor Linux local (Docker Desktop, WSL2) | `ml-service:latest` | 8.881784197001252e-16 | 8.881784197001252e-16 | 495/500 | iguales a la referencia (dif. 0.0) | PASS | `reproducibilidad/contenedor_local.json` |
| GitHub Actions (`ubuntu-latest`) | `gusrivasmna/mibici-demand-service:1.0.0` | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | artefacto `reproducibilidad-github-actions` del flujo `docker-reproducibilidad` |

Las rutas de evidencia son relativas a `reports/fase_final/`. Las 5 filas que difieren en Linux lo hacen en
exactamente 1 ulp: la suma de los árboles (escala log) es idéntica bit a bit en ambos sistemas y la diferencia
está solo en el `exp()` final de la pérdida Poisson, que numpy delega en la biblioteca matemática del sistema
(UCRT en Windows, glibc 2.41 en Linux); son casos casi a la mitad entre dos flotantes y glibc redondea
correctamente 4 de los 5 (`reproducibilidad/diagnostico_exp/resumen.txt`). Con `--tolerance 0` el veredicto en
Linux es FAIL (`reproducibilidad/contenedor_local_tolerancia0.json`); por eso la tolerancia por omisión es `1e-9`,
~6 órdenes de magnitud por encima de lo observado y muy por debajo de cualquier diferencia con significado
(predicciones en viajes por hora).

La reproducibilidad del **pipeline** (reentrenar en un contenedor Linux limpio con `docker/Dockerfile.pipeline` y
comparar métricas con Windows) está en
[`reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md`](../reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md).

### Integración continua

`.github/workflows/docker-reproducibilidad.yml` (push a `main`/`fase-final` y ejecución manual con el tag a
verificar): hace lint del `Dockerfile` con hadolint y, en `ubuntu-latest`, descarga la imagen publicada (no se puede
construir en CI porque `models/champion/` no está en Git), espera `healthy`, corre la verificación y sube el JSON
como artefacto `reproducibilidad-github-actions`.

## Publicación (la ejecuta quien tenga la cuenta de DockerHub)

```bash
docker login
docker build -t ml-service:latest .
docker tag ml-service:latest gusrivasmna/mibici-demand-service:1.0.0
docker tag ml-service:latest gusrivasmna/mibici-demand-service:1.0.0-model-v1
docker tag ml-service:latest gusrivasmna/mibici-demand-service:latest
docker push gusrivasmna/mibici-demand-service:1.0.0
docker push gusrivasmna/mibici-demand-service:1.0.0-model-v1
docker push gusrivasmna/mibici-demand-service:latest
docker buildx imagetools inspect gusrivasmna/mibici-demand-service:1.0.0   # digest publicado -> tabla de arriba
```
