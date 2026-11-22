# Contenedor Docker del servicio y verificación de reproducibilidad

La imagen empaqueta la API FastAPI (`src/serving/`), sus dependencias (`requirements-serve.txt`) y el modelo
champion exportado (`models/champion/` + `models/champion_metadata.json`, `models:/mibici-demand-forecaster/1`),
incluido su feature store (`models/champion/feature_store.csv.gz`). También incluye los dos patrones oro
(`models/reference_predictions.csv` y `models/reference_feature_store_predictions.csv`) para que cualquiera pueda
verificar la imagen. Servicio actual: **1.1.0** (rezagos opcionales en `/predict`, ver [`docs/api.md`](api.md)).

## Construir y correr (comandos de la tarea)

Desde la versión 1.1.0 el modelo está en Git (idea tomada de la rama `entrega-final`): `models/champion/` (~6.6 MB:
`model.pkl`, `MLmodel`, `stations.csv`, `feature_store.csv.gz`, ...) se versiona junto con el código, así que la
imagen se construye en un clon limpio sin ningún paso previo:

```bash
git clone https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici.git
cd Operacion-de-aprendizaje-automatico---Proyecto-Bici
docker build -t ml-service:latest .
docker run -p 8000:8000 ml-service:latest
```

- Documentación interactiva: <http://localhost:8000/docs>; estado: `GET /health` (ver [`docs/api.md`](api.md)).
- En segundo plano: `docker run -d --name mibici-api -p 8000:8000 ml-service:latest` y
  `docker inspect --format '{{.State.Health.Status}}' mibici-api` (llegó a `healthy` en menos de 10 s).
- Con Docker Compose: `docker compose up -d` (imagen publicada) o `docker compose up -d --build` (build local).
- Prueba de humo completa (build + run + healthy + verificación + limpieza) en un solo comando:
  `bash scripts/docker_smoke.sh` o `powershell -ExecutionPolicy Bypass -File scripts\docker_smoke.ps1`.
- Registro de predicciones (opcional, JSONL; la imagen trae `/app/logs` con permisos para el usuario `app`):
  `docker run -p 8000:8000 -e PREDICTION_LOG=/app/logs/predicciones.jsonl -v "$PWD/logs:/app/logs" ml-service:latest`.
  Ejemplo real: `reports/fase_final/logs/21_prediction_log_contenedor.log`.

El paso "Verifica el modelo" del Dockerfile detiene el build si falta un archivo del modelo o si el SHA-256 de
`model.pkl`, `stations.csv` o `feature_store.csv.gz` no coincide con `models/champion_metadata.json`
(`reports/fase_final/reproducibilidad/docker_build_v1.1.0.log`, líneas `Verificado ...`). La carpeta
`models/fase2_original/` (binario de la Fase 2) está en Git pero **no** entra a la imagen (`.dockerignore`).

Prueba de "clon limpio" (22-nov-2026): se copiaron a una carpeta temporal fuera del repositorio solo los 226 archivos
que estarían en Git (`git ls-files` + archivos nuevos no ignorados; sin `data/`, `mlruns.db` ni `.venv`) y ahí
`docker build -t ml-service:clon-limpio .` terminó sin errores (4 s con las capas en caché y 56 s con `--no-cache`,
en la que se ejecutó el paso "Verifica el modelo"); el contenedor llegó a `healthy` y
`scripts/verify_reproducibility.py`, corrido desde esa carpeta, dio PASS
(`reports/fase_final/reproducibilidad/clon_limpio.json`, `clon_limpio.log` y `clon_limpio_build.log`).

## Podman (comandos equivalentes)

Adaptado de la rama `entrega-final`, que construyó y publicó su imagen con Podman. Los comandos
son los mismos cambiando `docker` por `podman`, con dos notas:

```bash
podman build --format docker -t ml-service:latest .
podman run --rm -p 8000:8000 ml-service:latest
podman pull docker.io/gusrivasmna/mibici-demand-service:1.1.0
podman run -d --name mibici-api -p 8000:8000 docker.io/gusrivasmna/mibici-demand-service:1.1.0
podman healthcheck run mibici-api && echo healthy
python scripts/verify_reproducibility.py --url http://localhost:8000 --label podman
podman rm -f mibici-api
```

- `--format docker`: el formato OCI por omisión de Podman descarta la instrucción `HEALTHCHECK`; con el formato
  Docker se conserva (`podman healthcheck run` la ejecuta).
- Windows: si `localhost:8000` no responde con el contenedor en ejecución (Podman sobre WSL en modo *rootless* no
  reenvía bien el puerto), cambiar la máquina a modo *rootful* y volver a construir:
  `podman machine stop; podman machine set --rootful; podman machine start`.
- Con Podman se escribe el nombre completo del registro (`docker.io/...`) para no depender de
  `unqualified-search-registries`.

Estos comandos no se ejecutaron en esta máquina (no tiene Podman); los documentó y usó la rama `entrega-final`.

## Usar la imagen publicada (DockerHub)

```bash
docker pull gusrivasmna/mibici-demand-service:1.1.0
docker run -d --name mibici-api -p 8000:8000 gusrivasmna/mibici-demand-service:1.1.0
python scripts/verify_reproducibility.py --url http://localhost:8000 --label mi-maquina \
    --output reports/fase_final/reproducibilidad/mi_maquina.json
docker rm -f mibici-api
```

## Tags versionados

| Tag | Significado | Digest en DockerHub | ¿Mutable? |
|---|---|---|---|
| `1.0.0` | Primera versión del servicio: `/predict` exige los tres rezagos | `sha256:c28158a1fa7f6f1e6dfe6bf482e00e5ddd60ee882ff4b21942e1bd98bb081fb6` | No: nunca se vuelve a publicar |
| `1.0.0-model-v1` | Servicio 1.0.0 + versión 1 del modelo (`models:/mibici-demand-forecaster/1`) | el mismo de `1.0.0` | No |
| `1.1.0` | Feature store: rezagos opcionales (`lags_source`), `PREDICTION_LOG`; con rezagos responde igual que 1.0.0 | `sha256:00747b9667e5a7f0b160185624920332b4452e16bcc672770c399a5910f3efe2` | No |
| `1.1.0-model-v1` | Servicio 1.1.0 + versión 1 del modelo (mismo `model.pkl` que 1.0.0) | el mismo de `1.1.0` | No |
| `latest` | Última imagen publicada (hoy 1.1.0) | el mismo de `1.1.0` | Sí: se mueve en cada publicación |

Política: un cambio de código o dependencias sube la versión del servicio (SemVer: MAJOR si rompe el esquema de la
API, MINOR si agrega funcionalidad compatible, PATCH si corrige). 1.1.0 es MINOR porque los rezagos pasan de
obligatorios a opcionales: toda petición válida en 1.0.0 sigue siendo válida y da la misma predicción (la respuesta
solo agrega campos). Un modelo nuevo promovido a `champion` con el mismo código publica `<servicio>-model-v<N>`
(p. ej. `1.1.0-model-v2`). Para producción se fija el digest (`imagen@sha256:...`), no el tag. Las etiquetas de la
imagen lo documentan: `docker image inspect gusrivasmna/mibici-demand-service:1.1.0 --format '{{json .Config.Labels}}'`
(`org.opencontainers.image.version`, `mibici.model.version`, `mibici.model.uri`, `org.opencontainers.image.source`).

Publicación verificada con la API pública de DockerHub (`reports/fase_final/reproducibilidad/dockerhub_tags.json`):
`1.1.0`, `1.1.0-model-v1` y `latest` apuntan a `sha256:00747b96…` (127,182,682 bytes comprimida, publicada el
22-nov-2026); `1.0.0` y `1.0.0-model-v1` siguen publicadas con `sha256:c28158a1…` (126,937,001 bytes).

## Detalles de la imagen

| Concepto | Valor |
|---|---|
| Base | `python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86` (misma Python 3.13.9 del entrenamiento, fijada por digest) |
| Construcción | Multi-etapa: `pip install --no-cache-dir -r requirements-serve.txt` en un venv (`/opt/venv`) que se copia a la etapa final |
| Contenido de `/app` | `config.yaml`, `src/{__init__,config,features}.py`, `src/serving/` (incluye `feature_store.py`), `models/champion/` (con `feature_store.csv.gz`), `models/champion_metadata.json`, `models/reference_predictions.csv`, `models/reference_feature_store_predictions.csv`, `logs/` |
| Verificación en el build | SHA-256 de `model.pkl`, `stations.csv` y `feature_store.csv.gz` contra `champion_metadata.json` |
| Usuario | `app` (UID/GID 10001, no root) |
| Variables fijas | `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=42`, `MLFLOW_DISABLE_AGENT_HINT=1` |
| Puerto / comando | `EXPOSE 8000`; `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000` (forma exec) |
| HEALTHCHECK | `GET /health` con `urllib` de Python cada 15 s (503 sin modelo → `unhealthy`) |
| Tamaño | 1.1.0: 556 MB en disco (`docker image ls`) y 127,182,682 bytes (≈127 MB) comprimida en DockerHub; 1.0.0: 555 MB en disco y 126,937,001 bytes comprimida (`reproducibilidad/dockerhub_tags.json`) |
| Tiempo de build | 1.1.0: 30 s con las capas de dependencias en caché (`docker_build_v1.1.0.log`); 1.0.0: 72 s sin caché |
| Dependencias | `pip freeze` de la imagen 1.1.0 (`reproducibilidad/imagen_pip_freeze.txt`): 52 paquetes, idéntico al de 1.0.0 |

Variables de entorno opcionales (`docker run -e ...`): `MODEL_DIR` (carpeta del modelo MLflow con
`stations.csv` y el feature store), `MODEL_METADATA` (archivo de metadatos) y `PREDICTION_LOG` (registro JSONL).
Sin ellas se usan `serving.model_dir` y `serving.metadata_file` de `config.yaml` (`/app/models/champion` y
`/app/models/champion_metadata.json`). Para probar otro modelo sin reconstruir: `docker run -v /ruta/modelo:/modelo:ro
-e MODEL_DIR=/modelo -e MODEL_METADATA=/modelo/metadata.json -p 8000:8000 ml-service:latest`.

## Verificación de reproducibilidad

`scripts/verify_reproducibility.py` usa solo la biblioteca estándar (Python 3.10+): corre en cualquier máquina o
runner sin instalar nada. Espera a `/health`, lee `/model-info`, compara el SHA-256 del modelo y del feature store
servidos con `models/champion_metadata.json` y prueba tres caminos:

1. `/predict/raw` con las 13 columnas del modelo de las 500 filas de `models/reference_predictions.csv`;
2. `/predict` con estación + hora + los tres rezagos de esas mismas 500 filas (`lags_source` debe ser `request`);
3. `/predict` **sin rezagos** con las 200 filas de `models/reference_feature_store_predictions.csv` (horas de la
   ventana del feature store: 179 dentro del periodo de datos y 21 posteriores al último dato, con rezagos y
   predicción calculados fuera de línea en el entorno de entrenamiento). Se comparan la predicción y los tres
   rezagos que devuelve el servicio (deben ser idénticos bit a bit).

Reporta diferencia máxima y media, filas idénticas bit a bit y MAE/RMSE contra el valor real; escribe un JSON
(entorno del cliente, entorno del servidor, veredicto) y sale con código ≠ 0 si falla. Con un servicio sin feature
store (1.0.0) el tercer camino se omite y lo indica.

```bash
python scripts/verify_reproducibility.py --url http://localhost:8000 --label contenedor-local \
    --output reports/fase_final/reproducibilidad/contenedor_local.json [--tolerance 1e-9]
```

### Resultados por entorno (servicio 1.1.0)

El modelo es el mismo binario en todos los casos (SHA-256 `ca4331a49948c99b29580463d89e8aa1dc8d14451f866c029588f3de67ce2ed5`)
y el feature store también (`aa1a39cd1cc8e354728ba835496573ea125925293792bf0a42d203000bf05ee6`).
Referencia con rezagos: MAE 0.752896241060654 y RMSE 1.1785947698217105 sobre las 500 filas.

| Entorno | Servidor | Dif. máx. `/predict/raw` | Dif. máx. `/predict` con rezagos | Idénticas bit a bit | Dif. máx. `/predict` sin rezagos (200 filas) | Rezagos idénticos | Veredicto (tol. 1e-9) | Evidencia |
|---|---|---|---|---|---|---|---|---|
| Windows 11, venv de entrenamiento | Python 3.13.9, uvicorn local | 0.0 | 0.0 | 500/500 | 0.0 (200/200) | 200/200 | PASS | `reproducibilidad/windows_venv.json` |
| Contenedor Linux local (Docker Desktop, WSL2) | `ml-service:latest` (1.1.0) | 8.881784197001252e-16 | 8.881784197001252e-16 | 495/500 | 0.0 (200/200) | 200/200 | PASS | `reproducibilidad/contenedor_local.json` |
| Clon limpio (solo archivos de Git), contenedor Linux | `ml-service:clon-limpio` (1.1.0, `--no-cache`) | 8.881784197001252e-16 | 8.881784197001252e-16 | 495/500 | 0.0 (200/200) | 200/200 | PASS | `reproducibilidad/clon_limpio.json` |
| Imagen descargada de DockerHub (`docker pull`) | `gusrivasmna/mibici-demand-service:1.1.0` | 8.881784197001252e-16 | 8.881784197001252e-16 | 495/500 | 0.0 (200/200) | 200/200 | PASS | `reproducibilidad/imagen_dockerhub.json` |
| GitHub Actions, imagen construida desde el repo (`ubuntu-latest`) | job `docker` de `tests.yml` | 8.881784197001252e-16 | 8.881784197001252e-16 | 495/500 | 0.0 (200/200) | 200/200 | PASS | `reproducibilidad/github_actions_build.json`, [corrida 45066506318](https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici/actions/runs/45066506318) |
| GitHub Actions, imagen publicada (`ubuntu-latest`) | flujo `docker-reproducibilidad` (tag `1.1.0`) | 8.881784197001252e-16 | 8.881784197001252e-16 | 495/500 | 0.0 (200/200) | 200/200 | PASS | `reproducibilidad/github_actions.json`, [corrida 45066506421](https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici/actions/runs/45066506421) |

Las rutas de evidencia son relativas a `reports/fase_final/`. Las 5 filas que difieren en Linux lo hacen en
exactamente 1 ulp: la suma de los árboles (escala log) es idéntica bit a bit en ambos sistemas y la diferencia
está solo en el `exp()` final de la pérdida Poisson, que numpy delega en la biblioteca matemática del sistema
(UCRT en Windows, glibc 2.41 en Linux); son casos casi a la mitad entre dos flotantes y glibc redondea
correctamente 4 de los 5 (`reproducibilidad/diagnostico_exp/resumen.txt`). Con `--tolerance 0` el veredicto en
Linux es FAIL por esas 5 filas (`reproducibilidad/contenedor_local_tolerancia0.json`); por eso la tolerancia por
omisión es `1e-9`, ~6 órdenes de magnitud por encima de lo observado y muy por debajo de cualquier diferencia con
significado (predicciones en viajes por hora). En las 200 filas del feature store no apareció ninguno de esos casos
(0.0 en Linux también).

**Historial 1.0.0.** Los resultados de la versión 1.0.0 (Windows, contenedor local, imagen de DockerHub y runner de
GitHub Actions, todos PASS con las mismas diferencias máximas 0.0 / 8.881784197001252e-16) y sus logs de build se
conservan en `reports/fase_final/reproducibilidad/v1.0.0/`.

**Mismo binario en otro entorno (Fase 2).** `models/fase2_original/` es el `model.joblib` original de la Fase 2
(rama `entrega-final`, entrenado con Python 3.12.10): `python scripts/compare_fase2_binary.py` reproduce sus 1,000
predicciones de referencia con diferencia máxima 0.0 en Windows (Python 3.13.9) y 8.881784197001252e-16 dentro del
contenedor Linux, y lo compara con el champion reentrenado
(`reproducibilidad/fase2_binario_vs_reentrenado.md` y `..._linux.md`).

La reproducibilidad del **pipeline** (reentrenar en un contenedor Linux limpio con `docker/Dockerfile.pipeline` y
comparar métricas con Windows) está en
[`reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md`](../reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md).

### Integración continua

- `.github/workflows/tests.yml`, job `docker` (adaptado de la rama `entrega-final`): después de las pruebas, en
  `ubuntu-latest` construye la imagen **desde el repositorio** con `docker build -t ml-service:latest .`, la levanta,
  espera `healthy`, corre `python3 scripts/verify_reproducibility.py` contra el contenedor (los tres caminos), hace
  una petición real a `/predict` sin rezagos y publica el JSON como anotaciones del job (dividido en partes) y como
  artefacto `reproducibilidad-github-actions-build`.
- `.github/workflows/docker-reproducibilidad.yml` (push a `main`/`fase-final` y ejecución manual con el tag): lint
  del `Dockerfile` con hadolint y verificación de la imagen **publicada** en DockerHub (tag `1.1.0` por omisión);
  publica el JSON como anotaciones del job, dividido en partes (`reproducibilidad-json-01de02`, ...), y como artefacto `reproducibilidad-github-actions`.

## Publicación (la ejecuta quien tenga la cuenta de DockerHub)

```bash
docker login
docker build -t ml-service:latest .
docker tag ml-service:latest gusrivasmna/mibici-demand-service:1.1.0
docker tag ml-service:latest gusrivasmna/mibici-demand-service:1.1.0-model-v1
docker tag ml-service:latest gusrivasmna/mibici-demand-service:latest
docker push gusrivasmna/mibici-demand-service:1.1.0
docker push gusrivasmna/mibici-demand-service:1.1.0-model-v1
docker push gusrivasmna/mibici-demand-service:latest
docker buildx imagetools inspect gusrivasmna/mibici-demand-service:1.1.0   # digest publicado -> tabla de arriba
```
