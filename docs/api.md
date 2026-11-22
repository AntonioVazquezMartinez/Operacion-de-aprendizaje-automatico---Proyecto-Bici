# API de predicción de demanda (FastAPI)

Servicio REST que predice cuántos viajes saldrán de una estación de MiBici (Guadalajara) en una hora
dada. Código: `src/serving/app.py` (rutas, carga del modelo, errores) y `src/serving/schemas.py`
(esquemas Pydantic v2). Versión del servicio: `1.0.0` (`serving.service_version` en `config.yaml`).

## Artefacto del modelo

| Concepto | Valor |
|---|---|
| Modelo registrado (MLflow Model Registry) | `mibici-demand-forecaster` |
| Versión servida | `models:/mibici-demand-forecaster/1` |
| Alias | `champion` (`models:/mibici-demand-forecaster@champion`) |
| Carpeta local que carga la API | `models/champion` (modelo MLflow autocontenido + `stations.csv`) |
| Metadatos | `models/champion_metadata.json` (run_id, SHA-256, métricas, columnas, versiones) |
| SHA-256 de `models/champion/model.pkl` | `ca4331a49948c99b29580463d89e8aa1dc8d14451f866c029588f3de67ce2ed5` |
| Algoritmo | `hist_gradient_boosting` (`sklearn.ensemble.HistGradientBoostingRegressor`, pérdida Poisson) |

La carpeta se genera con `python -m src.modeling.export` y se carga con `mlflow.sklearn.load_model`
sin la base `mlruns.db`. Al arrancar, el servicio recalcula el SHA-256 de `model.pkl` y lo compara con
los metadatos; si no coincide (con `serving.require_sha256_match: true`) no carga el modelo y responde 503.

## Cómo levantar el servicio

```bash
pip install -r requirements-serve.txt        # o requirements.txt para el entorno completo
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

- Documentación interactiva: <http://localhost:8000/docs> (Swagger UI) y <http://localhost:8000/redoc>.
- Esquema OpenAPI: `GET /openapi.json`, exportado en [`docs/openapi.json`](openapi.json) con
  `python scripts/export_openapi.py`.
- Colección de Postman (v2.1): [`docs/postman_collection.json`](postman_collection.json)
  (variable `baseUrl`, por defecto `http://localhost:8000`).
- Rutas sobrescribibles con variables de entorno: `MODEL_DIR` (carpeta del modelo, con `stations.csv`
  dentro) y `MODEL_METADATA` (archivo de metadatos). Si no se definen, se usan `serving.model_dir` y
  `serving.metadata_file` de `config.yaml` (relativas a la raíz del repositorio); una ruta relativa en la
  variable de entorno se resuelve contra la carpeta de trabajo.
- Prueba de humo contra un servicio levantado: `python scripts/api_smoke.py --url http://127.0.0.1:8000`.

## Endpoints

| Método | Ruta | Descripción | Códigos |
|---|---|---|---|
| GET | `/` | Resumen del servicio y lista de endpoints | 200 |
| GET | `/health` | Estado: modelo cargado, URI, SHA-256 verificado | 200, 503 |
| GET | `/model-info` | Metadatos del modelo: nombre, versión, URI, alias, run_id, hiperparámetros, SHA-256, métricas, columnas de entrada, versiones de librerías (entrenamiento y ejecución) | 200, 503 |
| POST | `/predict` | Predicción a partir de estación, hora y rezagos (el servidor deriva el resto) | 200, 404, 422, 503, 500 |
| POST | `/predict/raw` | Predicción a partir de las 13 columnas del modelo ya construidas (verificación de reproducibilidad) | 200, 422, 503, 500 |

## Esquema de entrada de `POST /predict`

```json
{"instances": [{"station_id": 24, "datetime": "2025-11-08T10:00:00", "demand_lag_24h": 5.0,
                "demand_lag_168h": 0.0, "demand_same_hour_mean_7d": 3.5714285714285716}]}
```

| Campo | Tipo | Reglas | Significado |
|---|---|---|---|
| `instances` | lista | 1 a `serving.max_batch_size` (1000) elementos | Instancias a predecir; la respuesta conserva el orden |
| `station_id` | entero | ≥ 1, debe ser una de las 370 estaciones del modelo | `Origen_Id` de la estación |
| `datetime` | texto ISO-8601 | fecha válida; no se aceptan números (época Unix) | Hora a predecir en hora local de Guadalajara |
| `demand_lag_24h` | número | finito, ≥ 0 | Viajes en la misma hora del día anterior (t − 24 h) |
| `demand_lag_168h` | número | finito, ≥ 0 | Viajes en la misma hora de la semana anterior (t − 168 h) |
| `demand_same_hour_mean_7d` | número | finito, ≥ 0 | Promedio a la misma hora de los 7 días previos |

No se admiten campos adicionales (`extra_forbidden`) ni valores `true/false` o textos en los campos numéricos.

**Variables que deriva el servidor** con las mismas funciones del entrenamiento (`src/features.py`):
`MiBiciFeatureEngineer.add_calendar_features` (seno/coseno de la hora y del día de la semana,
`is_weekend`, `is_holiday` con `mexican_holidays`) y `MiBiciFeatureEngineer.add_station_features`
(`latitude`, `longitude`, `capacity` desde `models/champion/stations.csv`). Así no hay diferencia entre
entrenamiento y servicio: las 500 filas de `models/reference_predictions.csv` enviadas por `/predict`
reproducen `demand_pred` exactamente (diferencia máxima 0.0, ver más abajo).

**Zona horaria y truncado de `datetime`:**

- Sin zona horaria (`2025-11-08T10:00:00`): se interpreta como hora local de Guadalajara.
- Con zona horaria (`2025-11-08T16:30:00Z`, `...-06:00`): se convierte a la hora local con el desfase
  fijo `serving.local_utc_offset` (`-06:00`; Jalisco no tiene horario de verano desde 2022). Ejemplo
  real: `16:30Z` → `10:00` local.
- Se trunca a la hora: `10:45` predice el bloque 10:00–10:59 (igual que `floor("h")` del entrenamiento).
  La respuesta devuelve la hora local ya truncada, sin zona horaria.

## Esquema de salida de `POST /predict`

Respuesta real (fila 1 de la referencia; `demand_pred` en la referencia: `2.6064224732591033`):

```json
{"predictions":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_pred":2.6064224732591033}],
 "model":{"name":"mibici-demand-forecaster","version":1,"uri":"models:/mibici-demand-forecaster/1",
          "alias":"champion","algorithm":"hist_gradient_boosting"},"n_instances":1}
```

`demand_pred` es la predicción del modelo recortada a ≥ 0 (`clip(prediccion, 0, None)`, igual que en
`src/modeling/predict.py`).

## `POST /predict/raw`

Recibe `{"rows": [...]}` con exactamente las 13 columnas `input_columns` de `/model-info`, que son las
columnas de `models/reference_predictions.csv` excepto `fecha_hora`, `demand_count` y `demand_pred`.
`capacity`, `Origen_Id`, `is_weekend` e `is_holiday` deben ser enteros.
Responde `{"predictions": [<float>, ...], "model": {...}, "n_instances": N}`.

## Ejemplos

### curl (salidas reales)

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_uri":"models:/mibici-demand-forecaster/1","model_sha256_verified":true,"service_version":"1.0.0","detail":null}

curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
# {"predictions":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_pred":2.6064224732591033}],"model":{"name":"mibici-demand-forecaster","version":1,"uri":"models:/mibici-demand-forecaster/1","alias":"champion","algorithm":"hist_gradient_boosting"},"n_instances":1}

curl -s -w " HTTP %{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":99999,"datetime":"2025-11-08T10:00:00","demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
# {"error":"unknown_station","message":"Estaciones desconocidas para el modelo: [99999]. Consulta /model-info.","unknown_station_ids":[99999]} HTTP 404

curl -s -w " HTTP %{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_lag_24h":-1,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
# {"error":"validation_error","message":"La petición no cumple el esquema de entrada.","details":[{"loc":["body","instances",0,"demand_lag_24h"],"msg":"Debe ser mayor o igual que 0.0.","type":"greater_than_equal"}]} HTTP 422
```

### PowerShell (`Invoke-RestMethod`, salidas reales)

```powershell
$body = @{ instances = @(@{ station_id = 80; datetime = "2025-11-08T11:00:00"; demand_lag_24h = 1.0;
           demand_lag_168h = 2.0; demand_same_hour_mean_7d = 1.4285714285714286 }) } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body
$r.predictions | Format-Table
# station_id datetime                   demand_pred
# ---------- --------                   -----------
#         80 2025-11-08T11:00:00 2.2542647065879575
$r.model.uri
# models:/mibici-demand-forecaster/1
(Invoke-RestMethod http://localhost:8000/health).status
# ok

# Errores: Invoke-RestMethod lanza una excepción; el cuerpo JSON está en ErrorDetails.
try { Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"instances":[]}' }
catch { $_.Exception.Response.StatusCode.value__; $_.ErrorDetails.Message }
# 422
# {"error":"validation_error","message":"La petición no cumple el esquema de entrada.","details":[{"loc":["body","instances"],"msg":"La lista debe tener al menos 1 elemento(s).","type":"too_short"}]}
```

## Errores

Todos los errores tienen el mismo cuerpo JSON (`ErrorResponse`): `error` (código corto), `message`
(en español) y, según el caso, `details` (lista de `{loc, msg, type}`) o `unknown_station_ids`.

| HTTP | `error` | Cuándo |
|---|---|---|
| 422 | `validation_error` | Tipos incorrectos, negativos, NaN/Infinity, fecha inválida o numérica, campo faltante o extra, lote vacío (`too_short`) o mayor que `serving.max_batch_size` (`too_long`), JSON mal formado (`json_invalid`), columnas incorrectas en `/predict/raw` (`columns_mismatch`, `int_expected`) |
| 404 | `unknown_station` | Alguna `station_id` no está en el catálogo del modelo; `unknown_station_ids` lista todas |
| 404 | `not_found` | Ruta inexistente |
| 405 | `method_not_allowed` | Método HTTP no permitido en la ruta |
| 503 | `model_unavailable` | El modelo no está cargado (carpeta inexistente, SHA-256 distinto, esquema incompatible); `/health` también responde 503 con `detail` |
| 500 | `internal_error` | Error no previsto; el mensaje es genérico y la traza solo va al log del servidor |

Ejemplo real de 503 (servicio levantado con `MODEL_DIR=models/no_existe`): ver la sección C de
`reports/fase_final/logs/20_api_smoke.log`.

## Configuración (`serving:` en `config.yaml`)

| Clave | Valor | Uso |
|---|---|---|
| `model_dir` | `models/champion` | Carpeta del modelo (sobrescribible con `MODEL_DIR`) |
| `metadata_file` | `models/champion_metadata.json` | Metadatos (sobrescribible con `MODEL_METADATA`) |
| `stations_file` | `stations.csv` | Catálogo de estaciones dentro de `model_dir` |
| `service_version` | `1.0.0` | Versión del servicio en OpenAPI y `/health` |
| `max_batch_size` | `1000` | Máximo de instancias por petición |
| `local_utc_offset` | `-06:00` | Desfase usado para convertir fechas con zona horaria |
| `require_sha256_match` | `true` | No cargar un modelo cuyo SHA-256 no coincide |
| `strict_startup` | `false` | `true`: un error de carga detiene el arranque en lugar de responder 503 |

Si los rezagos de `features` en `config.yaml` o las `input_columns` de los metadatos dejan de coincidir
con los campos del esquema de `/predict` (`LAG_FIELDS` en `src/serving/schemas.py`), el servicio no
carga el modelo y lo explica en `/health` (`detail`).

## Verificación realizada

- `reports/fase_final/logs/20_api_smoke.log`: peticiones y respuestas reales contra uvicorn
  (A: venv del proyecto; B: venv limpio con solo `requirements-serve.txt`; C: sin modelo → 503).
  En A y B las 500 filas de la referencia dieron diferencia máxima absoluta `0.0` por `/predict` y por
  `/predict/raw` (500/500 iguales bit a bit).
- `reports/fase_final/logs/20_api_server.log`: log del servidor (carga del modelo, una línea por
  petición con su latencia) y `pip freeze` del venv limpio.
- Pruebas: `pytest tests/test_serving.py`.
