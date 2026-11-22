# API de predicción de demanda (FastAPI)

Servicio REST que predice cuántos viajes saldrán de una estación de MiBici (Guadalajara) en una hora
dada. Código: `src/serving/app.py` (rutas, carga del modelo, errores) y `src/serving/schemas.py`
(esquemas Pydantic v2), con el feature store en `src/serving/feature_store.py`. Versión del servicio: `1.1.0`
(`serving.service_version` en `config.yaml`).

**Cambios de la versión 1.1.0** (fusión con la rama `entrega-final` de Jesús Rincón; las ideas del feature store,
de `lags_source` y del registro de predicciones vienen de esa rama):

- Los tres rezagos de `POST /predict` son **opcionales**: se envían los tres o ninguno. Sin rezagos, el servicio
  los calcula con el *feature store* del modelo (`models/champion/feature_store.csv.gz`: demanda de las últimas
  360 h por estación) usando `MiBiciFeatureEngineer.add_lag_features`, la misma función del entrenamiento.
- Cada predicción indica el origen de los rezagos (`lags_source`: `request` o `feature_store`) y los valores usados.
- Compatibilidad: una petición con los tres rezagos da exactamente la misma predicción que en la versión 1.0.0
  (las 500 filas de `models/reference_predictions.csv` dan diferencia 0.0); `/predict/raw` no cambia.
- Registro opcional de predicciones en JSONL con la variable de entorno `PREDICTION_LOG`.

## Artefacto del modelo

| Concepto | Valor |
|---|---|
| Modelo registrado (MLflow Model Registry) | `mibici-demand-forecaster` |
| Versión servida | `models:/mibici-demand-forecaster/1` |
| Alias | `champion` (`models:/mibici-demand-forecaster@champion`) |
| Carpeta local que carga la API | `models/champion` (modelo MLflow autocontenido + `stations.csv` + `feature_store.csv.gz`), versionada en Git |
| Metadatos | `models/champion_metadata.json` (run_id, SHA-256, métricas, columnas, versiones) |
| SHA-256 de `models/champion/model.pkl` | `ca4331a49948c99b29580463d89e8aa1dc8d14451f866c029588f3de67ce2ed5` |
| Algoritmo | `hist_gradient_boosting` (`sklearn.ensemble.HistGradientBoostingRegressor`, pérdida Poisson) |

La carpeta se genera con `python -m src.modeling.export` y se carga con `mlflow.sklearn.load_model`
sin la base `mlruns.db`. Al arrancar, el servicio recalcula el SHA-256 de `model.pkl` y del feature store y los
compara con los metadatos; si no coinciden (con `serving.require_sha256_match: true`) no carga el modelo y
responde 503.

| Feature store | Valor (de `/model-info`, sección `feature_store`) |
|---|---|
| Archivo | `models/champion/feature_store.csv.gz` (133,200 filas = 370 estaciones x 360 h; SHA-256 `aa1a39cd1cc8e354728ba835496573ea125925293792bf0a42d203000bf05ee6`) |
| Periodo | `2025-12-17T00:00:00` a `2025-12-31T23:00:00` (últimas `serving.feature_store_hours` = 360 h de los datos) |
| Ventana predecible sin rezagos | `2025-12-24T00:00:00` a `2026-01-01T23:00:00` (desde la hora 168 del feature store hasta 24 h después del último dato) |
| Patrón oro | `models/reference_feature_store_predictions.csv`: 200 pares (estación, hora) de la ventana con rezagos y predicción calculados fuera de línea (179 dentro del periodo de datos, 21 posteriores) |

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
- Registro de predicciones (opcional): `PREDICTION_LOG=logs/predicciones.jsonl` (o `serving.prediction_log_file`)
  agrega una línea JSON por petición a `/predict` que llega a la ruta: marca de tiempo UTC, versión del servicio,
  URI y versión del modelo, código HTTP, número de instancias, latencia en ms, entradas y salidas (o el error 404/422
  del servicio). Un fallo al escribir solo deja un aviso en el log del servidor; la petición responde igual.
  Ejemplo real en el contenedor: `reports/fase_final/logs/21_prediction_log_contenedor.log`.

## Endpoints

| Método | Ruta | Descripción | Códigos |
|---|---|---|---|
| GET | `/` | Resumen del servicio y lista de endpoints | 200 |
| GET | `/health` | Estado: modelo cargado, URI, SHA-256 verificado | 200, 503 |
| GET | `/model-info` | Metadatos del modelo: nombre, versión, URI, alias, run_id, hiperparámetros, SHA-256, métricas, columnas de entrada, versiones de librerías (entrenamiento y ejecución), feature store (periodo y ventana predecible) | 200, 503 |
| POST | `/predict` | Predicción a partir de estación y hora, con rezagos opcionales (el servidor deriva el resto) | 200, 404, 422, 503, 500 |
| POST | `/predict/raw` | Predicción a partir de las 13 columnas del modelo ya construidas (verificación de reproducibilidad) | 200, 422, 503, 500 |

## Esquema de entrada de `POST /predict`

```json
{"instances": [{"station_id": 24, "datetime": "2025-11-08T10:00:00", "demand_lag_24h": 5.0,
                "demand_lag_168h": 0.0, "demand_same_hour_mean_7d": 3.5714285714285716},
               {"station_id": 24, "datetime": "2025-12-31T18:00:00"}]}
```

| Campo | Tipo | Reglas | Significado |
|---|---|---|---|
| `instances` | lista | 1 a `serving.max_batch_size` (1000) elementos | Instancias a predecir; la respuesta conserva el orden. Se pueden mezclar instancias con y sin rezagos |
| `station_id` | entero | obligatorio, ≥ 1, debe ser una de las 370 estaciones del modelo | `Origen_Id` de la estación |
| `datetime` | texto ISO-8601 | obligatorio, fecha válida; no se aceptan números (época Unix) | Hora a predecir en hora local de Guadalajara |
| `demand_lag_24h` | número | opcional; finito, ≥ 0 | Viajes en la misma hora del día anterior (t − 24 h) |
| `demand_lag_168h` | número | opcional; finito, ≥ 0 | Viajes en la misma hora de la semana anterior (t − 168 h) |
| `demand_same_hour_mean_7d` | número | opcional; finito, ≥ 0 | Promedio a la misma hora de los 7 días previos |

Los tres rezagos van juntos: o se envían los tres o ninguno (con uno o dos responde 422 `lags_incomplete`). Sin
rezagos, la hora debe estar dentro de la ventana predecible del feature store (`2025-12-24T00:00:00` a
`2026-01-01T23:00:00`); fuera de ella responde 422 `outside_feature_store_window` con la ventana válida. No se
admiten campos adicionales (`extra_forbidden`) ni valores `true/false` o textos en los campos numéricos.

**Rezagos desde el feature store** (sin *training-serving skew*): el servicio arma la cuadrícula estación x hora del
feature store, agrega las horas posteriores al último dato con demanda desconocida y aplica
`MiBiciFeatureEngineer.add_lag_features` (la función que construyó la matriz de entrenamiento). Las 200 filas de
`models/reference_feature_store_predictions.csv` (rezagos y predicción calculados fuera de línea con el pipeline)
se reproducen con diferencia 0.0 en rezagos y predicciones (`tests/test_serving_feature_store.py` y
`scripts/verify_reproducibility.py`).

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

Respuesta real a la petición de arriba (contenedor `ml-service:latest` 1.1.0, sección B de
`reports/fase_final/logs/20_api_smoke.log`; figura `reports/figures/14_api_respuesta_200.png`). La primera
predicción es la misma de la versión 1.0.0 (`2.6064224732591033`, fila 1 de la referencia):

```json
{"predictions":[
  {"station_id":24,"datetime":"2025-11-08T10:00:00","demand_pred":2.6064224732591033,"lags_source":"request",
   "demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716},
  {"station_id":24,"datetime":"2025-12-31T18:00:00","demand_pred":3.1631559452523517,"lags_source":"feature_store",
   "demand_lag_24h":0.0,"demand_lag_168h":1.0,"demand_same_hour_mean_7d":1.8571428571428572}],
 "model":{"name":"mibici-demand-forecaster","version":1,"uri":"models:/mibici-demand-forecaster/1",
          "alias":"champion","algorithm":"hist_gradient_boosting"},"n_instances":2}
```

| Campo de cada predicción | Significado |
|---|---|
| `station_id`, `datetime` | Estación y hora local predicha (truncada a la hora) |
| `demand_pred` | Predicción del modelo recortada a ≥ 0 (`clip(prediccion, 0, None)`, igual que en `src/modeling/predict.py`) |
| `lags_source` | `request` (rezagos enviados por el cliente) o `feature_store` (calculados por el servicio) |
| `demand_lag_24h`, `demand_lag_168h`, `demand_same_hour_mean_7d` | Rezagos usados en la predicción |

## `POST /predict/raw`

Recibe `{"rows": [...]}` con exactamente las 13 columnas `input_columns` de `/model-info`, que son las
columnas de `models/reference_predictions.csv` excepto `fecha_hora`, `demand_count` y `demand_pred`.
`capacity`, `Origen_Id`, `is_weekend` e `is_holiday` deben ser enteros.
Responde `{"predictions": [<float>, ...], "model": {...}, "n_instances": N}`.

## Ejemplos

### curl (salidas reales del servicio 1.1.0)

Salidas copiadas de `reports/fase_final/logs/20_api_smoke.log` (sección A, mismas peticiones) y de la ejecución de
estos comandos contra el servicio.

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_uri":"models:/mibici-demand-forecaster/1","model_sha256_verified":true,"service_version":"1.1.0","detail":null}

# Con los tres rezagos (mismo resultado que la versión 1.0.0)
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
# {"predictions":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_pred":2.6064224732591033,"lags_source":"request","demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}],"model":{"name":"mibici-demand-forecaster","version":1,"uri":"models:/mibici-demand-forecaster/1","alias":"champion","algorithm":"hist_gradient_boosting"},"n_instances":1}

# Sin rezagos: los calcula el servicio con el feature store
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-12-31T18:00:00"}]}'
# {"predictions":[{"station_id":24,"datetime":"2025-12-31T18:00:00","demand_pred":3.1631559452523517,"lags_source":"feature_store","demand_lag_24h":0.0,"demand_lag_168h":1.0,"demand_same_hour_mean_7d":1.8571428571428572}],"model":{...},"n_instances":1}

# Sin rezagos, hora posterior al último dato (dentro de la ventana)
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2026-01-01T08:00:00"}]}'
# {"predictions":[{"station_id":24,"datetime":"2026-01-01T08:00:00","demand_pred":1.5051365530193872,"lags_source":"feature_store","demand_lag_24h":2.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.142857142857143}],"model":{...},"n_instances":1}

# Sin rezagos fuera de la ventana -> 422 con la ventana válida
curl -s -w " HTTP %{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-11-08T10:00:00"}]}'
# {"error":"outside_feature_store_window","message":"Sin rezagos solo se pueden predecir las horas del 2025-12-24T00:00:00 al 2026-01-01T23:00:00 (hora local; feature store del 2025-12-17T00:00:00 al 2025-12-31T23:00:00). Fuera de esa ventana envía los tres rezagos.","predictable_window":{"start":"2025-12-24T00:00:00","end":"2026-01-01T23:00:00"},"details":[{"loc":["body","instances",0,"datetime"],"type":"outside_feature_store_window","msg":"2025-11-08T10:00:00 (estación 24) está fuera de la ventana 2025-12-24T00:00:00 a 2026-01-01T23:00:00."}]} HTTP 422

# Rezagos incompletos -> 422
curl -s -w " HTTP %{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-12-31T18:00:00","demand_lag_24h":1.0}]}'
# {"error":"validation_error","message":"La petición no cumple el esquema de entrada.","details":[{"loc":["body","instances",0],"msg":"Se envían los tres rezagos o ninguno. Faltan: demand_lag_168h, demand_same_hour_mean_7d. Sin rezagos el servicio los calcula con el feature store.","type":"lags_incomplete"}]} HTTP 422

curl -s -w " HTTP %{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":99999,"datetime":"2025-11-08T10:00:00","demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
# {"error":"unknown_station","message":"Estaciones desconocidas para el modelo: [99999]. Consulta /model-info.","unknown_station_ids":[99999]} HTTP 404

curl -s -w " HTTP %{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_lag_24h":-1,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
# {"error":"validation_error","message":"La petición no cumple el esquema de entrada.","details":[{"loc":["body","instances",0,"demand_lag_24h"],"msg":"Debe ser mayor o igual que 0.0.","type":"greater_than_equal"}]} HTTP 422
```

En las salidas largas, `"model":{...}` abrevia el mismo objeto `model` del primer ejemplo.

### PowerShell (`Invoke-RestMethod`, salidas reales)

```powershell
# Una instancia con rezagos y otra sin rezagos (feature store)
$body = @{ instances = @(@{ station_id = 80; datetime = "2025-11-08T11:00:00"; demand_lag_24h = 1.0;
           demand_lag_168h = 2.0; demand_same_hour_mean_7d = 1.4285714285714286 },
           @{ station_id = 80; datetime = "2026-01-01T08:00:00" }) } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body
$r.predictions | Format-Table station_id, datetime, demand_pred, lags_source, demand_lag_24h
# station_id datetime                   demand_pred lags_source   demand_lag_24h
# ---------- --------                   ----------- -----------   --------------
#         80 2025-11-08T11:00:00 2.2542647065879575 request                  1.0
#         80 2026-01-01T08:00:00 1.6360795542990862 feature_store            2.0
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
(en español) y, según el caso, `details` (lista de `{loc, msg, type}`), `unknown_station_ids` o
`predictable_window`.

| HTTP | `error` | Cuándo |
|---|---|---|
| 422 | `validation_error` | Tipos incorrectos, negativos, NaN/Infinity, fecha inválida o numérica, campo faltante o extra, rezagos incompletos (`lags_incomplete`: uno o dos de los tres), lote vacío (`too_short`) o mayor que `serving.max_batch_size` (`too_long`), JSON mal formado (`json_invalid`), columnas incorrectas en `/predict/raw` (`columns_mismatch`, `int_expected`) |
| 422 | `outside_feature_store_window` | Instancia sin rezagos cuya hora está fuera de la ventana predecible del feature store; `predictable_window` trae `start` y `end`, y `details` las instancias afectadas |
| 422 | `feature_store_unavailable` | Instancia sin rezagos con un modelo exportado sin feature store (formato 1.0.0) |
| 404 | `unknown_station` | Alguna `station_id` no está en el catálogo del modelo; `unknown_station_ids` lista todas |
| 404 | `not_found` | Ruta inexistente |
| 405 | `method_not_allowed` | Método HTTP no permitido en la ruta |
| 503 | `model_unavailable` | El modelo no está cargado (carpeta inexistente, SHA-256 del modelo o del feature store distinto, esquema incompatible); `/health` también responde 503 con `detail` |
| 500 | `internal_error` | Error no previsto; el mensaje es genérico y la traza solo va al log del servidor |

Ejemplo real de 503 (servicio levantado con `MODEL_DIR=models/no_existe`): ver la sección C de
`reports/fase_final/logs/20_api_smoke.log`.

## Configuración (`serving:` en `config.yaml`)

| Clave | Valor | Uso |
|---|---|---|
| `model_dir` | `models/champion` | Carpeta del modelo (sobrescribible con `MODEL_DIR`) |
| `metadata_file` | `models/champion_metadata.json` | Metadatos (sobrescribible con `MODEL_METADATA`) |
| `stations_file` | `stations.csv` | Catálogo de estaciones dentro de `model_dir` |
| `feature_store_file` | `feature_store.csv.gz` | Feature store dentro de `model_dir` |
| `feature_store_hours` | `360` | Horas de demanda que guarda la exportación (debe superar el rezago de 168 h) |
| `feature_store_reference_file` | `models/reference_feature_store_predictions.csv` | Patrón oro de `/predict` sin rezagos |
| `feature_store_reference_rows` | `200` | Filas de ese patrón oro (semilla `seed`) |
| `service_version` | `1.1.0` | Versión del servicio en OpenAPI y `/health` |
| `max_batch_size` | `1000` | Máximo de instancias por petición |
| `local_utc_offset` | `-06:00` | Desfase usado para convertir fechas con zona horaria |
| `require_sha256_match` | `true` | No cargar un modelo (o un feature store) cuyo SHA-256 no coincide |
| `strict_startup` | `false` | `true`: un error de carga detiene el arranque en lugar de responder 503 |
| `prediction_log_file` | `null` | Registro JSONL de predicciones (apagado); la variable `PREDICTION_LOG` tiene prioridad |

Si los rezagos de `features` en `config.yaml` o las `input_columns` de los metadatos dejan de coincidir
con los campos del esquema de `/predict` (`LAG_FIELDS` en `src/serving/schemas.py`), el servicio no
carga el modelo y lo explica en `/health` (`detail`).

## Verificación realizada

- `reports/fase_final/logs/20_api_smoke.log` (servicio 1.1.0): peticiones y respuestas reales
  (A: uvicorn en el venv del proyecto, Windows; B: contenedor Linux `ml-service:latest`, que solo tiene
  `requirements-serve.txt`; C: sin modelo → 503).
  - A: las 500 filas de `models/reference_predictions.csv` dieron diferencia máxima absoluta `0.0` por `/predict`
    y por `/predict/raw` (500/500 iguales bit a bit), y las 200 de `models/reference_feature_store_predictions.csv`
    por `/predict` sin rezagos, `0.0` (200/200).
  - B: `8.881784197001252e-16` (495/500 iguales bit a bit; 1 ulp del `exp()` de glibc, ver `docs/docker.md`) con
    rezagos y `0.0` (200/200) sin rezagos.
- `reports/fase_final/logs/20_api_server.log`: log de los tres servidores (carga del modelo y del feature store,
  una línea por petición con su latencia). El historial de la versión 1.0.0 está en Git.
- `reports/fase_final/logs/21_prediction_log_contenedor.log`: registro `PREDICTION_LOG` dentro del contenedor.
- Pruebas: `pytest tests/test_serving.py tests/test_serving_feature_store.py`.
