# Pruebas

Las pruebas usan **pytest** y corren desde la raíz del repositorio con un solo comando:

```bash
pytest -q
```

Otras formas útiles:

```bash
pytest -q -m "not integration"                 # solo pruebas rápidas (sin el flujo de punta a punta)
pytest -q -m integration                       # solo integración
pytest -q --cov=src --cov-report=term-missing  # con cobertura por módulo (la que corre CI)
```

La configuración está en `pyproject.toml` (`[tool.pytest.ini_options]` y `[tool.coverage.*]`).
CI (`.github/workflows/tests.yml`) corre `pytest -q --cov=src --cov-report=term-missing` con Python 3.13.

## Aislamiento

- Ninguna prueba usa la red (`requests.get` se sustituye con `monkeypatch`), los datos reales de `data/`
  ni la base `mlruns.db` del repositorio. Todo se genera con datos sintéticos en directorios temporales
  (`tmp_path` / `tmp_path_factory`).
- MLflow usa una base SQLite temporal por configuración: se migra una sola vez por sesión
  (`mlflow_db_template`) y se copia. `MLFLOW_TRACKING_URI` apunta a una base temporal durante la sesión.
- La única lectura de archivos reales es la de `models/champion_metadata.json` y
  `models/reference_predictions.csv` (artefactos versionados) en `test_versioned_artifacts.py`.

## Archivos

| Archivo | Qué cubre |
|---|---|
| `conftest.py` | Fixtures: configuración aislada (`session_config`, `config`), datos sintéticos crudos, limpios, cuadrícula, matriz de características, particiones y un modelo pequeño entrenado una vez por sesión |
| `conftest_data.py` | Generador de viajes sintéticos con anomalías en cantidades conocidas y configuraciones temporales (lo reutilizan otras pruebas) |
| `unit/test_dataset_reading.py` | `read_month` (UTF-8, latin-1, comillas mal cerradas, columnas y encabezado incorrectos), descarga simulada, catálogo de estaciones |
| `unit/test_dataset_cleaning.py` | Una prueba por regla de limpieza con conteos calculados a mano, imputación, métricas de retención, `run`/`main` sin red |
| `unit/test_features.py` | Festivos de México, codificación cíclica, fin de semana, cuadrícula con ceros, rezagos exactos y sin fuga de información, historial incompleto, `temporal_split`, columnas del modelo |
| `unit/test_preprocessing.py` | Preprocesador ajustado solo con train, sin NaN en la salida, número y orden de columnas |
| `unit/test_metrics.py` | `evaluate`: MAE, RMSE, R² y devianza de Poisson con ejemplos a mano, recorte de negativos, prefijos |
| `unit/test_modeling.py` | `build_estimator`, `build_pipeline`, `suggest_params` (con trials de Optuna reales), reproducibilidad, CLI de entrenamiento |
| `unit/test_config_tracking.py` | `load_config` con sobrescrituras, rutas, URI de MLflow, semillas, SHA-256 contra hashes conocidos, versionado de datos en MLflow |
| `unit/test_inference.py` | `predict` (columnas faltantes, no negativos, salida), carpeta exportada que reproduce la referencia con diferencia 0 |
| `unit/test_plots.py` | Figuras del EDA y de evaluación (PNG válidos) |
| `integration/test_pipeline_e2e.py` | Carga → limpieza → características → Optuna → Model Registry temporal → recarga del champion → predicción → métricas |
| `integration/test_export_champion.py` | Exportación del champion del Registry temporal: metadatos, hashes y referencia reproducida exactamente |
| `test_serving*.py`, `test_drift*.py` | API de FastAPI y simulación de data drift (paquetes de serving y monitoreo; usan las mismas fixtures) |
| `integration/test_versioned_artifacts.py` | Coherencia de los artefactos versionados en `models/`; si `models/champion/` existe localmente, que reproduce la referencia (en CI se omite) |

## Marcadores

| Marcador | Uso |
|---|---|
| `integration` | Flujo de punta a punta con MLflow y Optuna (unos 30 s). Se excluye con `-m "not integration"` |

`--strict-markers` hace que un marcador no registrado sea un error.
