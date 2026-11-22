# Sistema MLOps: Predicción de Demanda en MiBici Guadalajara - Equipo 19

Proyecto de la materia Operaciones de aprendizaje automático (TC5061), Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey.

- Repositorio: https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici
- Integrantes:
  - José Antonio Vázquez Martínez (A01797208) - Data Engineer & ML Engineer
  - Jesús Ángel Rincón Ruiz (A01793960) - Data Scientist
  - Erick Francisco Siller Ojeda (A01382929) - Software Engineer
  - Gustavo Alejandro Vázquez Rivas (A01611559) - Site Reliability Engineer (SRE)

## Caso de negocio

Las estaciones de MiBici se vacían o se saturan en las horas pico. El modelo estima las **extracciones de bicicletas por estación y hora** (`demand_count`) con 1 a 24 horas de anticipación para planear el rebalanceo con camionetas antes de que ocurra el desabasto.

---

## Estructura del proyecto (Cookiecutter Data Science v2)

```text
├── .github/workflows/     <- CI: pruebas, verificación del modelo y construcción de la imagen en cada pull request
├── Dockerfile             <- Imagen del servicio FastAPI (Entrega Final)
├── dvc.yaml               <- Etapas del pipeline declaradas para DVC (dependencias, parámetros y salidas)
├── Makefile               <- Comandos del pipeline (make data, make train, ...)
├── README.md
├── config.yaml            <- Configuración única: rutas, meses, reglas, semillas, modelos e hiperparámetros
├── pyproject.toml         <- Metadatos del paquete (pip install -e .)
├── requirements.txt       <- Dependencias
├── environment.yml        <- Entorno Conda reproducible
│
├── data/
│   ├── external/          <- Datos de terceros (p. ej. nuevas horas a predecir)
│   ├── raw/               <- Datos originales inmutables de MiBici
│   ├── interim/           <- Viajes limpios
│   └── processed/         <- Matriz estación-hora para el modelo y predicciones
│
├── models/
│   └── champion/          <- Paquete de despliegue exportado de models:/mibici-demand-forecaster/1
├── notebooks/
│   └── 1.0-eq19-eda-mibici.ipynb   <- Exploración; importa los módulos de src/
├── references/
│   └── diccionario_datos.md
├── reports/
│   ├── figures/           <- Figuras del EDA y comparación de modelos
│   ├── fase2/             <- Rol y reflexión de cada integrante en la Fase 2
│   ├── drift/             <- Resultados de la simulación de data drift y alertas
│   ├── reproducibility/   <- Verificaciones del modelo en distintos entornos
│   ├── model_comparison.csv
│   ├── ML_Canvas_MiBici.md
│   └── roles_mlops.md
│
├── src/                   <- Código fuente del proyecto
│   ├── config.py          <- Carga de config.yaml y rutas del proyecto
│   ├── dataset.py         <- MiBiciDataDownloader + MiBiciDataCleaner
│   ├── features.py        <- MiBiciFeatureEngineer, partición temporal y preprocesador
│   ├── plots.py           <- Gráficas del EDA y de evaluación
│   ├── tracking.py        <- Conexión a MLflow, hashes SHA-256 y versionado de datos
│   ├── modeling/
│   │   ├── pipeline.py    <- Pipeline de scikit-learn (preprocesamiento + modelo) y métricas
│   │   ├── train.py       <- Optuna + MLflow Tracking + Model Registry
│   │   └── predict.py     <- Predicción por lotes con el modelo `champion`
│   ├── serving/
│   │   ├── bundle.py      <- Paquete de despliegue: modelo, feature store, ficha y referencias
│   │   ├── export_model.py <- Exporta una versión del Model Registry al paquete
│   │   ├── service.py     <- Construcción de variables en línea y predicción
│   │   ├── schemas.py     <- Esquemas Pydantic de entrada y salida
│   │   ├── app.py         <- API FastAPI (/predict, /health, /model-info)
│   │   └── verify.py      <- Verificación de reproducibilidad en cualquier entorno
│   └── monitoring/
│       └── drift.py       <- Simulación de data drift, PSI/KS, umbrales y alertas
│
└── tests/                 <- Pruebas unitarias y de integración
```

---

## Cómo reproducir el proyecto

### 1. Entorno

```bash
git clone https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici.git
cd Operacion-de-aprendizaje-automatico---Proyecto-Bici

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Pipeline completo

```bash
python -m src.dataset            # descarga + limpieza            -> data/raw, data/interim
python -m src.features           # características y rezagos       -> data/processed
python -m src.plots              # figuras del EDA                 -> reports/figures
python -m src.modeling.train     # Optuna + MLflow + Model Registry
python -m src.modeling.predict   # predicción con el modelo champion
```

Con `make` disponible: `make pipeline`. Todos los comandos aceptan `--config otra_config.yaml`.

Opciones útiles del entrenamiento:

```bash
python -m src.modeling.train --models poisson hist_gradient_boosting   # solo algunos modelos
python -m src.modeling.train --n-trials 5                               # prueba rápida
```

### 3. Pruebas

```bash
python -m pytest -v
```

El flujo `.github/workflows/tests.yml` ejecuta la misma suite en GitHub Actions en cada pull request y en cada push a `main`.

La prueba de integración entrena, registra y vuelve a cargar el modelo `champion` en una base de MLflow temporal.

### 4. MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

- Experimento `MiBici_Datos`: versiones raw, interim y processed con su hash SHA-256.
- Experimento `MiBici_Fase2_Modelos`: una corrida por modelo (con sus trials de Optuna anidados) y la corrida `00_Comparativa_Modelos`.
- Pestaña **Models**: `mibici-demand-forecaster`, con una versión por algoritmo y los alias `champion` y `challenger`.

---

## Decisiones de modelado

| Decisión | Justificación |
|---|---|
| Rezagos de 24 h y 168 h (se eliminó el de 1 h) | El Canvas plantea un horizonte de 1 a 24 h; el rezago de 1 h no estaría disponible al predecir con más anticipación |
| Partición cronológica 70/15/15 | Evita usar información del futuro para predecir el pasado |
| Selección por MAE de validación; test se usa una sola vez | El modelo final se reentrena con train + val y se evalúa en test |
| Regresión de Poisson | Línea base interpretable para conteos |
| HistGradientBoosting con pérdida Poisson | Captura interacciones no lineales (estación × hora) y escala a millones de filas |
| Random Forest | Ensamble por *bagging* como punto de comparación |
| Sin PCA | En la Fase 1 empeoró el RMSE y con 13 variables no aporta |

## Ampliar el periodo de datos

Agrega los meses en `data.months` de `config.yaml`. Si el CSV `datos_abiertos_AAAA_MM.csv` ya está en `data/raw/` se usa directamente; si no, agrega su URL en `data.trip_urls`. Con un año de datos conviene fijar `tuning.train_sample_rows` para acelerar Optuna.

### Datos de entrada (descarga manual)

El portal de MiBici bloquea la descarga automática (HTTP 406), por lo que los archivos se descargan a mano desde https://www.mibici.net/es/datos-abiertos/ y se colocan en `data/raw/` con estos nombres:

- Viajes: `datos_abiertos_2025_01.csv` ... `datos_abiertos_2025_12.csv`
- Catálogo de estaciones: `nomenclatura_2026_09.csv`

`src/dataset.py` usa primero los archivos presentes en `data/raw/` y solo intenta descargar los que falten. Lee archivos en UTF-8 o latin-1 y tolera registros con comillas mal cerradas (caso de julio de 2025), reportando cuántas filas descarta.

### Resultados de la Fase 2 (datos 2025)

| Modelo | MAE val. | MAE prueba | RMSE prueba | R² prueba | Registry |
|---|---|---|---|---|---|
| HistGradientBoosting (Poisson) | 0.804 | **0.766** | 1.304 | 0.693 | v1 · `champion` |
| RandomForest | 0.826 | 0.783 | 1.351 | 0.670 | v2 · `challenger` |
| PoissonRegressor | 1.249 | 1.098 | 3.115 | -0.753 | v3 |
| Dummy (media) | 1.630 | 1.535 | 2.355 | -0.002 | — |


---

## Entrega Final: pruebas, servicio, reproducibilidad, contenedor y data drift

### Resumen

| Requisito | Implementación | Comando |
|---|---|---|
| Pruebas unitarias y de integración | `tests/` con pytest: datos, características, modelos, servicio, API, drift y una prueba de extremo a extremo | `pytest -q` |
| Servicio con FastAPI | `src/serving/app.py`: `POST /predict`, `GET /health`, `GET /model-info`, Swagger en `/docs` | `uvicorn src.serving.app:app --port 8000` |
| Artefacto del modelo | `models:/mibici-demand-forecaster/1` (alias `champion`), exportado a `models/champion/` | `python -m src.serving.export_model` |
| Reproducibilidad | Versiones fijas, semilla global, hashes SHA-256, artefactos en MLflow y verificación en 3 entornos | `python -m src.serving.verify --full` |
| Contenedor | `Dockerfile` (Python 3.12, usuario sin privilegios, healthcheck) | `podman build --format docker -t mibici-demand-api:1.0.0 .` |
| Data drift | `src/monitoring/drift.py`: 4 escenarios simulados + monitoreo semanal, PSI, KS, umbrales y alertas | `python -m src.monitoring.drift` |

### Artefacto del modelo desplegado

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-demand-forecaster` |
| URI de la versión | `models:/mibici-demand-forecaster/1` |
| URI por alias | `models:/mibici-demand-forecaster@champion` |
| Algoritmo | HistGradientBoostingRegressor (pérdida Poisson) dentro de un `Pipeline` de scikit-learn |
| Corrida de entrenamiento | `b230b27936e2439db0c9eae68986d884` (experimento `MiBici_Fase2_Modelos`) |
| Paquete de despliegue | `models/champion/` (hash del modelo en `model_card.json`) |
| Imagen | `docker.io/angrinconr/mibici-demand-api:1.0.0` (también `:latest`): https://hub.docker.com/r/angrinconr/mibici-demand-api |

`models/champion/` contiene el pipeline (`model.joblib`), la ficha del modelo (`model_card.json`: URI, versión, run, métricas, hashes y versiones de librerías), el *feature store* con las últimas 360 horas de demanda por estación, el catálogo de estaciones y 1,000 predicciones de referencia. El servicio no necesita la base de MLflow: usa esta copia identificada del modelo registrado.

### 1. Pruebas

```bash
pytest -q                      # toda la suite (datos sintéticos en una carpeta temporal)
pytest -q tests/test_api.py    # solo la API
```

| Archivo | Qué valida |
|---|---|
| `test_pipeline.py` | Limpieza, cuadrícula estación-hora, rezagos sin fuga, días festivos, partición cronológica, preprocesador |
| `test_modeling.py` | Cada modelo de `config.yaml` entrena y predice, semilla, reproducibilidad e integración con MLflow + Optuna |
| `test_integration_e2e.py` | Carga → limpieza → características → entrenamiento → predicción → métricas → paquete → servicio, y que la API calcule las mismas variables y predicciones que el pipeline (sin *training-serving skew*) |
| `test_serving.py` | Feature store, rezagos enviados por el cliente, estaciones desconocidas, zona horaria, tipos de entrada, esquemas Pydantic y semilla global |
| `test_api.py` | `/health`, `/model-info`, `/predict` (200 y 422) y el esquema OpenAPI |
| `test_drift.py` | PSI, escenarios de drift, reglas de alerta y el monitor de extremo a extremo |

### 2. Servicio con FastAPI

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# Documentación interactiva: http://localhost:8000/docs
```

Entrada (`POST /predict`). El cliente envía estación y fecha-hora; los rezagos se toman del *feature store* o se pueden enviar:

```json
{
  "instances": [
    {"station_id": 2, "fecha_hora": "2025-12-31T18:00:00"},
    {"station_id": 3, "fecha_hora": "2026-03-01T08:00:00",
     "demand_lag_24h": 2, "demand_lag_168h": 1, "demand_same_hour_mean_7d": 1.4}
  ]
}
```

Salida:

```json
{
  "model_name": "mibici-demand-forecaster",
  "model_version": "1",
  "model_uri": "models:/mibici-demand-forecaster/1",
  "n_predictions": 1,
  "predictions": [
    {"station_id": 2, "fecha_hora": "2025-12-31T18:00:00", "demand_pred": 3.4493,
     "lags_source": "feature_store", "demand_lag_24h": 1.0, "demand_lag_168h": 3.0,
     "demand_same_hour_mean_7d": 1.7143}
  ]
}
```

Errores: `422` si el JSON no cumple el esquema (Pydantic), si la estación no existe o si no hay historial para calcular los rezagos; `503` si el modelo no se pudo cargar. Con `PREDICTION_LOG=logs/predicciones.jsonl` cada solicitud se registra (entradas, salidas, versión del modelo y latencia) para auditoría y monitoreo.

### 3. Reproducibilidad en otro entorno

- Versiones fijas en `requirements.txt` (entrenamiento) y `requirements-api.txt` (servicio); scikit-learn, NumPy, pandas y SciPy coinciden con las que MLflow guardó junto al modelo.
- `load_config()` fija la semilla de Python, NumPy y `PYTHONHASHSEED`; los estimadores, el TargetEncoder y Optuna reciben `random_state` desde `config.yaml`.
- Artefactos versionados en MLflow: cada versión de datos (raw, interim, processed) con su hash SHA-256 en `MiBici_Datos`, el modelo en el Model Registry y el paquete de despliegue completo (modelo, feature store y predicciones de referencia) en la corrida `05_Export_mibici-demand-forecaster_v1` del experimento `MiBici_Despliegue`.
- `dvc.yaml` declara las etapas del pipeline (datos → características → entrenamiento → exportación → drift) con sus dependencias y parámetros de `config.yaml`, para que `dvc repro` vuelva a ejecutar solo lo que cambió.

Entornos verificados (resultados en `reports/reproducibility/`):

| Entorno | Python | Diferencia máxima en 1,000 predicciones | MAE de prueba |
|---|---|---|---|
| Windows 11 (entrenamiento) | 3.12.10 | 0 | 0.76559 |
| Linux (otra máquina) | 3.13.18 | 8.9e-16 | 0.76559 |
| Contenedor Podman (Linux, WSL2) | 3.12.16 | 8.9e-16 | — |

```bash
python -m src.serving.verify --full --tag windows-local --output reports/reproducibility
podman run --rm mibici-demand-api:1.0.0 python -m src.serving.verify --tag contenedor
```

`verify` compara el hash del modelo y 1,000 predicciones de referencia (tolerancia de 1e-6 bicicletas por hora; entre Windows/Python 3.12 y Linux/Python 3.13 la diferencia medida fue de ~1e-15) y, con `--full`, recalcula el MAE del periodo de prueba contra el registrado en el Model Registry.

### 4. Contenedor

Imagen publicada: https://hub.docker.com/r/angrinconr/mibici-demand-api (tags `1.0.0` y `latest`, ~165 MB comprimida). Para usarla sin clonar el repositorio:

```bash
podman run --rm -p 8000:8000 docker.io/angrinconr/mibici-demand-api:1.0.0
```


```bash
podman build --format docker -t mibici-demand-api:1.0.0 -t mibici-demand-api:latest .
podman run --rm -p 8000:8000 mibici-demand-api:1.0.0
# Publicar en Docker Hub
podman login docker.io
podman tag mibici-demand-api:1.0.0 docker.io/angrinconr/mibici-demand-api:1.0.0
podman tag mibici-demand-api:1.0.0 docker.io/angrinconr/mibici-demand-api:latest
podman push docker.io/angrinconr/mibici-demand-api:1.0.0
podman push docker.io/angrinconr/mibici-demand-api:latest
```

En Windows, si `localhost:8000` no responde con el contenedor en ejecución (Podman sobre WSL en modo *rootless* no reenvía bien el puerto), cambiar la máquina a modo *rootful* y reconstruir la imagen: `podman machine stop; podman machine set --rootful; podman machine start`.

Con Docker los comandos son iguales cambiando `podman` por `docker` (`docker build -t ml-service:latest .` y `docker run -p 8000:8000 ml-service:latest` también funcionan). Tags: `MAJOR.MINOR.PATCH` de la API; una nueva versión del modelo en el Model Registry implica un nuevo tag de la imagen.

### 5. Data drift

```bash
python -m src.monitoring.drift            # resultados en reports/drift/ y MLflow (MiBici_Monitoreo)
python -m src.monitoring.drift --evidently  # además, reportes HTML de Evidently (opcional)
```

Línea base: el periodo de prueba (8 nov - 31 dic 2025), que el champion no vio. Referencia de distribuciones: train + val.

| Señal | Advertencia | Alerta |
|---|---|---|
| Aumento del MAE contra la línea base | ≥ 10 % | ≥ 25 % |
| Sesgo de la predicción | ≥ ±10 % | ≥ ±20 % |
| PSI de una variable | ≥ 0.10 | ≥ 0.25 |
| Filas con variables faltantes | | ≥ 5 % |

| Diagnóstico | Acción |
|---|---|
| Faltantes sobre el umbral | Revisar el feature pipeline / feature store; no reentrenar con datos incompletos |
| Drift en las entradas sin cambio en la demanda real | Probable error en el cálculo de variables: revisar el feature pipeline |
| Pérdida de desempeño o sesgo con demanda distinta | Reentrenar con datos recientes, comparar con el challenger y promover por alias |
| Cambio moderado | Aumentar la frecuencia de monitoreo y programar reentrenamiento si persiste |

---

## Declaración de uso de Inteligencia Artificial

Ver `reports/declaracion_ia.md`. Resumen:

- **Fase 1:** Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como herramienta de consulta y de apoyo en la revisión de la redacción, la organización, el formato del reporte y la rúbrica.
- **Fase 2:** Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como apoyo en la reestructuración del repositorio a Cookiecutter Data Science v2, la refactorización a módulos configurables, la generación del código de entrenamiento con Optuna, MLflow Tracking y Model Registry, las pruebas automatizadas y la documentación. El equipo revisó, ejecutó y validó el código; los resultados reportados corresponden a la ejecución real del pipeline sobre los datos abiertos de MiBici, registrada en MLflow. https://claude.ai/
