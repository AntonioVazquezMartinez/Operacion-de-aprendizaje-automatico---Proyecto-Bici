# Sistema MLOps: predicción de demanda horaria por estación en MiBici Guadalajara (Equipo 19)

Proyecto de la materia Operaciones de aprendizaje automático (TC5061), Maestría en Inteligencia Artificial
Aplicada, Tecnológico de Monterrey. Integra las Fases 1, 2 y final: datos versionados, experimentos trazables,
modelo registrado, pruebas automatizadas, API REST, contenedor publicado y monitoreo de data drift.

- Repositorio: https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici
- Imagen Docker: https://hub.docker.com/r/gusrivasmna/mibici-demand-service
- Modelo en producción: **`models:/mibici-demand-forecaster/1`** (alias `champion`)
- Reporte final: `reports/Entrega_Final_Equipo19.docx` (generado con `python reports/build_report.py`)

| Integrante | Matrícula | Rol |
|---|---|---|
| José Antonio Vázquez Martínez | A01797208 | Data Engineer y ML Engineer |
| Jesús Ángel Rincón Ruiz | A01793960 | Data Scientist |
| Erick Francisco Siller Ojeda | A01382929 | Software Engineer |
| Gustavo Alejandro Vázquez Rivas | A01611559 | Site Reliability Engineer (SRE) |

Roles, actividades por fase y matriz RACI: `reports/roles_mlops.md`. Análisis de cada integrante en la fase final:
`reports/fase_final/<nombre>.md`.

## Caso de negocio

Las estaciones de MiBici se vacían o se saturan en las horas pico (8:00 y 18:00) y el rebalanceo con camionetas
es reactivo. El modelo estima las **extracciones de bicicletas por estación y hora** (`demand_count`) con 1 a 24
horas de anticipación para programar la redistribución antes del desabasto. Planteamiento completo en
`reports/ML_Canvas_MiBici.md`.

![Arquitectura de la solución](reports/figures/00_arquitectura.png)

---

## Requisitos de la entrega final y dónde está la evidencia

| Requisito | Evidencia |
|---|---|
| 1. Pruebas unitarias y de integración con pytest, comando único | `pytest -q` → 270 aprobadas, cobertura 93 % (`reports/fase_final/logs/10_pytest_cobertura.log`); `tests/README.md`; sección [Pruebas](#pruebas) |
| 2. API con FastAPI, `POST /predict`, validación Pydantic, esquema documentado | `src/serving/`, `docs/api.md`, `docs/openapi.json`, `docs/postman_collection.json`; sección [API](#api-de-predicción) |
| 2. Ruta y versión del artefacto del modelo en el README | Sección [Artefacto del modelo](#artefacto-del-modelo): `models:/mibici-demand-forecaster/1` |
| 3. Dependencias fijadas, semillas, entorno limpio, comparación con referencia | `requirements*.txt` (`==`), `config.yaml` (`seed: 42`), `reports/fase_final/comparacion_fase2.md`, `reports/fase_final/reproducibilidad/`; sección [Reproducibilidad](#reproducibilidad) |
| 3. Artefactos versionados (DVC / MLflow) | `dvc.yaml`, `dvc.lock`, `docs/dvc.md`, MLflow Model Registry; sección [Versionado](#versionado-de-datos-y-modelos) |
| 4. Dockerfile, `docker build` / `docker run`, DockerHub con tags | `Dockerfile`, `docs/docker.md`; sección [Docker](#docker) |
| 5. Data drift: datos alterados, métricas vs. línea base, gráficos, umbrales, alertas y acción | `src/monitoring/`, `reports/drift/drift_report.md`, figuras 07-10; sección [Data drift](#data-drift) |

---

## Estructura del proyecto (Cookiecutter Data Science v2)

```text
├── .github/workflows/
│   ├── tests.yml                    <- pytest + cobertura (Python 3.13) en push a main/fase-final y en PR
│   └── docker-reproducibilidad.yml  <- hadolint y verificación de la imagen publicada en DockerHub
├── Dockerfile, .dockerignore, docker-compose.yml   <- Imagen del servicio (FastAPI + modelo champion)
├── docker/                <- Imagen y scripts para reentrenar el pipeline completo en Linux (comparación de entornos)
├── Makefile               <- make test, make serve, make drift, make docker-build, make report, ...
├── config.yaml            <- Configuración única: rutas, URL, reglas, semilla, modelos, serving, monitoreo
├── dvc.yaml, dvc.lock     <- Pipeline de DVC (data → features → train → export → drift) y huellas
├── pyproject.toml         <- Paquete, configuración de pytest y cobertura
├── requirements.txt       <- Dependencias con versión exacta (entrenamiento, pruebas, DVC, reporte)
├── requirements-serve.txt <- Dependencias mínimas del servicio (imagen Docker)
├── requirements-lock.txt  <- pip freeze completo del entorno de entrenamiento
├── environment.yml        <- Entorno Conda (alternativo)
│
├── data/                  <- raw/ (CSV del portal), interim/ (viajes limpios), processed/ (matriz estación-hora); no van a Git
├── docs/                  <- api.md, openapi.json, postman_collection.json, docker.md, dvc.md
├── models/
│   ├── champion/          <- Modelo MLflow exportado (no va a Git; DVC lo rastrea y viaja en la imagen)
│   ├── champion_metadata.json      <- Versión, URI, SHA-256, métricas, columnas, librerías
│   └── reference_predictions.csv   <- 500 predicciones de referencia (patrón oro de reproducibilidad)
├── notebooks/1.0-eq19-eda-mibici.ipynb
├── references/diccionario_datos.md
├── reports/
│   ├── build_report.py, report_data.py, docx_helpers.py   <- Generador del reporte final en Word
│   ├── Entrega_Final_Equipo19.docx, Fase 1_Equipo19.*, Fase2_Equipo19.docx
│   ├── drift/             <- drift_report.md / .json, drift_summary.csv
│   ├── fase2/, fase_final/ <- Archivos por integrante, comparación con Fase 2, logs y reproducibilidad
│   ├── figures/           <- 00 y 00b arquitectura y ciclo MLOps, 01-05 EDA, 06 modelos, 07-10 drift, 11 GitHub
│   ├── ML_Canvas_MiBici.md, roles_mlops.md, declaracion_ia.md, model_comparison.csv
├── scripts/               <- api_smoke.py, verify_reproducibility.py, export_openapi.py, docker_smoke.*,
│                             build_architecture_diagram.py
├── src/
│   ├── config.py          <- config.yaml, rutas y semillas globales (set_global_seeds)
│   ├── dataset.py         <- Descarga y limpieza
│   ├── features.py        <- Cuadrícula estación-hora, calendario, rezagos, partición, preprocesador
│   ├── plots.py           <- Figuras del EDA y de evaluación
│   ├── tracking.py        <- MLflow, SHA-256 y versionado de datos
│   ├── modeling/          <- pipeline.py, train.py (Optuna + Registry), predict.py, export.py
│   ├── serving/           <- app.py (FastAPI) y schemas.py (Pydantic v2)
│   └── monitoring/        <- scenarios, metrics, rules, figures, report y drift (CLI)
└── tests/
    ├── unit/              <- Pruebas unitarias por módulo
    ├── integration/       <- Flujo de punta a punta, exportación y artefactos versionados
    └── test_serving.py, test_drift.py, conftest.py
```

---

## Inicio rápido

```bash
git clone https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici.git
cd Operacion-de-aprendizaje-automatico---Proyecto-Bici
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q                      # 270 pruebas (no requieren datos ni MLflow)
```

Solo para usar el servicio, sin instalar nada de Python: ver [Docker](#docker).

## Pipeline completo

```bash
python -m src.dataset            # descarga (URL de config.yaml) + limpieza     -> data/raw, data/interim
python -m src.features           # características y rezagos                    -> data/processed
python -m src.plots              # figuras del EDA                              -> reports/figures
python -m src.modeling.train     # Optuna + MLflow Tracking + Model Registry
python -m src.modeling.predict   # predicción del periodo de prueba con @champion
python -m src.modeling.export    # models/champion + metadatos + 500 predicciones de referencia
python -m src.monitoring.drift   # simulación de data drift                      -> reports/drift
```

Equivalentes: `make pipeline` o `dvc repro`. Todos los módulos aceptan `--config otra_config.yaml`.

**Datos.** `src/dataset.py` descarga automáticamente los 12 CSV mensuales de 2025 y el catálogo
`nomenclatura_2026_10.csv` con las URL de `config.yaml` (`data.trip_urls`, `data.stations_url`); el portal exige
un User-Agent de navegador, que el código envía. Si un CSV ya está en `data/raw/` se usa directamente. La lectura
tolera UTF-8 o latin-1 y valida el encabezado. El catálogo de la Fase 2 (`nomenclatura_2026_09.csv`) ya no está
publicado, y el portal republicó los CSV de 2025: hoy se leen 4,532,032 viajes contra 4,520,095 en la Fase 2, y
julio ya se lee con el lector estándar (detalle en `reports/fase_final/comparacion_fase2.md`).

**MLflow UI:** `mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000` (experimentos `MiBici_Datos`,
`MiBici_Fase2_Modelos`, `MiBici_Monitoreo`; pestaña Models: `mibici-demand-forecaster`).

---

## Pruebas

```bash
pytest -q                                          # comando único: unitarias + integración
pytest -q -m "not integration"                     # solo unitarias (rápidas)
pytest -q -m integration                           # solo integración (datos → entrenamiento → Registry → predicción)
pytest -q --cov=src --cov-report=term-missing      # con cobertura (lo que corre CI)
```

Resultado de la fase final (`reports/fase_final/logs/10_pytest_cobertura.log`): **270 aprobadas, 0 fallidas,
cobertura 93 % de `src/`** (líneas y ramas). Las pruebas usan datos sintéticos y una base de MLflow temporal: no
necesitan red, datos reales ni `mlruns.db`. Las dos pruebas que comparan contra las predicciones de referencia
usan tolerancia `1e-9` (ver [Reproducibilidad](#reproducibilidad)); con ella la suite completa también pasa en
un contenedor Linux con el modelo real: 270 aprobadas
(`reports/fase_final/reproducibilidad/pipeline_linux_pytest_v2.log`; la primera corrida, con igualdad exacta, dio
268/2 y está en `pipeline_linux_pytest.log` como antecedente). Detalle por archivo y marcadores en
`tests/README.md`. CI: `.github/workflows/tests.yml` corre la suite en cada push
([pestaña Actions](https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici/actions);
corrida exitosa del push final: [45060105154](https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici/actions/runs/45060105154));
ahí se omiten las pruebas que necesitan `models/champion/`, que no está en Git.

---

## API de predicción

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000     # o: make serve
```

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado: modelo cargado y SHA-256 verificado (200 / 503) |
| GET | `/model-info` | Nombre, versión, URI, alias, run_id, hiperparámetros, métricas, columnas y versiones |
| POST | `/predict` | Lote de 1 a 1,000 instancias: estación, fecha-hora y tres rezagos; el servidor deriva el resto |
| POST | `/predict/raw` | Filas con las 13 columnas del modelo (verificación de reproducibilidad) |

Ejemplo real (fila 1 de `models/reference_predictions.csv`, tomado de `reports/fase_final/logs/20_api_smoke.log`):

```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"instances":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_lag_24h":5.0,"demand_lag_168h":0.0,"demand_same_hour_mean_7d":3.5714285714285716}]}'
```

```json
{"predictions":[{"station_id":24,"datetime":"2025-11-08T10:00:00","demand_pred":2.6064224732591033}],
 "model":{"name":"mibici-demand-forecaster","version":1,"uri":"models:/mibici-demand-forecaster/1",
          "alias":"champion","algorithm":"hist_gradient_boosting"},"n_instances":1}
```

Validación con Pydantic v2: tipos estrictos, rezagos finitos y ≥ 0, fecha ISO-8601, sin campos extra; los errores
responden 422 (validación), 404 (estación desconocida) o 503 (modelo no cargado) con el mismo cuerpo JSON.
Documentación completa: [`docs/api.md`](docs/api.md); esquema OpenAPI: [`docs/openapi.json`](docs/openapi.json)
(también en `/docs` y `/openapi.json`); colección de Postman: [`docs/postman_collection.json`](docs/postman_collection.json).

---

## Artefacto del modelo

| Concepto | Valor |
|---|---|
| **Modelo y versión (MLflow Model Registry)** | **`models:/mibici-demand-forecaster/1`** |
| Alias | `champion` (`models:/mibici-demand-forecaster@champion`); `challenger` = v2 (RandomForest) |
| Algoritmo | `HistGradientBoostingRegressor` con pérdida Poisson (scikit-learn 1.9.1) |
| Carpeta que carga la API y la imagen | `models/champion/` (modelo MLflow autocontenido + `stations.csv`) |
| SHA-256 de `models/champion/model.pkl` | `ca4331a49948c99b29580463d89e8aa1dc8d14451f866c029588f3de67ce2ed5` |
| Metadatos | `models/champion_metadata.json` (run_id `9ef2f57d97374770bf956469ba97a02d`, hiperparámetros, SHA-256 del dataset, métricas, columnas, versiones) |
| Métricas en prueba (8-nov a 31-dic-2025) | MAE 0.7651 · RMSE 1.3018 · R² 0.6937 · devianza Poisson 0.9144 |
| Referencia de reproducibilidad | `models/reference_predictions.csv` (500 filas de prueba, semilla 42) |

Exportar desde el Registry (requiere `mlruns.db` de quien entrenó) y verificar:

```bash
python -m src.modeling.export                 # alias champion -> models/champion + metadatos + referencia
python -m src.modeling.export --verify-only   # recarga la carpeta y compara con la referencia
```

La API recalcula el SHA-256 al arrancar y no carga un modelo distinto al de los metadatos (responde 503).

---

## Docker

Comandos de la entrega (desde la raíz, con `models/champion/` presente):

```bash
docker build -t ml-service:latest .
docker run -p 8000:8000 ml-service:latest
```

Imagen publicada (no requiere el repositorio ni los datos):

```bash
docker pull gusrivasmna/mibici-demand-service:1.0.0
docker run -p 8000:8000 gusrivasmna/mibici-demand-service:1.0.0
```

| Tag | Significado |
|---|---|
| `1.0.0` | Versión del servicio (`serving.service_version`) |
| `1.0.0-model-v1` | Servicio 1.0.0 con `models:/mibici-demand-forecaster/1` |
| `latest` | Última imagen publicada |

Los tres tags apuntan al digest `sha256:c28158a1fa7f6f1e6dfe6bf482e00e5ddd60ee882ff4b21942e1bd98bb081fb6`
(127 MB comprimida, 555 MB en disco). DockerHub: https://hub.docker.com/r/gusrivasmna/mibici-demand-service.
Imagen multi-etapa sobre `python:3.13.9-slim` fijada por digest, usuario sin privilegios y `HEALTHCHECK`.
Detalle, política de tags y publicación: [`docs/docker.md`](docs/docker.md).

---

## Reproducibilidad

- **Dependencias fijadas:** `requirements.txt` y `requirements-serve.txt` con `==`; `requirements-lock.txt` con el
  entorno completo; imagen base fijada por digest.
- **Semillas:** `seed: 42` en `config.yaml`, inyectada en los estimadores, el `TargetEncoder`, Optuna (TPE) y en
  `set_global_seeds` (random, NumPy, `PYTHONHASHSEED`). Optuna corre sin corte por tiempo
  (`tuning.timeout_seconds: null`) para que el número de trials no dependa de la máquina.
- **Entorno limpio vs. Fase 2:** reentrenamiento completo en un venv nuevo; MAE de prueba 0.7651 contra 0.7656 de
  la Fase 2 (−0.06 %), mismos hiperparámetros y misma asignación de alias
  (`reports/fase_final/comparacion_fase2.md`).
- **Mismo artefacto en otros entornos:** `scripts/verify_reproducibility.py --url <servicio>` envía las 500 filas de
  la referencia y compara fila por fila (tolerancia por omisión `1e-9`).

| Entorno | Dif. máx. | Idénticas bit a bit | Veredicto | Evidencia |
|---|---:|---:|---|---|
| Windows 11, venv de entrenamiento | 0.0 | 500/500 | PASS | `reports/fase_final/reproducibilidad/windows_venv.json` |
| Contenedor Linux local (`ml-service:latest`) | 8.88e-16 | 495/500 | PASS | `reproducibilidad/contenedor_local.json` |
| Imagen descargada de DockerHub (`docker pull`) | 8.88e-16 | 495/500 | PASS | `reproducibilidad/imagen_dockerhub.json` |
| Runner de GitHub Actions (`ubuntu-latest`, imagen de DockerHub) | 8.88e-16 | 495/500 | PASS | `reproducibilidad/github_actions.json`, [corrida 45060105120](https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici/actions/runs/45060105120) |

La diferencia de 1 ulp en Linux está en el `exp()` final de la pérdida Poisson, que calcula la biblioteca
matemática del sistema (UCRT en Windows, glibc en Linux); la suma de los árboles es idéntica bit a bit
(`reproducibilidad/diagnostico_exp/resumen.txt`). Reentrenar el pipeline completo en un contenedor Linux da
DummyRegressor y RandomForest idénticos bit a bit, pero el boosting elige otros hiperparámetros (diferencia máxima
de métricas 9.5e-4) por ese mismo redondeo: lo reproducible entre sistemas operativos es el binario versionado y la
conclusión del experimento, no el reentrenamiento bit a bit
(`reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md`).

---

## Data drift

```bash
python -m src.monitoring.drift        # o: make drift
```

Referencia de distribución: conjunto de validación; línea base de desempeño: conjunto de prueba (MAE 0.7651,
WAPE 0.6025). Dos familias de escenarios: corrupción de entradas (la demanda real no cambia) y cambio real (se
alteran los viajes y se recalculan variables y objetivo), más un control sin alterar. Resultados
(`reports/drift/drift_summary.csv`):

| Escenario | Familia | MAE | ΔMAE | ΔWAPE | Nivel | Nivel sin etiquetas | Acción (regla) |
|---|---|---:|---:|---:|---|---|---|
| control | Control | 0.7651 | +0.0 % | +0.0 % | OK | OK | `R0_ok` |
| rezagos_faltantes | Corrupción | 0.8222 | +7.5 % | +7.5 % | ALERTA | ALERTA | `R1_pipeline` |
| rezagos_escala | Corrupción | 0.8843 | +15.6 % | +15.6 % | ALERTA | ALERTA | `R1_pipeline` |
| reloj_desfasado | Corrupción | 1.0499 | +37.2 % | +37.2 % | ALERTA | ALERTA | `R1_pipeline` |
| aumento_demanda | Cambio real | 1.1113 | +45.2 % | +3.7 % | ALERTA | AVISO | `R3_reentrenar` |
| caida_demanda | Cambio real | 0.6611 | −13.6 % | +43.8 % | ALERTA | ALERTA | `R3_reentrenar` |
| vacaciones_diciembre | Cambio real | 0.7190 | −6.0 % | +33.7 % | ALERTA | AVISO | `R3_reentrenar` |
| estaciones_nuevas | Cambio real | 0.7892 | +3.1 % | +3.1 % | ALERTA | ALERTA | `R2_catalogo` |

En las ventanas semanales del conjunto de prueba (sin alterar), el nivel pasa de OK a ALERTA a partir del
20-dic-2025 (WAPE +38.8 %): drift real por vacaciones. El MAE por sí solo no basta: baja cuando baja la demanda,
por eso se vigila también el WAPE.

**Umbrales** (`config.yaml` → `monitoring.thresholds`, aviso / alerta): PSI 0.10 / 0.25; estadístico D de KS
0.10 / 0.20; Wasserstein normalizada 0.10 / 0.25; faltantes, rezagos fuera de dominio y estaciones no vistas
1 % / 5 %; desfase del perfil horario 1 h / 2 h; degradación de MAE y WAPE +10 % / +25 %.
**Acciones:** R1 revisar el pipeline antes de reentrenar; R2 actualizar el catálogo; R3 reentrenar, evaluar el
challenger y mover el alias si mejora; R6/R7 vigilar o preparar un challenger cuando aún no hay etiquetas.
Reporte completo: [`reports/drift/drift_report.md`](reports/drift/drift_report.md); figuras
`reports/figures/07_drift_distribuciones.png` a `10_drift_ventanas_tiempo.png`.

---

## Versionado de datos y modelos

- **MLflow Tracking:** cada versión de datos (raw, interim, processed) con su SHA-256 y `mlflow.log_input`; cada
  corrida con parámetros, métricas y figuras.
- **MLflow Model Registry:** `mibici-demand-forecaster` v1 (`champion`), v2 (`challenger`), v3.
- **DVC:** `dvc.yaml` con cinco etapas (`data → features → train → export → drift`) que dependen del código, los
  datos y las secciones de `config.yaml`; `dvc.lock` guarda la huella MD5 de cada dependencia y salida (incluidos
  los 12 CSV crudos y la carpeta del modelo); `dvc status` → "Data and pipelines are up to date". Sin remoto
  externo: los datos son públicos y se descargan con `dvc repro`; el modelo viaja en la imagen. Ver
  [`docs/dvc.md`](docs/dvc.md).

```bash
dvc status    # ¿los artefactos locales corresponden a dvc.lock?
dvc dag       # grafo de etapas
dvc repro     # ejecuta solo las etapas afectadas por un cambio
```

---

## Resultados del modelo

| Modelo | MAE val. | MAE prueba | RMSE prueba | R² prueba | Registry |
|---|---:|---:|---:|---:|---|
| HistGradientBoosting (Poisson) | 0.8042 | **0.7651** | 1.3018 | 0.6937 | v1 · `champion` |
| RandomForest | 0.8255 | 0.7836 | 1.3510 | 0.6702 | v2 · `challenger` |
| PoissonRegressor | 1.2512 | 1.0983 | 3.1102 | −0.7481 | v3 |
| Dummy (media) | 1.6316 | 1.5374 | 2.3552 | −0.0024 | no se registra |

Fuente: `reports/model_comparison.csv`. Partición cronológica 70/15/15; el champion se elige por MAE de
validación, se reentrena con train + val y se evalúa una sola vez en prueba.

## Decisiones de modelado

| Decisión | Justificación |
|---|---|
| Rezagos de 24 h y 168 h y media de la misma hora en 7 días (se eliminó el de 1 h) | El horizonte es de 1 a 24 h: el rezago de 1 h no estaría disponible (fuga de información) |
| Partición cronológica 70/15/15; test usado una sola vez | Evita usar el futuro para predecir el pasado y el sesgo optimista |
| Pérdida de Poisson | Conteos no negativos con muchos ceros (53 % de las combinaciones estación-hora) |
| HistGradientBoosting como champion | Captura interacciones estación × hora; la regresión de Poisson lineal produce predicciones extremas (R² negativo) |
| TargetEncoder para la estación | 370 estaciones sin cientos de columnas one-hot |
| Sin PCA | En la Fase 1 empeoró el RMSE y con 13 variables no aporta |
| Desplegar el binario verificado (SHA-256), no reentrenar en cada entorno | El reentrenamiento entre sistemas operativos no es idéntico bit a bit |

---

## Reporte final

```bash
python scripts/build_architecture_diagram.py   # figuras 00 (arquitectura y ciclo MLOps)
python reports/build_report.py                 # reports/Entrega_Final_Equipo19.docx (o: make report)
```

El generador lee todas las cifras de los artefactos (CSV, JSON, logs, `git log`) y marca en amarillo lo pendiente.
Las reflexiones individuales se toman de `reports/fase_final/<nombre>.md`.

## Declaración de uso de Inteligencia Artificial

Ver `reports/declaracion_ia.md`. Resumen: se usó Anthropic. (2026). Claude [Modelo de lenguaje grande] (Opus 5.5 en
las Fases 1 y 2; Fable 5.2 y Opus 5.5 en la fase final) como apoyo en la redacción, la estructura del repositorio, el código y la documentación. El equipo
revisó, ejecutó y validó el código; los resultados corresponden a ejecuciones reales registradas en los logs y
artefactos del repositorio. https://claude.ai/
