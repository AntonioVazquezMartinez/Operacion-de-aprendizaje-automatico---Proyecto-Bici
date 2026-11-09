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
├── .github/workflows/     <- Integración continua: corre las pruebas en cada pull request y en push a main
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
├── models/                <- (Los modelos se versionan en el MLflow Model Registry)
├── notebooks/
│   └── 1.0-eq19-eda-mibici.ipynb   <- Exploración; importa los módulos de src/
├── references/
│   └── diccionario_datos.md
├── reports/
│   ├── figures/           <- Figuras del EDA y comparación de modelos
│   ├── fase2/             <- Rol y reflexión de cada integrante en la Fase 2
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
│   └── modeling/
│       ├── pipeline.py    <- Pipeline de scikit-learn (preprocesamiento + modelo) y métricas
│       ├── train.py       <- Optuna + MLflow Tracking + Model Registry
│       └── predict.py     <- Predicción por lotes con el modelo `champion`
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

## Declaración de uso de Inteligencia Artificial

Ver `reports/declaracion_ia.md`. Resumen:

- **Fase 1:** Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como herramienta de consulta y de apoyo en la revisión de la redacción, la organización, el formato del reporte y la rúbrica.
- **Fase 2:** Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como apoyo en la reestructuración del repositorio a Cookiecutter Data Science v2, la refactorización a módulos configurables, la generación del código de entrenamiento con Optuna, MLflow Tracking y Model Registry, las pruebas automatizadas y la documentación. El equipo revisó, ejecutó y validó el código; los resultados reportados corresponden a la ejecución real del pipeline sobre los datos abiertos de MiBici, registrada en MLflow. https://claude.ai/
