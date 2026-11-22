# Roles MLOps e interacción del equipo (Fases 1, 2 y final) - Equipo 19

Desde la Fase 1 adoptamos los cinco roles de MLOps con cuatro integrantes: José Antonio Vázquez Martínez cubre
Data Engineer y ML Engineer. Este archivo resume qué hizo cada rol en cada fase, cómo se alinea con el ciclo de
vida MLOps y la matriz RACI de la fase final. Las actividades de las Fases 1 y 2 provienen de los reportes de cada
fase, de `reports/fase2/*.md` y del historial de Git.

## 1. Integrantes y roles

| Rol | Integrante | Etapas del ciclo de vida que le corresponden |
|---|---|---|
| Data Engineer (DE) | José Antonio Vázquez Martínez (A01797208) | Datos: ingesta, limpieza, versionado |
| Data Scientist (DS) | Jesús Ángel Rincón Ruiz (A01793960) | Experimentación (variables, modelos, métricas) y monitoreo (interpretación del drift) |
| Software Engineer (SWE) | Erick Francisco Siller Ojeda (A01382929) | Empaquetado (estructura, pruebas, CI) y despliegue (API) |
| Site Reliability Engineer (SRE) | Gustavo Alejandro Vázquez Rivas (A01611559) | Entorno reproducible, despliegue (contenedor) y monitoreo operativo |
| ML Engineer (MLE) | José Antonio Vázquez Martínez (A01797208) | Tracking, Model Registry, exportación y retroalimentación (reentrenamiento) |

## 2. Actividades por rol y fase

| Rol | Fase 1 (25-oct-2026) | Fase 2 (8-nov-2026) | Fase final |
|---|---|---|---|
| DE (Antonio) | Descarga y limpieza (`downloader.py`, `cleaner.py`); versiones raw e interim en MLflow | Ingesta de los 12 meses de 2025, codificaciones y registros mal formados; revisión de la limpieza (97.36 % de retención) | Responsable del análisis y revisión de la etapa de datos: cambios del portal, retención, hashes y DVC (`reports/fase_final/antonio.md`) |
| DS (Jesús) | ML Canvas, EDA, variables cíclicas y de rezago | Rediseño de variables (sin rezago de 1 h), algoritmos y espacios de Optuna; integró la migración a Cookiecutter v2 y `config.yaml` (commit `93586e4`) | Responsable del análisis de resultados del modelo y de la simulación de drift (`reports/fase_final/jesus.md`) |
| SWE (Erick) | Estructura Cookiecutter, refactorización a POO, suite `unittest`, Makefile, pyproject y CI | Revisión de la migración, corrección de la prueba de festivos, pruebas de `read_month`, integración de `main` y CI (`reports/fase2/erick.md`) | Responsable del análisis y revisión de la calidad de software: pruebas, cobertura, API y CI (`reports/fase_final/erick.md`) |
| SRE (Gustavo) | `requirements.txt`, `environment.yml`, MLflow con SQLite; descarga de datos reales y baselines en MLflow | Verificación en un clon limpio; hallazgos de dependencias sin versión fija (`reports/fase2/gustavo.md`) | Ejecución técnica completa: entorno fijado, reentrenamiento, exportación, pytest, API, Docker/DockerHub, verificación entre entornos, DVC y drift (`reports/fase_final/gustavo.md`) |
| MLE (Antonio) | `MLflowDataVersioner` con SHA-256; flujo de punta a punta | Pipeline con MLflow Tracking, Model Registry (champion/challenger) y predicción por lotes | Responsable del análisis del ciclo de vida del modelo: Registry → exportación → servicio; merge del PR final |

**Acuerdo de la fase final.** Para trabajar en una sola máquina con los datos, la base de MLflow y Docker (que no
están en GitHub), Gustavo ejecutó toda la parte técnica. Cada uno de los demás integrantes es responsable del
análisis y la revisión de los entregables que corresponden a su rol, y lo documenta con un commit de su archivo en
`reports/fase_final/`. Cada integrante ajusta su fila si lo considera necesario.

## 3. Alineación con el ciclo de vida MLOps

| Etapa | Roles (A = rinde cuentas) | Entregables | Fase en que se construyó |
|---|---|---|---|
| Datos | DE (A), DS | `src/dataset.py`, `src/features.py`, hashes en `MiBici_Datos`, `dvc.yaml` / `dvc.lock` | F1 → F2 → final |
| Experimentación | DS (A), MLE | `config.yaml` (modelos y espacios), `src/modeling/train.py`, Optuna | F1 → F2 |
| Empaquetado | MLE (A), SWE | Model Registry, `src/modeling/export.py`, `tests/`, `.github/workflows/tests.yml` | F2 → final |
| Despliegue | SRE (A), SWE | `src/serving/`, `Dockerfile`, imagen en DockerHub | Final |
| Monitoreo | DS (A), SRE | `src/monitoring/`, `reports/drift/` | Final |
| Retroalimentación | MLE (A), DE, SRE | Reglas R1-R7: corregir el pipeline o reentrenar y mover el alias | Final (propuesta) |

Diagrama: `reports/figures/00b_ciclo_mlops.png`.

## 4. Matrices RACI

R = responsable de ejecutar, A = rinde cuentas y revisa, C = consultado, I = informado.

### Fase 1

| Actividad | DE | DS | SWE | SRE | MLE |
|---|:---:|:---:|:---:|:---:|:---:|
| Estructura Cookiecutter | I | I | A/R | C | C |
| ML Canvas | C | A/R | I | I | C |
| Ingesta y limpieza | A/R | C | C | I | C |
| EDA | C | A/R | I | I | C |
| Ingeniería de características | C | R | C | I | A |
| MLflow y entorno | C | I | I | R | A |
| Versionado de datos | R | I | I | C | A |
| Pruebas unitarias | C | C | A/R | I | C |
| Reporte | C | C | C | C | A/R |

### Fase 2

| Actividad | DE | DS | SWE | SRE | MLE |
|---|:---:|:---:|:---:|:---:|:---:|
| Datos 2025 y lectura robusta | A/R | C | C | I | C |
| Variables y selección de algoritmos | C | A/R | I | I | C |
| Cookiecutter v2 y `config.yaml` | I | R | A | C | C |
| Pipeline, Optuna, Tracking y Registry | C | C | I | I | A/R |
| Pruebas y CI | I | C | A/R | C | C |
| Verificación en clon limpio | I | I | C | A/R | C |

### Fase final

| Actividad | DE | DS | SWE | SRE | MLE |
|---|:---:|:---:|:---:|:---:|:---:|
| Datos, hashes y DVC | A | C | I | R | C |
| Reentrenamiento y comparación con Fase 2 | C | C | I | R | A |
| Exportación del champion | I | I | C | R | A |
| Pruebas pytest y cobertura | I | I | A | R | C |
| API FastAPI | I | C | A | R | C |
| Docker, DockerHub y reproducibilidad | I | I | C | A/R | C |
| Simulación de drift y umbrales | C | A | I | R | C |
| Análisis por rol (archivo `.md`) | R | R | R | R | R |
| Reporte final y PDF | C | C | C | R | A |

## 5. Flujo de colaboración

1. Una rama por fase (`fase1-datos-reales`, `fase2-modelado`, `fase-final`) y pull requests hacia `main`
   (#1 a #4 en las fases anteriores).
2. Cada push y cada PR ejecutan las pruebas en GitHub Actions (`tests.yml`); `docker-reproducibilidad.yml` revisa
   el Dockerfile y verifica la imagen publicada contra la referencia.
3. En la fase final cada integrante escribe en su propio archivo (`reports/fase_final/<nombre>.md`), sin conflictos,
   y lo integra con un commit en `fase-final`; José Antonio hace el merge del PR final.
4. El reporte se genera por código (`reports/build_report.py`), que toma las cifras de los artefactos y las
   reflexiones de los archivos de cada integrante.
