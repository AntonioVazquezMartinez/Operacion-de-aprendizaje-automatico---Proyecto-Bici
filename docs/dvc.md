# Versionado de datos y modelo con DVC

El proyecto versiona sus artefactos con dos herramientas complementarias:

| Herramienta | Qué versiona | Dónde queda |
|---|---|---|
| **MLflow Tracking** | Cada versión de los datos (raw, interim, processed) con su SHA-256, y cada corrida de entrenamiento con parámetros, métricas y figuras | Experimentos `MiBici_Datos`, `MiBici_Fase2_Modelos` y `MiBici_Monitoreo` |
| **MLflow Model Registry** | Los modelos entrenados, con versión y alias | `models:/mibici-demand-forecaster/1` (alias `champion`) |
| **DVC** | El pipeline completo: comando, código, parámetros, datos y modelo de cada etapa, con su hash | `dvc.yaml` y `dvc.lock` (en Git) |
| **Git** | Código, configuración, metadatos del modelo y predicciones de referencia | Repositorio |

## Pipeline (`dvc.yaml`)

```text
data ──> features ──> train ──> export ──> drift
```

| Etapa | Comando | Salidas |
|---|---|---|
| `data` | `python -m src.dataset` | `data/raw/viajes_raw.csv`, `data/raw/nomenclatura_estaciones.csv`, `data/interim/viajes_cleaned.csv` |
| `features` | `python -m src.features` | `data/processed/mibici_hourly_demand.csv` |
| `train` | `python -m src.modeling.train` | `reports/model_comparison.csv`, figura 06 (los modelos van al Model Registry) |
| `export` | `python -m src.modeling.export` | `models/champion/`, `models/champion_metadata.json`, `models/reference_predictions.csv` |
| `drift` | `python -m src.monitoring.drift` | `reports/drift/`, figuras 07 a 10 |

Cada etapa declara como dependencias su código, sus datos de entrada y las secciones de `config.yaml` que usa
(`seed`, `data`, `cleaning`, `features`, `split`, `tuning`, `models`, `serving`, `monitoring`). Si cambia un
parámetro, un archivo de datos o un módulo, DVC lo detecta y solo vuelve a ejecutar lo necesario.

## `dvc.lock`: la huella del proyecto

`dvc.lock` guarda el hash MD5 y el tamaño de cada dependencia y de cada salida, incluidos los 12 CSV mensuales
del portal de MiBici y la carpeta del modelo. Como está en Git, cada commit fija exactamente con qué datos,
código y parámetros se obtuvo el modelo publicado.

## Comandos

```bash
dvc status     # ¿los artefactos locales corresponden a dvc.lock?
dvc dag        # grafo de etapas
dvc repro      # vuelve a ejecutar solo las etapas afectadas por un cambio
```

Para reproducir desde un clon limpio:

```bash
pip install -r requirements.txt && pip install -e .
dvc repro      # descarga los CSV del portal (URL en config.yaml) y ejecuta las cinco etapas
```

## Notas

- **Sin almacenamiento remoto.** Los archivos pesados (unos 390 MB de CSV crudos y los datos intermedios) no se
  suben a un remoto de DVC: los datos son públicos y se descargan del portal con las URL de `config.yaml`; el
  hash de `dvc.lock` permite comprobar que son los mismos. El modelo entrenado se distribuye dentro de la imagen
  publicada en DockerHub (`gusrivasmna/mibici-demand-service:1.0.0`). Agregar un remoto (`dvc remote add`)
  permitiría recuperar los artefactos con `dvc pull` sin volver a ejecutar; queda como trabajo futuro.
- **Los datos del portal pueden cambiar.** Entre la Fase 2 y la fase final el portal publicó 11,937 viajes más
  para 2025 (ver `reports/fase_final/comparacion_fase2.md`). Si un CSV cambia, `dvc status` lo reporta como
  dependencia modificada.
- **`train` registra versiones nuevas.** Volver a ejecutar la etapa `train` crea versiones nuevas en el Model
  Registry y mueve el alias `champion`; la etapa `export` toma siempre el alias vigente.
- **Windows y rutas largas.** La caché de DVC usa rutas de más de 260 caracteres. En Windows sin rutas largas
  habilitadas conviene mover la caché a una ruta corta, solo en la máquina local:
  `dvc cache dir --local C:/dvc/mibici`.
