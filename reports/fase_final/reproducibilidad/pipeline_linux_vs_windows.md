# Reproducibilidad del pipeline: reentrenamiento en Linux (contenedor) vs Windows (host)

Mismo código, mismos CSV crudos (`data/raw/`, montados de solo lectura), misma configuración (`config.yaml`
con las rutas redirigidas a un volumen Docker), mismas semillas (`seed: 42`, `PYTHONHASHSEED=42`, TPE de
Optuna con semilla, `timeout_seconds: null`) y mismas versiones de librerías.

| | Windows (host, entrenamiento original) | Linux (contenedor limpio) |
|---|---|---|
| SO | Windows-11-10.0.26300-SP0 | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64, glibc 2.41 (Debian, `python:3.13.9-slim`) |
| Python / scikit-learn / numpy / pandas | 3.13.9 / 1.9.1 / 2.5.4 / 3.1.1 | 3.13.9 / 1.9.1 / 2.5.4 / 3.1.1 |
| Biblioteca matemática de C (exp, sin) | UCRT de Microsoft | glibc 2.41 |
| OpenMP | vcomp (MSVC), 20 hilos | libgomp, 20 hilos |
| Imagen | — | `docker/Dockerfile.pipeline` (`requirements.txt` completo) |

Comandos (ver `docker/run_pipeline_linux.sh`): `src.dataset` → `src.features` → `src.modeling.train` →
`src.modeling.predict` → `src.modeling.export`, con `--config /work/config_linux.yaml`. No se escribió nada en
`data/`, `mlruns.db`, `mlartifacts/`, `models/` ni `reports/` del host. No se corrió `src.plots` (figuras EDA).

| Paso | Windows (s) | Linux (s) |
|---|---|---|
| dataset | 88 | 80 |
| features | 42 | 40 |
| train (4 modelos, 65 trials de Optuna) | 194 | 215 |
| predict | 6 | 5 |
| export | 4 | 4 |

Fuentes: `reports/fase_final/logs/0*.log` (Windows) y `pipeline_linux/logs/*.log` (Linux).

## 1. Datos: idénticos salvo fin de línea y 1 ulp en `sin_hour`

| Archivo | ¿Bytes iguales? | Causa de la diferencia |
|---|---|---|
| `data/raw/nomenclatura_estaciones.csv` | No | Solo CRLF (Windows) vs LF (Linux): idénticos tras normalizar el fin de línea |
| `data/interim/viajes_cleaned.csv` (4.41 M filas) | No | Solo CRLF vs LF: idénticos tras normalizar |
| `models/champion/stations.csv` | No | Solo CRLF vs LF: idénticos tras normalizar |
| `data/processed/mibici_hourly_demand.csv` (3 179 040 × 18) | No | CRLF vs LF **y** `sin_hour` difiere en 1 ulp en 264 920 filas (2 de las 24 horas) |

`pandas.to_csv` usa `os.linesep` como fin de línea, por eso el SHA-256 de los CSV cambia entre sistemas
aunque el contenido sea el mismo (el `dataset_sha256` de los metadatos depende del SO). En la matriz procesada,
17 de las 18 columnas son idénticas; `sin_hour` toma dos valores distintos en una unidad del último lugar
(`-0.8660254037844385 → -0.8660254037844384` y `0.7071067811865476 → 0.7071067811865475`): `np.sin` delega en
la biblioteca matemática del sistema y UCRT y glibc redondean distinto esos dos casos.
Evidencia: `diagnostico_pipeline/data_cmp.out`, `diagnostico_pipeline/processed_cmp.out`.

## 2. Métricas por modelo (`pipeline_linux_vs_windows.csv`)

| Modelo | ¿Idéntico? | Dif. absoluta máxima entre las 8 métricas | Explicación |
|---|---|---|---|
| `dummy_mean` | Sí, bit a bit (8/8) | 0 | Sin funciones trascendentes |
| `random_forest` | Sí, bit a bit (8/8), mismos 5 trials | 0 | Divisiones por orden y medias; la diferencia de 1 ulp en `sin_hour` no cambia el orden de los valores |
| `poisson` (GLM) | No | 4.4e-10 (`val_r2`, rel. 3.1e-12); MAE ~3.7e-14 | lbfgs con `exp()`: diferencias de redondeo; los `alpha` de los trials 10, 13, 14, 16-18 difieren en el último dígito |
| `hist_gradient_boosting` (champion) | No | 9.5e-4 (`test_rmse`, rel. 7.3e-4) | Ver sección 3 |

Champion en ambos entornos: `hist_gradient_boosting` (mismo ranking de los 4 modelos), pero con otros
hiperparámetros:

| | Windows | Linux |
|---|---|---|
| Trial elegido | `trial_013` (val_mae 0.8144216495408687) | `trial_000` (val_mae 0.8144299124695795) |
| learning_rate / max_iter / max_leaf_nodes / min_samples_leaf | 0.054063 / 487 / 112 / 171 | 0.055147 / 481 / 97 / 128 |
| val_mae / test_mae | 0.804170096434065 / 0.7651310452008431 | 0.804681083633366 / 0.7652553680017535 |
| test_rmse / test_r2 | 1.3018304975138348 / 0.6937167354678235 | 1.3008810437358584 / 0.6941633308719144 |
| SHA-256 de `model.pkl` | `ca4331a4…2ed5` | `89839313…2a69` |

## 3. Por qué difiere el champion

1. **Búsqueda de Optuna.** Los trials 000-009 de TPE proponen exactamente los mismos hiperparámetros en ambos
   SO (son aleatorios con semilla 42), pero sus `val_mae` difieren en ~1e-5 a 1e-3 (p. ej. trial 000:
   0.8144783817186876 en Windows vs 0.8144299124695795 en Linux; los trials 001 y 003 sí salen idénticos).
   A partir del trial 010 TPE condiciona en esos valores y las propuestas divergen (primero en el último dígito,
   desde el trial 013 por completo). En Linux el mejor trial resultó el 000; en Windows, el 013, que Linux nunca
   propuso. Evidencia: `diagnostico_pipeline/trials_cmp.out`.
2. **Entrenamiento del boosting.** La pérdida Poisson usa `exp()` en cada iteración para gradientes y
   hessianos; un ulp de diferencia (UCRT vs glibc) cambia la suma de un histograma, eventualmente un empate de
   ganancia entre dos divisiones, y el árbol siguiente parte de otra predicción: el efecto se amplifica a lo largo
   de cientos de iteraciones.
3. **Control con hiperparámetros fijos.** Reentrenando en Linux con los hiperparámetros EXACTOS del champion de
   Windows (sin Optuna): el modelo entrenado solo con train reproduce `val_mae` = 0.804170096434065 idéntico
   (val_rmse difiere 2.2e-16); el modelo final (train + val) da `test_mae` 0.7651355724486015 (dif. 4.5e-6) y
   `test_rmse` dif. 1.8e-4. Repetido dos veces en Linux, las dos corridas son idénticas entre sí (el entrenamiento
   es determinista dentro de cada SO). Evidencia: `diagnostico_pipeline/fixed_params_linux_run{1,2}.json`.

## 4. Predicciones sobre las 500 filas de referencia (`models/reference_predictions.csv`)

| Modelo evaluado en Linux | Dif. máx. vs `demand_pred` | Dif. media | Filas idénticas bit a bit |
|---|---|---|---|
| **Champion de Windows** (mismo `model.pkl`, servido en el contenedor) | 8.881784197001252e-16 | 2.44e-18 | 495/500 |
| Champion reentrenado en Linux (Optuna completo) | 0.98544059733395 | 0.0380 | 0/500 |
| Reentrenado en Linux con hiperparámetros de Windows | 0.29462899042747726 | 0.0217 | 0/500 |

Fuentes: `pipeline_linux/comparison/comparacion_pipeline.json` y `diagnostico_pipeline/fixed_params_linux_run1.json`.
La muestra de referencia que exporta Linux tiene las mismas 500 filas (mismo muestreo con semilla 42) pero no es
igual columna a columna: solo `sin_hour` difiere (1 ulp, sección 1).

## 5. Conclusión

- **El mismo binario sí es reproducible entre entornos.** El `model.pkl` entrenado en Windows produce en Linux las
  mismas predicciones salvo 5 de 500 filas que difieren en exactamente 1 ulp (máx. 8.9e-16), con MAE/RMSE de la
  muestra idénticos. La suma de los árboles (escala log) es idéntica bit a bit; la diferencia está solo en el
  `exp()` final de la función de enlace (glibc redondea correctamente 4 de esos 5 casos, UCRT 1). Ver
  `diagnostico_exp/resumen.txt`. Por eso la tolerancia por omisión de `scripts/verify_reproducibility.py` es
  `1e-9`: ~6 órdenes de magnitud por encima de lo observado y muy por debajo de cualquier diferencia con
  significado (las predicciones son viajes por hora; el MAE es ~0.75).
- **El reentrenamiento es reproducible dentro de un mismo SO, pero no bit a bit entre SO.** Datos y modelos sin
  funciones trascendentes (dummy, random forest) salen idénticos; los que usan `exp()` difieren en el redondeo, y
  en el boosting con búsqueda de Optuna eso basta para elegir otros hiperparámetros. Las métricas del champion
  difieren como máximo 9.5e-4 (0.07 % relativo) y la conclusión del experimento (ranking de modelos, algoritmo
  champion, nivel de error) es la misma.
- Por eso el artefacto que se despliega y versiona es el **binario** (`model.pkl` con SHA-256 verificado al
  arrancar), no el resultado de reentrenar: la imagen Docker lleva ese binario y su referencia de predicciones.
- Pendiente para el equipo de pruebas: `tests/integration/test_versioned_artifacts.py::test_local_champion_reproduces_reference`
  y `tests/test_serving.py::test_modelo_real_reproduce_la_referencia` exigen igualdad exacta y fallan en Linux por
  ese ulp (`pipeline_linux_pytest.log`: 268 passed, 2 failed). En GitHub Actions se omiten porque `models/champion/`
  no está en Git, pero fallarían en cualquier Linux con el modelo (p. ej. tras `dvc pull`).

> **Nota (actualización):** las dos pruebas se cambiaron a tolerancia `1e-9`. Con el código actual y el modelo real,
> la suite completa en el contenedor Linux da `270 passed` (`pipeline_linux_pytest_v2.log`).
