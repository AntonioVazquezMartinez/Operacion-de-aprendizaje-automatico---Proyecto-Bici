# Fase final - Reentrenamiento en entorno limpio vs. Fase 2

Fecha de ejecución: 22 de noviembre de 2026. Entorno: Windows 11 Pro (10.0.26300), Python 3.13.9 en un
venv nuevo (`.venv`) instalado solo desde `requirements.txt` con versiones exactas: scikit-learn 1.9.1,
MLflow 3.18.0, Optuna 5.0.0, pandas 3.1.1, NumPy 2.5.4.

Fuentes de cada cifra:

| Dato | Artefacto |
|---|---|
| Métricas de la Fase 2 | `reports/fase_final/model_comparison_fase2_referencia.csv` (copia de `reports/model_comparison.csv` antes de reentrenar) |
| Métricas del reentrenamiento | `reports/model_comparison.csv` y `reports/fase_final/logs/04_train.log` |
| Conteos de limpieza y matriz | `reports/fase_final/logs/01_dataset.log`, `02_features.log` y corridas del experimento `MiBici_Datos` en MLflow |
| Conteos de la Fase 2 (por regla, partición) | `reports/Fase2_Equipo19.docx` (tablas de limpieza y partición) |
| Modelo exportado | `models/champion_metadata.json` |

## 1. Métricas por modelo

Diferencia = reentrenamiento - Fase 2; el porcentaje es relativo al valor de la Fase 2.

| Modelo | Métrica | Fase 2 | Reentrenamiento | Diferencia | Diferencia % |
|---|---|---:|---:|---:|---:|
| hist_gradient_boosting | val_mae | 0.8043 | 0.8042 | -0.0001 | -0.02 % |
| hist_gradient_boosting | val_rmse | 1.5358 | 1.5359 | +0.0001 | +0.00 % |
| hist_gradient_boosting | val_r2 | 0.6734 | 0.6734 | -0.0000 | -0.00 % |
| hist_gradient_boosting | **test_mae** | **0.7656** | **0.7651** | **-0.0005** | **-0.06 %** |
| hist_gradient_boosting | test_rmse | 1.3042 | 1.3018 | -0.0024 | -0.18 % |
| hist_gradient_boosting | test_r2 | 0.6926 | 0.6937 | +0.0011 | +0.16 % |
| random_forest | val_mae | 0.8255 | 0.8255 | -0.0000 | -0.00 % |
| random_forest | val_rmse | 1.5904 | 1.5900 | -0.0004 | -0.03 % |
| random_forest | val_r2 | 0.6498 | 0.6500 | +0.0002 | +0.03 % |
| random_forest | test_mae | 0.7833 | 0.7836 | +0.0002 | +0.03 % |
| random_forest | test_rmse | 1.3505 | 1.3510 | +0.0005 | +0.04 % |
| random_forest | test_r2 | 0.6704 | 0.6702 | -0.0002 | -0.03 % |
| poisson | val_mae | 1.2491 | 1.2512 | +0.0021 | +0.17 % |
| poisson | val_rmse | 31.0651 | 32.1485 | +1.0834 | +3.49 % |
| poisson | val_r2 | -132.6147 | -142.0968 | -9.4821 | -7.15 % |
| poisson | test_mae | 1.0977 | 1.0983 | +0.0005 | +0.05 % |
| poisson | test_rmse | 3.1147 | 3.1102 | -0.0045 | -0.15 % |
| poisson | test_r2 | -0.7532 | -0.7481 | +0.0051 | +0.68 % |
| dummy_mean | val_mae | 1.6295 | 1.6316 | +0.0021 | +0.13 % |
| dummy_mean | val_rmse | 2.6887 | 2.6886 | -0.0002 | -0.01 % |
| dummy_mean | val_r2 | -0.0009 | -0.0008 | +0.0001 | +12.37 % |
| dummy_mean | test_mae | 1.5355 | 1.5374 | +0.0020 | +0.13 % |
| dummy_mean | test_rmse | 2.3550 | 2.3552 | +0.0002 | +0.01 % |
| dummy_mean | test_r2 | -0.0023 | -0.0024 | -0.0002 | -7.88 % |

Los porcentajes grandes del `dummy_mean` en R² vienen de dividir entre valores cercanos a cero
(-0.0009 y -0.0023); en términos absolutos la diferencia es de 0.0001-0.0002.

**Criterio de aceptación: se cumple.** El champion es otra vez HistGradientBoosting con pérdida de Poisson
(menor MAE de validación, 0.8042), registrado como `models:/mibici-demand-forecaster/1` con alias
`champion`; su MAE de prueba es 0.7651 contra 0.7656 de la Fase 2 (-0.06 %, dentro de ±2 %). El orden de
los cuatro modelos y la asignación de alias no cambiaron: RandomForest es `challenger` (v2) y
PoissonRegressor queda como v3. `python -m src.modeling.predict` reproduce desde el Registry
MAE 0.7651 / RMSE 1.3018 / R² 0.6937 sobre las 476,930 filas de prueba (`logs/05_predict.log`).

Los hiperparámetros del champion seleccionados por Optuna coinciden con los reportados en la Fase 2
(487 iteraciones, tasa de aprendizaje 0.054, al menos 171 observaciones por hoja; detalle completo en
`models/champion_metadata.json`).

## 2. Datos: retención de limpieza y tamaño de la matriz

| Concepto | Fase 2 | Reentrenamiento | Diferencia |
|---|---:|---:|---:|
| Registros brutos 2025 | 4,520,095 | 4,532,032 | +11,937 (+0.26 %) |
| Duplicados de Viaje_Id | 0 | 0 | 0 |
| Duración negativa o cero | 2,773 | 2,780 | +7 |
| Viajes menores a 60 s | 114,180 | 114,446 | +266 |
| Viajes mayores a 120 min | 884 | 885 | +1 |
| Estación inexistente en el catálogo | 946 | 945 | -1 |
| Edad fuera de 14-90 años | 512 | 513 | +1 |
| Registros limpios | 4,400,800 | 4,412,463 | +11,663 (+0.27 %) |
| Retención | 97.36 % | 97.36 % | 0 |
| Estaciones en el catálogo | 470 | 470 | 0 |
| Matriz estación-hora (filas x columnas) | 3,179,040 filas | 3,179,040 x 18 | 0 filas |
| Estaciones activas x horas | 370 x 8,592 | 370 x 8,592 | 0 |
| Filas train / val / test | 2,225,180 / 476,930 / 476,930 | 2,225,180 / 476,930 / 476,930 | 0 |
| Inicio de prueba | 8-nov-2025 07:00 | 8-nov-2025 07:00 | igual |
| Demanda media val / test | 1.449 / 1.270 | 1.449 / 1.270 | igual a 3 decimales |

## 3. Explicación de las diferencias

1. **Los CSV del portal no son idénticos a los de la Fase 2.** Con los mismos 12 archivos mensuales
   (nombres iguales) se leen 11,937 viajes más. El conteo actual coincide exactamente con las líneas de
   los archivos (4,532,044 líneas en total menos 12 encabezados, según `wc -l`), es decir, la lectura no descartó
   ninguna fila. Hay evidencia de que el portal republicó al menos julio: en la Fase 2
   `datos_abiertos_2025_07.csv` tenía comillas mal cerradas y requería la lectura de respaldo
   (`QUOTE_NONE`); hoy el archivo trae todos los campos entre comillas y se lee con el lector estándar
   (el log de `src.dataset` ya no muestra el aviso de comillas). No hay conteos por mes de la Fase 2 en el
   repositorio, así que no se puede atribuir la diferencia a meses concretos. El efecto en el objetivo
   es pequeño pero visible: el `dummy_mean` (que solo depende de la media de entrenamiento y de los
   valores reales) cambia 0.13 % su MAE, lo que confirma que la diferencia viene de los datos y no del
   modelo.
2. **Catálogo de estaciones nuevo (`nomenclatura_2026_10.csv` en lugar de `nomenclatura_2026_09.csv`,
   que ya da 404).** Tiene las mismas 470 estaciones y el filtro de estaciones descarta 945 viajes contra
   946 de la Fase 2; la matriz conserva exactamente las mismas 370 estaciones activas. Su efecto es
   despreciable. No es posible comparar coordenadas o capacidades contra el catálogo anterior porque ya
   no está publicado ni versionado en el repositorio.
3. **Corte por tiempo de Optuna.** En la Fase 2 `tuning.timeout_seconds: 900` podía cortar la búsqueda
   según la velocidad de la máquina. Ahora es `null` y en MLflow se registra
   `tuning_trials_completed`: se completaron 5/5 trials en random_forest y 20/20 en poisson y
   hist_gradient_boosting (`logs/04_train.log`). En esta máquina todo el entrenamiento tardó 194 s, muy por
   debajo del límite anterior, y los hiperparámetros del champion coinciden con los de la Fase 2, lo que
   indica que el timeout tampoco recortó la búsqueda del boosting en la Fase 2. No se puede confirmar para
   los otros modelos porque las corridas de MLflow de la Fase 2 viven en la máquina de quien entrenó.
4. **Versiones de librerías.** El reentrenamiento usa las versiones fijadas en `requirements.txt` (las
   mismas que la verificación de `reports/fase2/gustavo.md`). Las versiones exactas de la máquina que
   entrenó la Fase 2 no quedaron registradas en el repositorio (el reporte menciona Python 3.12 para las
   pruebas), por lo que no se puede aislar su efecto; dado que el champion recupera los mismos
   hiperparámetros y su MAE cambia solo -0.06 %, cualquier efecto de versión es menor que el de los datos.

La mayor diferencia relativa es el RMSE de validación de la regresión de Poisson (+3.49 %). Ese RMSE está
dominado por unas cuantas predicciones extremas del enlace logarítmico (R² de -142), así que pequeños
cambios en los datos o en el `alpha` elegido lo mueven mucho; no afecta la selección del champion.

## 4. Reproducibilidad del modelo exportado

`python -m src.modeling.export` copió la versión 1 del Registry a `models/champion/` (SHA-256 del
`model.pkl`: `ca4331a49948c99b29580463d89e8aa1dc8d14451f866c029588f3de67ce2ed5`; SHA-256 del dataset
procesado: `fec850d2bdbc935a16cc3e439c1debe824647c801aea4652710d59d6a1a597dc`, igual al registrado en el
modelo) y generó `models/reference_predictions.csv` con 500 filas de prueba (semilla 42).

| Comprobación | Diferencia máxima absoluta | Log |
|---|---:|---|
| Proceso nuevo, venv de entrenamiento (`--verify-only`) | 0.000e+00 | `logs/07_verify_reference.log` |
| Venv aparte solo con `requirements-serve.txt`, copia de `models/` fuera del repositorio y sin `mlruns.db` | 0.000e+00 | `logs/08_verify_standalone_serve_env.log` |
