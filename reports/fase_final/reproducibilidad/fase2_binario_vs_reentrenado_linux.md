# Binario original de la Fase 2 contra el champion reentrenado

Generado con `python scripts/compare_fase2_binary.py` (2026-11-22T20:22:56+00:00). Datos completos en `fase2_binario_vs_reentrenado_linux.json`.

Ejecutado dentro del contenedor `ml-service:latest` (imagen 1.1.0, Linux) con `docker run --rm -v <repo>/models:/w/models:ro -v <repo>/scripts:/w/scripts:ro -v <carpeta>:/out --entrypoint python ml-service:latest /w/scripts/compare_fase2_binary.py --output-dir /out`. La misma comparación en Windows está en `fase2_binario_vs_reentrenado.md`.

`models/fase2_original/` es una copia sin modificar del paquete exportado en la rama `entrega-final` (Jesús Rincón): `models:/mibici-demand-forecaster/1`, corrida `b230b27936e2439db0c9eae68986d884`, exportado con Python 3.12.10 en Windows-11-10.0.26200-SP0 (scikit-learn 1.9.1).

## 1. Integridad

- SHA-256 de `model.joblib`: `ee03d0da0d1d0c83d00a44bc35a01fc624412591448442665a68364eb43a6f51` (ficha: coincide = True).
- SHA-256 del CSV de referencia: coincide con la ficha = True (Git lo guarda con LF; la ficha tiene el hash de la versión CRLF escrita en Windows).

## 2. El mismo binario en otro entorno

Entorno de esta verificación: Python 3.13.9, Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.41, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1.

| Filas | Diferencia máxima | Diferencia media | Idénticas bit a bit | Tolerancia 1e-9 |
|---|---|---|---|---|
| 1000 | 8.881784197001252e-16 | 1.8873791418627663e-18 | 997/1000 | PASS |

## 3. Binario de la Fase 2 contra el champion reentrenado (500 filas de `models/reference_predictions.csv`)

Mismos hiperparámetros: True. Los dos modelos se entrenaron con el mismo código y semilla, pero con datos del portal descargados en fechas distintas (ver `reports/fase_final/comparacion_fase2.md`), así que no son el mismo binario.

| Medida | Valor |
|---|---|
| Diferencia máxima entre predicciones | 0.5983954109243568 |
| Diferencia media absoluta | 0.03680256681793775 |
| Diferencia media (Fase 2 - reentrenado) | -0.009497567959190325 |
| Filas con diferencia > 0.1 viajes | 45 de 500 |
| Correlación de Pearson | 0.9989334445712195 |
| MAE contra la demanda real: binario Fase 2 | 0.7488708936743764 |
| MAE contra la demanda real: champion reentrenado | 0.752896241060654 |
| RMSE: binario Fase 2 / reentrenado | 1.1721359130807572 / 1.1785947698217105 |

El MAE de esta tabla es sobre la muestra de 500 filas, no sobre el conjunto de prueba completo (MAE de prueba registrado: Fase 2 0.7656, reentrenado 0.7651310452008431).
