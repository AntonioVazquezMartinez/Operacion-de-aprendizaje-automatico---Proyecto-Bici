# Fase final - Análisis y reflexión de Jesús Ángel Rincón Ruiz (A01793960)

Versión final. La sección **Reflexión** se copia tal cual al reporte (`python reports/build_report.py`).

## Rol y responsabilidad en la fase final

**Rol:** Data Scientist.

En esta fase el equipo trabajó dos aproximaciones. En la rama `entrega-final` (PR #5, commits `acf3df3` y `9110f58`)
está la que sirve el binario original de la Fase 2: servicio FastAPI con feature store, imagen construida con Podman y publicada en DockerHub
(`angrinconr/mibici-demand-api`), CI que construye la imagen, verificación en 3 entornos y simulación de drift. La otra aproximación (rama `fase-final`) reentrena en un entorno
limpio para verificar y complementar; se contrastaron y se integraron en `fase-final` (merge `14e4886`). Como DS
también soy responsable del **análisis e interpretación** de los resultados del modelo y del drift.

## Lo que se integró de tu rama (con su evidencia)

| Pieza de `entrega-final` | Cómo quedó en `fase-final` | Dónde verlo |
|---|---|---|
| Feature store (últimas 360 h por estación) y rezagos opcionales con `lags_source` | Servicio 1.1.0: sin rezagos, `/predict` los calcula con la misma función del entrenamiento; ventana predecible 2025-12-24 00:00 a 2026-01-01 23:00 | `src/serving/feature_store.py`, `docs/api.md` |
| Modelo dentro de Git | `models/champion/` en Git: `docker build` funciona en un clon limpio | `reproducibilidad/clon_limpio.json` |
| `PREDICTION_LOG` | Registro opcional de predicciones en JSONL | `logs/21_prediction_log_contenedor.log` |
| Prueba de coherencia servicio-pipeline | `tests/test_serving_feature_store.py` | `logs/10_pytest_cobertura.log` |
| CI que construye la imagen | Job `docker` en `.github/workflows/tests.yml` | pestaña Actions |
| Binario original de la Fase 2 | `models/fase2_original/`; reproduce sus 1,000 predicciones (dif. 0.0 en Windows, 8.9e-16 en Linux) | `reproducibilidad/fase2_binario_vs_reentrenado.md` |
| Comandos de Podman | Documentados junto a los de Docker | `docs/docker.md` |

## Resultados que debes revisar (con su archivo de origen)

| Tema | Resultado | Dónde verlo |
|---|---|---|
| Comparación de modelos (prueba) | HistGradientBoosting Poisson: MAE 0.7651, RMSE 1.3018, R² 0.6937; RandomForest: MAE 0.7836; PoissonRegressor: MAE 1.0983, R² −0.7481; Dummy: MAE 1.5374 | `reports/model_comparison.csv` |
| Fase 2 vs. reentrenamiento | MAE de prueba 0.7656 → 0.7651 (−0.06 %); mismos hiperparámetros | `reports/fase_final/comparacion_fase2.md` |
| Tu binario de la Fase 2 vs. el reentrenado | Mismos hiperparámetros; correlación 0.9989; diferencia media absoluta 0.037 viajes; MAE en 500 filas 0.7489 vs 0.7529 (datos del portal descargados en fechas distintas) | `reproducibilidad/fase2_binario_vs_reentrenado.md` |
| Error relativo | Demanda media en prueba 1.270; WAPE 0.6025; sesgo +4.6 % | `reports/drift/drift_report.md` |
| Escenarios de drift | Control OK; corrupción de entradas → R1 (revisar pipeline); cambios reales → R3 (reentrenar); estaciones nuevas → R2 | `reports/drift/drift_report.md`, figuras 07-09 |
| MAE vs. WAPE | Con −40 % de demanda el MAE baja 13.6 % pero el WAPE sube 43.8 %; en vacaciones de diciembre el MAE baja 6.0 % y el WAPE sube 33.7 % | `drift_report.md` |
| Drift real | Ventanas semanales: OK hasta el 13-dic; desde el 20-dic ALERTA con WAPE +38.8 % y sesgo +32.9 % | `reports/figures/10_drift_ventanas_tiempo.png` |

## Preguntas guía

1. ¿Por qué decidiste que el servicio calculara los rezagos con un feature store en vez de pedirlos al cliente? ¿Qué
   limitación tiene (ventana predecible fija, foto que caduca) y cómo la resolverías en operación?
2. ¿Por qué usaste Podman y Python 3.12? ¿Qué problemas encontraste (p. ej. el reenvío de puertos en modo rootless) y
   qué aprendiste al publicar la imagen?
3. ¿Qué aportó contrastar las dos aproximaciones (binario de la Fase 2 y reentrenamiento de verificación) y qué
   harías distinto para integrarlas con menos adaptación?
4. ¿Por qué el boosting supera a la regresión de Poisson lineal? ¿Qué significa que tu binario de la Fase 2 y el
   reentrenado tengan los mismos hiperparámetros pero predicciones ligeramente distintas?
5. ¿Por qué el MAE por sí solo engaña en la caída de demanda y en vacaciones? ¿Qué aporta el WAPE?
6. ¿Qué variables agregarías (clima, calendario escolar, eventos) y cómo esperarías que cambiara diciembre?

## Actividades por fase

### Fase 1
* Hice el ML Canvas y el EDA.
* Armé las variables cíclicas y los rezagos.

### Fase 2
* Rediseñé las variables y quité el rezago de 1 h porque era fuga de información; dejé rezagos de 24 h y 168 h, el promedio de la misma hora y los festivos.
* Definí los algoritmos y los espacios de Optuna.
* Integré la migración a Cookiecutter v2 y `config.yaml` (commit `93586e4`).

### Fase final
* En `entrega-final` hice el servicio FastAPI con feature store, la imagen con Podman publicada en DockerHub, el job de CI que construye la imagen, la verificación en 3 entornos y la simulación de drift.
* De ahí se integraron a `fase-final` el feature store, el modelo en Git, el registro de predicciones, la prueba de coherencia servicio-pipeline y el job `docker`.
* Analicé los resultados del modelo y del drift (tablas de la plantilla).

## Análisis

El servicio calcula los rezagos con un feature store (las últimas 360 h por estación) para que el cliente no tenga que mandarlos, y usa la misma función que el entrenamiento para que no haya diferencias. La limitación es que la ventana predecible es fija (del 24 de diciembre de 2025 al 1 de enero de 2026) y la foto caduca, así que en operación habría que refrescarla con los viajes nuevos. La imagen se construyó con Podman y se publicó en DockerHub (`angrinconr/mibici-demand-api`).

El boosting gana porque captura relaciones no lineales entre hora, estación y rezagos: HistGradientBoosting tiene MAE 0.7651 y R² 0.6937, y la regresión de Poisson lineal MAE 1.0983 y R² negativo (−0.7481). Mi binario de la Fase 2 y el reentrenado tienen los mismos hiperparámetros pero predicciones ligeramente distintas (correlación 0.9989, diferencia media de 0.037 viajes), porque los datos del portal se descargaron en fechas distintas.

El MAE solo engaña. Con −40 % de demanda el MAE baja 13.6 % y parece que el modelo mejora, pero el WAPE sube 43.8 %. En vacaciones de diciembre pasa algo parecido (MAE −6.0 %, WAPE +33.7 %). Por eso vigilamos el WAPE además del MAE. Con un WAPE de 60.3 % el modelo sirve para ordenar estaciones por riesgo, no para fijar el número exacto de bicicletas. Las ventanas semanales lo confirman: todo OK hasta el 13 de diciembre y alerta desde el 20 (WAPE +38.8 %, sesgo +32.9 %).

## Reflexión

Lo que más me quedó de esta fase es que un modelo con buen MAE puede fallar justo cuando la demanda cambia. En diciembre el MAE bajó y parecía que todo iba bien, pero el WAPE subió más de 30 %, y las ventanas semanales mostraron alerta real desde el 20 de diciembre. Por eso para mí el monitoreo tiene que mirar el error relativo y el sesgo, no solo el MAE. También aprendí que contrastar las dos aproximaciones del equipo valió la pena. Mi binario de la Fase 2 y el reentrenado dan casi lo mismo (MAE 0.7656 contra 0.7651), y de la otra solución tomamos el reentrenamiento en limpio, DVC y la verificación en GitHub Actions. Como mejora, me faltó integrar mi rama con menos adaptación, y el servicio depende de un feature store con ventana fija que habría que refrescar en operación. Agregaría clima, calendario escolar y eventos, porque diciembre es justo donde el modelo más falla.
