# Fase final - Análisis y reflexión de Jesús Ángel Rincón Ruiz (A01793960)

> Plantilla: escribe solo donde dice `<!-- ESCRIBE AQUÍ -->` (borra el comentario al escribir). No necesitas
> ejecutar nada: todos los resultados ya están en el repositorio. La sección **Reflexión** se copia tal cual
> al reporte final cuando vuelve a generarse, así que redáctala en primera persona y en párrafos.

## Rol y responsabilidad en la fase final

**Rol:** Data Scientist.

En esta fase implementaste en la rama `entrega-final` (PR #5, commits `acf3df3` y `9110f58`) una solución completa de
la entrega: servicio FastAPI con feature store, imagen construida con Podman y publicada en DockerHub
(`angrinconr/mibici-demand-api`), CI que construye la imagen, verificación en 3 entornos y simulación de drift. Gustavo
implementó otra en paralelo en `fase-final`; se compararon y se fusionaron en `fase-final` (merge `14e4886`). Como DS
también eres responsable del **análisis e interpretación** de los resultados del modelo y del drift.

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
3. ¿Qué aprendiste al comparar tu rama con `fase-final` y fusionarlas? ¿Qué harías distinto para coordinarse desde el
   inicio?
4. ¿Por qué el boosting supera a la regresión de Poisson lineal? ¿Qué significa que tu binario de la Fase 2 y el
   reentrenado tengan los mismos hiperparámetros pero predicciones ligeramente distintas?
5. ¿Por qué el MAE por sí solo engaña en la caída de demanda y en vacaciones? ¿Qué aporta el WAPE?
6. ¿Qué variables agregarías (clima, calendario escolar, eventos) y cómo esperarías que cambiara diciembre?

## Actividades por fase

### Fase 1
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (ML Canvas, EDA, variables cíclicas y rezagos, MiBiciFeatureEngineer...). -->

### Fase 2
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (rediseño de variables sin rezago de 1 h, algoritmos, espacios de Optuna,
     migración a Cookiecutter v2 y config.yaml en el commit 93586e4...). -->

### Fase final
<!-- ESCRIBE AQUÍ: tu implementación en entrega-final (servicio, feature store, Podman/DockerHub, CI, verificación,
     drift) y qué analizaste de los resultados. -->

## Análisis

<!-- ESCRIBE AQUÍ: 2 o 3 párrafos respondiendo las preguntas guía con datos de las tablas. -->

## Reflexión

<!-- ESCRIBE AQUÍ: un párrafo (8 a 12 líneas): lecciones aprendidas y áreas de mejora desde tu rol. -->
