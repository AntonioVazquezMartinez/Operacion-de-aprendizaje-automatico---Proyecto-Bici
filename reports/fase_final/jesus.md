# Fase final - Análisis y reflexión de Jesús Ángel Rincón Ruiz (A01793960)

> Plantilla: escribe solo donde dice `<!-- ESCRIBE AQUÍ -->` (borra el comentario al escribir). No necesitas
> ejecutar nada: todos los resultados ya están en el repositorio. La sección **Reflexión** se copia tal cual
> al reporte final cuando vuelve a generarse, así que redáctala en primera persona y en párrafos.

## Rol y responsabilidad en la fase final

**Rol:** Data Scientist.

En esta fase la ejecución técnica la hizo Gustavo (SRE). Tu responsabilidad como DS es el **análisis e
interpretación** de dos resultados:

1. **Resultados del modelo:** por qué gana el champion, qué significa su error para la operación de MiBici y qué
   tan estable es frente a la Fase 2.
2. **Simulación de data drift:** qué detecta cada escenario, si los umbrales y las acciones propuestas tienen sentido
   y qué dice el drift real de diciembre.

## Resultados que debes revisar (con su archivo de origen)

| Tema | Resultado | Dónde verlo |
|---|---|---|
| Comparación de modelos (prueba) | HistGradientBoosting Poisson: MAE 0.7651, RMSE 1.3018, R² 0.6937; RandomForest: MAE 0.7836; PoissonRegressor: MAE 1.0983, R² −0.7481; Dummy: MAE 1.5374 | `reports/model_comparison.csv`, `reports/figures/06_comparacion_modelos.png` |
| Fase 2 vs. reentrenamiento | MAE de prueba 0.7656 → 0.7651 (−0.06 %); mismos hiperparámetros del champion | `reports/fase_final/comparacion_fase2.md` |
| Error relativo | Demanda media en prueba 1.270; WAPE 0.6025; sesgo +4.6 % (sobrestima ligeramente) | `reports/drift/drift_report.md` (Referencia y línea base) |
| Escenarios de drift | Control OK; rezagos faltantes, rezagos ×1.5 y reloj desfasado → ALERTA y regla R1 (revisar pipeline); demanda ±40 % y vacaciones → R3 (reentrenar); estaciones nuevas → R2 (catálogo) | `reports/drift/drift_report.md`, `drift_summary.csv`, figuras 07-09 |
| MAE vs. WAPE | Con −40 % de demanda el MAE baja 13.6 % pero el WAPE sube 43.8 %; en vacaciones de diciembre el MAE baja 6.0 % y el WAPE sube 33.7 % | `drift_report.md` (Interpretación por escenario) |
| Drift real | Ventanas semanales: OK hasta el 13-dic; desde el 20-dic ALERTA con WAPE +38.8 % y sesgo +32.9 % | `drift_report.md` (Vista temporal), `reports/figures/10_drift_ventanas_tiempo.png` |
| Umbrales | PSI 0.10/0.25, KS 0.10/0.20, Wasserstein 0.10/0.25, MAE y WAPE +10 %/+25 % | `config.yaml` (`monitoring.thresholds`) |
| Reentrenar en Linux | El mismo pipeline en Linux elige otros hiperparámetros por diferencias de 1 ulp en `exp()`, con MAE de prueba 0.7653 vs 0.7651 | `reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md` |

## Preguntas guía

1. ¿Por qué el boosting supera a la regresión de Poisson lineal, que tiene R² negativo aunque mejora el MAE frente a
   la media? ¿Qué dice que RandomForest quede tan cerca del champion?
2. Un MAE de 0.77 bicicletas por hora con demanda media de 1.27: ¿para qué decisiones de rebalanceo sirve el modelo y
   para cuáles no?
3. ¿Por qué el MAE por sí solo engaña en los escenarios de caída de demanda y vacaciones? ¿Qué aporta el WAPE?
4. ¿Qué diferencia hay entre los escenarios de "corrupción de entradas" y de "cambio real", y por qué piden acciones
   distintas (corregir el pipeline vs. reentrenar)?
5. ¿Te parecen razonables los umbrales? ¿Cómo los calibrarías con el costo real de un error de rebalanceo?
6. ¿Qué variables agregarías (clima, calendario escolar, eventos) y cómo esperarías que cambiara diciembre?

## Actividades por fase

### Fase 1
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (ML Canvas, EDA, variables cíclicas y rezagos, MiBiciFeatureEngineer...). -->

### Fase 2
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (rediseño de variables sin rezago de 1 h, algoritmos, espacios de Optuna,
     migración a Cookiecutter v2 y config.yaml en el commit 93586e4...). -->

### Fase final
<!-- ESCRIBE AQUÍ: qué analizaste de los resultados del modelo y del drift. -->

## Análisis

<!-- ESCRIBE AQUÍ: 2 o 3 párrafos respondiendo las preguntas guía con datos de la tabla. -->

## Reflexión

<!-- ESCRIBE AQUÍ: un párrafo (8 a 12 líneas): lecciones aprendidas y áreas de mejora desde tu rol. -->
