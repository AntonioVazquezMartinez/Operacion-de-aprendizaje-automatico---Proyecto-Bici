# Reporte de data drift y pérdida de desempeño — MiBici (Equipo 19)

Generado automáticamente por `python -m src.monitoring.drift` el 2026-11-22T18:48:14+00:00 (semilla 42, modelo `models/champion`, 35.8 s). Todos los números salen de `drift_report.json`.

## Referencia y línea base

- **Referencia de distribución:** conjunto de `val` (476,930 filas, 2025-09-15 14:00:00 a 2025-11-08 06:00:00). Es lo que pide la tarea y el periodo más reciente que conoce el modelo; contra él se calculan PSI, KS y Wasserstein.
- **Línea base de desempeño:** conjunto de `test` (476,930 filas, 2025-11-08 07:00:00 a 2025-12-31 23:00:00): MAE 0.7651, RMSE 1.3018, R² 0.6937, devianza de Poisson 0.9144, WAPE 0.6025, sesgo +4.6 %.
- El modelo final se entrenó con train + val, así que su MAE sobre validación (0.7916) es optimista (vio esos datos); por eso el desempeño se compara contra prueba, que nunca vio.
- Los escenarios se generan a partir del conjunto de prueba, de modo que la diferencia contra la línea base se debe solo a la alteración.

**Verificación de la reconstrucción desde viajes:** reconstruir la ventana de prueba desde los viajes limpios sin alterar da 476,930 filas (matriz procesada: 476,930) con diferencia máxima 0 en demand_count, demand_lag_24h, demand_lag_168h, demand_same_hour_mean_7d, sin_hour, cos_hour, is_weekend, is_holiday, capacity → idéntica. Así los escenarios de cambio real parten exactamente de la misma línea base.

## Resumen por escenario

| Escenario | Familia | Nivel | MAE | ΔMAE | WAPE | ΔWAPE | Variables con drift | Nivel sin etiquetas | Acción (regla) |
|---|---|---|---:|---:|---:|---:|---|---|---|
| control | Control | **OK** | 0.7651 | +0.0 % | 0.6025 | +0.0 % | ninguna | OK | `R0_ok` |
| rezagos_faltantes | Corrupción de entradas | **ALERTA** | 0.8222 | +7.5 % | 0.6475 | +7.5 % | rezago 24 h, rezago 168 h, media misma hora 7 d, predicción | ALERTA | `R1_pipeline` |
| rezagos_escala | Corrupción de entradas | **ALERTA** | 0.8843 | +15.6 % | 0.6964 | +15.6 % | rezago 24 h, rezago 168 h, media misma hora 7 d, predicción | ALERTA | `R1_pipeline` |
| reloj_desfasado | Corrupción de entradas | **ALERTA** | 1.0499 | +37.2 % | 0.8268 | +37.2 % | predicción | ALERTA | `R1_pipeline` |
| aumento_demanda | Cambio real | **ALERTA** | 1.1113 | +45.2 % | 0.6251 | +3.7 % | rezago 24 h, rezago 168 h, media misma hora 7 d | AVISO | `R3_reentrenar` |
| caida_demanda | Cambio real | **ALERTA** | 0.6611 | -13.6 % | 0.8666 | +43.8 % | rezago 24 h, rezago 168 h, media misma hora 7 d, predicción | ALERTA | `R3_reentrenar` |
| vacaciones_diciembre | Cambio real | **ALERTA** | 0.7190 | -6.0 % | 0.8055 | +33.7 % | rezago 24 h, media misma hora 7 d, predicción | AVISO | `R3_reentrenar` |
| estaciones_nuevas | Cambio real | **ALERTA** | 0.7892 | +3.1 % | 0.6215 | +3.1 % | ninguna | ALERTA | `R2_catalogo` |

Nivel = el más severo entre calidad de datos, drift de entradas y desempeño. "Nivel sin etiquetas" = lo que diría el monitoreo en producción antes de conocer la demanda real (solo calidad y drift de entradas).

## Dos familias de escenarios

- **Corrupción de entradas:** la demanda real no cambia; falla el pipeline que calcula las variables. Si el desempeño cae, la solución es arreglar el pipeline, no reentrenar (reentrenar con datos corruptos aprendería el error).
- **Cambio real de distribución:** cambia el comportamiento de los usuarios o del sistema; las variables y el objetivo se recalculan de forma coherente. Si el desempeño cae, el modelo quedó desactualizado y la acción es reentrenar y promover un challenger.

## Umbrales (config.yaml → monitoring.thresholds)

| Señal | Aviso (≥) | Alerta (≥) | Justificación |
|---|---:|---:|---|
| PSI por variable | 0.1 | 0.25 | Convención usual de la industria (riesgo crediticio); regla práctica, no prueba estadística. |
| Estadístico D de Kolmogorov-Smirnov | 0.1 | 0.2 | Tamaño del efecto: el p-valor es ~0 con cientos de miles de filas. Criterio del equipo. |
| Wasserstein / desv. est. de la referencia | 0.1 | 0.25 | Desplazamiento de 0.10 / 0.25 desviaciones estándar. Criterio del equipo. |
| Aumento de la tasa de faltantes | 0.01 | 0.05 | La referencia no tiene faltantes; 1 % ya indica una falla del pipeline. |
| Filas con rezagos de conteo no enteros o negativos | 0.01 | 0.05 | Los rezagos de 24 h y 168 h son conteos; mismo criterio que los faltantes. |
| Filas con estación no vista | 0.01 | 0.05 | 1 % de filas equivale a ~4 de 370 estaciones. |
| Desfase del perfil horario (horas) | 1 | 2 | Cualquier desfase indica reloj/zona horaria equivocados. |
| Degradación relativa del MAE | 0.1 | 0.25 | Lo prometido en el ML Canvas (sección 9): degradación del MAE. |
| Degradación relativa del WAPE | 0.1 | 0.25 | El MAE baja solo cuando baja la demanda; el WAPE corrige por el nivel. |

PSI con 10 intervalos por cuantiles de la referencia (variables con ≤ 30 valores distintos: un intervalo por valor), faltantes como categoría propia y épsilon 0.0001 en intervalos vacíos. KS sobre una submuestra fija de 50,000 filas por lado: con tantas filas el p-valor es prácticamente 0 ante cualquier diferencia, así que se decide con el estadístico D (tamaño del efecto) y el p-valor solo se reporta en el JSON. Los umbrales se fijaron antes de ver los resultados y no se recalibraron.

Tres ajustes de diseño (ningún umbral se movió) se hicieron después de la primera ejecución; los dos primeros porque el escenario de control salía en AVISO solo por `is_holiday`:
1. La Wasserstein normalizada no decide en variables discretas: en una binaria rara la desviación estándar es pequeña y una diferencia mínima de proporciones parece un desplazamiento grande; ahí deciden PSI y KS (que en una binaria es |p − q|).
2. Las variables de calendario (`sin_hour`, `cos_hour`, `sin_dayofweek`, `cos_dayofweek`, `is_weekend`, `is_holiday`) se reportan como informativas y no deciden el nivel de drift: son funciones deterministas de la fecha y su cambio solo refleja qué días caen en la ventana. Su falla real (reloj o zona horaria equivocados) la detecta el desfase del perfil horario.
3. Se agregó la validación de dominio de los rezagos de conteo (enteros no negativos) porque, sin ella, multiplicar los rezagos por una constante (error de unidades) se confundía con un aumento real de demanda y la regla proponía reentrenar. Limitación: un factor entero (p. ej. ×2) no rompe el dominio y seguiría pareciendo un cambio real hasta comparar los rezagos contra los viajes registrados.

## Reglas de decisión (en orden; gana la primera que aplica)

- `R1_pipeline`: Revisar el pipeline de variables y la calidad de datos (faltantes, unidades, zona horaria, esquema) antes de reentrenar: reentrenar con entradas corruptas solo aprendería el error.
- `R2_catalogo`: Actualizar el catálogo de estaciones y verificar el alta de las estaciones nuevas; reentrenar cuando acumulen historial (>= 7 días para los rezagos) y vigilar su error aparte.
- `R3_reentrenar`: Reentrenar con datos recientes, evaluar el challenger contra el champion en la misma ventana y, si mejora, promoverlo moviendo el alias challenger -> champion en el Registry.
- `R4_programar`: Programar reentrenamiento con datos recientes y aumentar la frecuencia de monitoreo; promover el challenger solo si supera al champion.
- `R5_concepto`: Posible concept drift (cambia la relación entradas-demanda sin drift claro en las entradas): revisar las etiquetas y el calendario (vacaciones, eventos) y reentrenar incorporándolos.
- `R6_vigilar`: Drift en entradas sin degradación del desempeño: vigilar, aumentar la frecuencia de monitoreo y confirmar con las etiquetas en cuanto lleguen.
- `R7_sin_etiquetas`: Drift fuerte en entradas y aún sin valor real: preparar un reentrenamiento (challenger) y evaluarlo con las primeras etiquetas disponibles; no promover sin evidencia de desempeño.
- `R0_ok`: Sin acción: continuar con el monitoreo programado.

Las reglas funcionan sin valor real (R1, R2, R6, R7): en producción la demanda de una hora se conoce después, así que la primera alarma sale de calidad y drift de entradas; el desempeño confirma cuando llegan las etiquetas.

## Interpretación por escenario

### control — OK

- Qué prueba: Sin alteración: mide si los umbrales dan falsas alarmas con datos normales.
- Configuración: Conjunto de prueba sin alteración (valida que los umbrales no den falsas alarmas). Parámetros: `{}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 0.7651 (+0.0 % vs. 0.7651), WAPE 0.6025 (+0.0 %), R² 0.6937, sesgo +4.6 % (demanda media 1.270 vs. predicción media 1.329).
- Mayor PSI: rezago 24 h 0.004, media misma hora 7 d 0.004, predicción 0.003; KS máx. 0.030; Wasserstein norm. máx. 0.059 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.995).
- Niveles: calidad OK, drift OK, desempeño OK → regla `R0_ok`. Sin etiquetas: OK (`R0_ok`).
- Acción propuesta: Sin acción: continuar con el monitoreo programado.

### rezagos_faltantes — ALERTA

- Qué prueba: Falla del pipeline: rezagos vacíos que el imputador rellena con la mediana de entrenamiento.
- Configuración: 30 % de las filas sin rezagos (falla de la consulta histórica; el imputador usa la mediana). Parámetros: `{"fraction": 0.3}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 0.8222 (+7.5 % vs. 0.7651), WAPE 0.6475 (+7.5 %), R² 0.5770, sesgo -7.1 % (demanda media 1.270 vs. predicción media 1.179).
- Mayor PSI: rezago 24 h 2.517, media misma hora 7 d 2.517, rezago 168 h 2.515; KS máx. 0.060; Wasserstein norm. máx. 0.117 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +30.1 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.995).
- Niveles: calidad ALERTA, drift ALERTA, desempeño OK → regla `R1_pipeline`. Sin etiquetas: ALERTA (`R1_pipeline`).
- Acción propuesta: Revisar el pipeline de variables y la calidad de datos (faltantes, unidades, zona horaria, esquema) antes de reentrenar: reentrenar con entradas corruptas solo aprendería el error.

### rezagos_escala — ALERTA

- Qué prueba: Falla del pipeline: rezagos en otra escala (unidades o agregación equivocadas).
- Configuración: Rezagos multiplicados por 1.5 (unidades o agregación equivocadas en el pipeline). Parámetros: `{"factor": 1.5}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 0.8843 (+15.6 % vs. 0.7651), WAPE 0.6964 (+15.6 %), R² 0.5545, sesgo +30.8 % (demanda media 1.270 vs. predicción media 1.661).
- Mayor PSI: rezago 168 h 1.461, rezago 24 h 1.448, media misma hora 7 d 0.478; KS máx. 0.163; Wasserstein norm. máx. 0.257 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 46.1 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.995).
- Niveles: calidad ALERTA, drift ALERTA, desempeño AVISO → regla `R1_pipeline`. Sin etiquetas: ALERTA (`R1_pipeline`).
- Acción propuesta: Revisar el pipeline de variables y la calidad de datos (faltantes, unidades, zona horaria, esquema) antes de reentrenar: reentrenar con entradas corruptas solo aprendería el error.

### reloj_desfasado — ALERTA

- Qué prueba: Falla del pipeline: calendario calculado con otra zona horaria. No cambia ninguna distribución marginal (cada hora sigue apareciendo igual de seguido), solo su relación con la demanda; por eso se vigila el perfil horario.
- Configuración: Variables de calendario calculadas con 6 h de desfase (UTC en lugar de hora de Guadalajara). Parámetros: `{"hours": 6}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 1.0499 (+37.2 % vs. 0.7651), WAPE 0.8268 (+37.2 %), R² 0.3571, sesgo -15.8 % (demanda media 1.270 vs. predicción media 1.069).
- Mayor PSI: predicción 0.332, rezago 24 h 0.004, media misma hora 7 d 0.004; KS máx. 0.151; Wasserstein norm. máx. 0.175 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +6 h (correlación sin desfase -0.149).
- Niveles: calidad ALERTA, drift ALERTA, desempeño ALERTA → regla `R1_pipeline`. Sin etiquetas: ALERTA (`R1_pipeline`).
- Acción propuesta: Revisar el pipeline de variables y la calidad de datos (faltantes, unidades, zona horaria, esquema) antes de reentrenar: reentrenar con entradas corruptas solo aprendería el error.

### aumento_demanda — ALERTA

- Qué prueba: Cambio real: más o menos viajes de forma sostenida; se recalculan la cuadrícula, los rezagos y el objetivo desde los viajes con la lógica de MiBiciFeatureEngineer.
- Configuración: 40 % más viajes (remuestreo con reemplazo); se recalculan rezagos y objetivo. Parámetros: `{"factor": 1.4}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 1.1113 (+45.2 % vs. 0.7651), WAPE 0.6251 (+3.7 %), R² 0.6477, sesgo -9.8 % (demanda media 1.778 vs. predicción media 1.603).
- Mayor PSI: rezago 168 h 0.046, rezago 24 h 0.040, media misma hora 7 d 0.038; KS máx. 0.061; Wasserstein norm. máx. 0.195 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.995).
- Niveles: calidad OK, drift AVISO, desempeño ALERTA → regla `R3_reentrenar`. Sin etiquetas: AVISO (`R6_vigilar`).
- Acción propuesta: Reentrenar con datos recientes, evaluar el challenger contra el champion en la misma ventana y, si mejora, promoverlo moviendo el alias challenger -> champion en el Registry.

### caida_demanda — ALERTA

- Qué prueba: Cambio real: más o menos viajes de forma sostenida; se recalculan la cuadrícula, los rezagos y el objetivo desde los viajes con la lógica de MiBiciFeatureEngineer.
- Configuración: 40 % menos viajes (adelgazamiento binomial); se recalculan rezagos y objetivo. Parámetros: `{"factor": 0.6}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 0.6611 (-13.6 % vs. 0.7651), WAPE 0.8666 (+43.8 %), R² 0.4942, sesgo +38.3 % (demanda media 0.763 vs. predicción media 1.055).
- Mayor PSI: media misma hora 7 d 0.177, rezago 24 h 0.114, rezago 168 h 0.097; KS máx. 0.155; Wasserstein norm. máx. 0.298 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.995).
- Niveles: calidad OK, drift ALERTA, desempeño ALERTA → regla `R3_reentrenar`. Sin etiquetas: ALERTA (`R7_sin_etiquetas`).
- Acción propuesta: Reentrenar con datos recientes, evaluar el challenger contra el champion en la misma ventana y, si mejora, promoverlo moviendo el alias challenger -> champion en el Registry.
- Nota: el MAE baja (-13.6 %) pero el WAPE sube (+43.8 %): la demanda bajó de nivel y el MAE absoluto, por sí solo, ocultaría que el error relativo empeoró.

### vacaciones_diciembre — ALERTA

- Qué prueba: Cambio real: periodo observado con un patrón estacional distinto (sin datos sintéticos).
- Configuración: Periodo real de vacaciones y fiestas de fin de año (20 al 31 de diciembre de 2025). Parámetros: `{"start": "2025-12-20 00:00", "end": "2025-12-31 23:00"}`; 106,560 filas (2025-12-20 00:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 0.7190 (-6.0 % vs. 0.7651), WAPE 0.8055 (+33.7 %), R² 0.5204, sesgo +27.6 % (demanda media 0.893 vs. predicción media 1.139).
- Mayor PSI: rezago 24 h 0.055, media misma hora 7 d 0.049, predicción 0.027; KS máx. 0.093; Wasserstein norm. máx. 0.187 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 0.0 % (0 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.960).
- Niveles: calidad OK, drift AVISO, desempeño ALERTA → regla `R3_reentrenar`. Sin etiquetas: AVISO (`R6_vigilar`).
- Acción propuesta: Reentrenar con datos recientes, evaluar el challenger contra el champion en la misma ventana y, si mejora, promoverlo moviendo el alias challenger -> champion en el Registry.
- Nota: el MAE baja (-6.0 %) pero el WAPE sube (+33.7 %): la demanda bajó de nivel y el MAE absoluto, por sí solo, ocultaría que el error relativo empeoró.

### estaciones_nuevas — ALERTA

- Qué prueba: Cambio real: estaciones con Id nunca visto que abren al inicio de la ventana: sin historial la primera semana (rezagos en 0) y sin codificación en el TargetEncoder.
- Configuración: 15 % de estaciones con Id nuevo que abren al inicio de la ventana (sin historial ni codificación). Parámetros: `{"fraction": 0.15, "id_offset": 100000}`; 476,930 filas (2025-11-08 07:00:00 a 2025-12-31 23:00:00).
- Desempeño: MAE 0.7892 (+3.1 % vs. 0.7651), WAPE 0.6215 (+3.1 %), R² 0.6582, sesgo +2.3 % (demanda media 1.270 vs. predicción media 1.300).
- Mayor PSI: predicción 0.008, media misma hora 7 d 0.005, rezago 24 h 0.005; KS máx. 0.032; Wasserstein norm. máx. 0.062 (variables que deciden; calendario en el JSON).
- Calidad: faltantes +0.0 %, rezagos fuera de dominio 0.0 %, filas con estación no vista 15.1 % (56 estaciones), desfase del perfil horario +0 h (correlación sin desfase 0.995).
- Niveles: calidad ALERTA, drift OK, desempeño OK → regla `R2_catalogo`. Sin etiquetas: ALERTA (`R2_catalogo`).
- Acción propuesta: Actualizar el catálogo de estaciones y verificar el alta de las estaciones nuevas; reentrenar cuando acumulen historial (>= 7 días para los rezagos) y vigilar su error aparte.

## Vista temporal: conjunto de prueba en ventanas semanales (sin alterar)

| Ventana | Inicio | Filas | MAE | ΔMAE | WAPE | ΔWAPE | Sesgo | PSI máx. (variable) | Nivel | Regla |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | 2025-11-08 | 62,160 | 0.7784 | +1.7 % | 0.5415 | -10.1 % | -1.0 % | 0.001 (media misma hora 7 d) | OK | `R0_ok` |
| 1 | 2025-11-15 | 62,160 | 0.7626 | -0.3 % | 0.5725 | -5.0 % | -3.9 % | 0.007 (predicción) | OK | `R0_ok` |
| 2 | 2025-11-22 | 62,160 | 0.7830 | +2.3 % | 0.5454 | -9.5 % | -2.1 % | 0.002 (rezago 168 h) | OK | `R0_ok` |
| 3 | 2025-11-29 | 62,160 | 0.7903 | +3.3 % | 0.5624 | -6.7 % | +0.7 % | 0.000 (media misma hora 7 d) | OK | `R0_ok` |
| 4 | 2025-12-06 | 62,160 | 0.7819 | +2.2 % | 0.5971 | -0.9 % | +4.7 % | 0.002 (rezago 24 h) | OK | `R0_ok` |
| 5 | 2025-12-13 | 62,160 | 0.7477 | -2.3 % | 0.5759 | -4.4 % | +3.9 % | 0.004 (media misma hora 7 d) | OK | `R0_ok` |
| 6 | 2025-12-20 | 62,160 | 0.7451 | -2.6 % | 0.8363 | +38.8 % | +32.9 % | 0.053 (rezago 24 h) | ALERTA | `R3_reentrenar` |
| 7 | 2025-12-27 | 41,810 | 0.7161 | -6.4 % | 0.7599 | +26.1 % | +20.4 % | 0.120 (media misma hora 7 d) | ALERTA | `R3_reentrenar` |

ΔMAE y ΔWAPE contra el conjunto de prueba completo; la última ventana puede estar incompleta. PSI máx. sin variables de calendario (informativas).

## Figuras

- `reports/figures/07_drift_distribuciones.png`
- `reports/figures/08_drift_psi_por_variable.png`
- `reports/figures/09_drift_desempeno.png`
- `reports/figures/10_drift_ventanas_tiempo.png`
