# Fase final - Análisis y reflexión de Gustavo Alejandro Vázquez Rivas (A01611559)

Versión final. La sección **Reflexión** se copia tal cual al reporte (`python reports/build_report.py`).

## Rol y responsabilidad en la fase final

**Rol:** Site Reliability Engineer (SRE).

Implementé la solución de la rama `fase-final` en una sola máquina (Windows 11, Python 3.13.9, Docker), con los
datos, la base de MLflow y Docker. Jesús Rincón implementó en paralelo otra solución completa en la rama
`entrega-final`. Comparé las dos ramas, integré en `fase-final` las piezas de la suya que resolvían mejor el problema y
publiqué la versión 1.1.0 del servicio. Los demás integrantes analizan los entregables de su rol en su propio archivo
de `reports/fase_final/`.

## Actividades por fase

### Fase 1
- Sustituí los datos simulados por la descarga real de los datos abiertos de MiBici (commit `b790aea`).
- Agregué el preprocesamiento y los modelos de referencia registrados en MLflow (`0c5de28`) y actualicé el reporte
  con resultados reales y capturas de MLflow (`70dd515`); hice el merge del PR #1.
- Como SRE: `requirements.txt`, `environment.yml` y la base SQLite de MLflow (`mlruns.db`).

### Fase 2
- Verifiqué el pipeline en un clon limpio con un venv nuevo (Windows 11, Python 3.13): 17 pruebas aprobadas.
- Documenté los hallazgos para esta fase: dependencias con `>=`, avisos de obsolescencia y el límite de rutas largas
  de Windows al instalar `jupyter` (`reports/fase2/gustavo.md`).

### Fase final
| Entregable | Qué hice | Evidencia |
|---|---|---|
| Entorno fijado | Versiones exactas (`==`) en `requirements.txt`, `requirements-serve.txt` y `requirements-lock.txt`; `set_global_seeds`; Optuna sin corte por tiempo; descarga automática de los 12 meses | commits `d0d7923`, `262bf86` |
| Reentrenamiento en limpio | Pipeline completo en un venv nuevo y comparación con la Fase 2: MAE de prueba 0.7651 vs 0.7656 (−0.06 %) | `reports/fase_final/comparacion_fase2.md` |
| Exportación | `models:/mibici-demand-forecaster/1` → `models/champion` con SHA-256 y patrones oro de referencia | `models/champion_metadata.json` |
| Pruebas | Migración a pytest y CI con Python 3.13: 305 pruebas, cobertura 94 % | `logs/10_pytest_cobertura.log` |
| API | FastAPI con `POST /predict`, validación Pydantic, OpenAPI y Postman; versión 1.1.0 con rezagos opcionales (feature store) y `PREDICTION_LOG` | `src/serving/`, `docs/api.md` |
| Docker | Dockerfile multi-etapa, imagen `gusrivasmna/mibici-demand-service` (1.0.0 y 1.1.0, con tags `-model-v1` y `latest`) | `Dockerfile`, `docs/docker.md`, `reproducibilidad/dockerhub_tags.json` |
| Reproducibilidad | Verificación del servicio en Windows, contenedor Linux, clon limpio, imagen descargada de DockerHub y GitHub Actions; reentrenamiento completo en Linux; binario original de la Fase 2 | `reports/fase_final/reproducibilidad/` |
| DVC | `dvc.yaml` con cinco etapas y `dvc.lock` | `docs/dvc.md` |
| Drift | 8 escenarios, PSI/KS/Wasserstein, MAE/WAPE, umbrales y reglas de acción | `reports/drift/` |
| Integración | Comparación de `fase-final` con `entrega-final` y merge (`14e4886`); feature store, modelo en Git, registro de predicciones, prueba de coherencia y job de build del CI tomados de la rama de Jesús | `git log`, sección 2.5 del reporte |
| Documentación | README, diagrama de la solución y reporte generado por código | `README.md`, `reports/build_report.py` |

## Análisis

**Reproducibilidad.** La medí en varios niveles. Reentrenar en un entorno limpio reprodujo la conclusión de la Fase 2
con una diferencia de −0.06 % en el MAE de prueba, explicada por cambios en los CSV del portal. El mismo artefacto
(`model.pkl` con SHA-256 verificado) reproduce las predicciones de referencia con diferencia 0 en Windows y con
diferencia máxima de 8.9e-16 en Linux (contenedor local, clon limpio e imagen descargada de DockerHub): 5 de 500 filas
difieren en 1 ulp porque la biblioteca matemática de C de Windows (UCRT) y la de Linux (glibc) redondean distinto el
`exp()` final del enlace de Poisson. Reentrenar el pipeline completo en Linux da modelos sin funciones trascendentes
idénticos bit a bit, pero el boosting elige otros hiperparámetros. Por eso lo que se despliega es el binario
verificado y la tolerancia de la verificación es 1e-9. El binario original de la Fase 2, entrenado en otra máquina
con Python 3.12, también reproduce aquí sus 1,000 predicciones de referencia.

**Integración de las dos ramas.** La rama `entrega-final` resolvía dos problemas que la mía tenía: el cliente debía
enviar los rezagos, y la imagen no se podía construir en un clon limpio porque el modelo no estaba en Git. Integré el
feature store (adaptado para usar la misma función de rezagos del entrenamiento), el modelo en Git, el registro de
predicciones, la prueba de coherencia servicio-pipeline y el job del CI que construye la imagen. Subí el servicio a
1.1.0 (cambio compatible) y conservé 1.0.0 publicada.

**Operación.** El servicio no depende de `mlruns.db`: carga una carpeta MLflow autocontenida, verifica el SHA-256 del
modelo y del feature store al arrancar y responde 503 si no coinciden. El contenedor corre como usuario sin
privilegios, con `HEALTHCHECK` y una base fijada por digest.

## Reflexión

En la Fase 2 propuse fijar las versiones exactas, automatizar las pruebas en GitHub Actions y monitorear el data
drift; en esta fase comprobé que cada una de esas medidas resolvía un problema real. La lección más importante para
mí como SRE es que la reproducibilidad tiene niveles y hay que declarar cuál se promete: el mismo modelo se reproduce
entre Windows y Linux con una diferencia de una unidad en el último bit, pero reentrenar en otro sistema operativo no
da el mismo modelo, porque una diferencia de redondeo en exp() cambia la búsqueda de Optuna. Sin patrones oro y una
tolerancia explícita no habríamos podido distinguir esa diferencia inocua de un error. También aprendí que comparar
dos soluciones independientes es una forma muy efectiva de revisión: la implementación de Jesús en la rama
entrega-final me mostró que pedirle los rezagos al cliente y dejar el modelo fuera de Git eran decisiones débiles que
yo no había cuestionado, y la integración dejó una solución mejor que cualquiera de las dos por separado. Como área de
mejora reconozco que trabajar en paralelo sin coordinarnos duplicó esfuerzo y que concentrar la ejecución en pocas
máquinas limitó la práctica de los demás roles; en un proyecto real acordaría desde el inicio quién implementa cada
pieza, abriría pull requests tempranos y automatizaría el pipeline completo en CI con un remoto de DVC para que
cualquier integrante pueda reproducirlo con un solo comando.
