# Fase final - Análisis y reflexión de Gustavo Alejandro Vázquez Rivas (A01611559)

> Borrador para revisión de Gustavo. La sección **Reflexión** se copia tal cual al reporte final cuando vuelve a
> generarse (`python reports/build_report.py`).

## Rol y responsabilidad en la fase final

**Rol:** Site Reliability Engineer (SRE).

Por acuerdo del equipo, ejecuté toda la parte técnica de la fase final en una sola máquina (Windows 11, Python
3.13.9, Docker), porque los datos, la base de MLflow y Docker no están en GitHub. Los demás integrantes analizan y
revisan los entregables de su rol en su propio archivo de `reports/fase_final/`.

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
| Reentrenamiento en limpio | Pipeline completo en un venv nuevo y comparación con la Fase 2: MAE de prueba 0.7651 vs 0.7656 (−0.06 %) | `reports/fase_final/comparacion_fase2.md`, `logs/01`-`05` (commit `5f14430`) |
| Exportación | `models:/mibici-demand-forecaster/1` → `models/champion` con SHA-256 y 500 predicciones de referencia | `models/champion_metadata.json` (commit `3dc21f7`) |
| API | FastAPI con `POST /predict`, validación Pydantic, OpenAPI y Postman | `src/serving/`, `docs/` (commit `3eeb45e`) |
| Drift | 8 escenarios, PSI/KS/Wasserstein, MAE/WAPE, umbrales y reglas de acción | `reports/drift/` (commit `853fc92`) |
| Pruebas | Migración a pytest: 270 pruebas, cobertura 93 %, CI con Python 3.13 | `logs/10_pytest_cobertura.log` (commit `fe66e5a`) |
| Docker y reproducibilidad | Dockerfile multi-etapa, imagen `gusrivasmna/mibici-demand-service`, verificación del servicio en contenedor Linux y reentrenamiento completo en Linux | `Dockerfile`, `docs/docker.md`, `reports/fase_final/reproducibilidad/` |
| Documentación | README, diagrama de la solución y reporte generado por código | `README.md`, `scripts/build_architecture_diagram.py`, `reports/build_report.py` |

## Análisis

**Reproducibilidad.** La medí en tres niveles. (1) Reentrenar en un entorno limpio reprodujo la conclusión de la
Fase 2 con una diferencia de −0.06 % en el MAE de prueba, explicada por cambios en los CSV del portal. (2) El mismo
artefacto (`model.pkl` con SHA-256 verificado) reproduce las 500 predicciones de referencia con diferencia 0 en
Windows (venv completo y venv solo de servicio) y con diferencia máxima de 8.9e-16 en el contenedor Linux: 5 filas
difieren en 1 ulp porque la biblioteca matemática de C de Windows (UCRT) y la de Linux (glibc) redondean distinto el
`exp()` final del enlace de Poisson. (3) Reentrenar todo el pipeline en Linux da modelos sin funciones trascendentes
idénticos bit a bit, pero el boosting elige otros hiperparámetros, con MAE de prueba 0.7653 contra 0.7651. Por eso lo
que se despliega es el binario verificado y la tolerancia de la verificación es 1e-9.

**Operación.** El servicio no depende de `mlruns.db` ni del código de entrenamiento: carga una carpeta MLflow
autocontenida, verifica el SHA-256 al arrancar y responde 503 si el modelo no es el esperado. El contenedor corre
como usuario sin privilegios, tiene `HEALTHCHECK` y una base fijada por digest.

**Monitoreo.** Los umbrales no generan falsas alarmas en el escenario de control y el drift real aparece desde el
20 de diciembre (WAPE +38.8 %). La acción que propongo para operación es reentrenar antes de las vacaciones e
incorporar el calendario escolar.

## Reflexión

En la Fase 2 propuse fijar las versiones exactas, automatizar las pruebas en GitHub Actions y monitorear el data
drift; en esta fase pude comprobar que cada una de esas medidas resolvía un problema real. La lección más importante
para mí como SRE es que la reproducibilidad tiene niveles y hay que declarar cuál se promete: el mismo modelo se
reproduce entre Windows y Linux con una diferencia de una unidad en el último bit, pero reentrenar en otro sistema
operativo no da el mismo modelo, porque una diferencia de redondeo en exp() cambia la búsqueda de Optuna. Sin las
500 predicciones de referencia y una tolerancia explícita no habríamos podido distinguir esa diferencia inocua de un
error. También aprendí que los hashes de los archivos intermedios dependen del sistema operativo (fin de línea) y
que el portal de datos cambia sin avisar, por lo que versionar el artefacto final con su SHA-256 es más confiable
que confiar en reentrenar. Como área de mejora reconozco que concentrar toda la ejecución en una sola máquina
simplificó la coordinación pero limitó la práctica de los demás roles; en un proyecto real automatizaría el pipeline
completo en CI con un remoto de DVC, para que cualquier integrante pueda reproducirlo con un solo comando, y
agregaría métricas del servicio (latencia y errores) al monitoreo.
