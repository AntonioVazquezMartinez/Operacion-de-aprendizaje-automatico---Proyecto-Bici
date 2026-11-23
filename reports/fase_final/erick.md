# Fase final - Análisis y reflexión de Erick Francisco Siller Ojeda (A01382929)

Versión final. La sección **Reflexión** se copia tal cual al reporte (`python reports/build_report.py`).

## Rol y responsabilidad en la fase final

**Rol:** Software Engineer.

En esta fase el equipo trabajó dos aproximaciones (binario de la Fase 2 en `entrega-final` y reentrenamiento de
verificación en `fase-final`), que se integraron en `fase-final`. Mi responsabilidad como SWE fue el **análisis y la
revisión de la calidad de software**: la suite de pruebas (punto 1 de la entrega), la API (punto 2) y la
integración continua. Hice una revisión de los resultados de la implementación contra el repositorio; lo que sigue
es lo que encontré.

## Resultados revisados (con su archivo de origen)

| Tema | Resultado | Dónde verlo |
|---|---|---|
| Migración de pruebas | De `unittest` (21 pruebas heredadas, `logs/09_pytest.log`) a pytest con fixtures, marcador `integration` y `--strict-markers` | `tests/README.md`, `tests/conftest.py`, `pyproject.toml` |
| Resultado | `pytest -q`: 305 aprobadas, 0 fallidas, 48.1 s | `reports/fase_final/logs/10_pytest_cobertura.log` |
| Cobertura | 94 % de `src/` (2,164 sentencias y 380 ramas); el más bajo es `src/monitoring/drift.py` con 74 %; `src/serving/feature_store.py` 95 % | mismo log |
| Prueba de coherencia (sin training-serving skew) | El servicio recalcula rezagos y predicciones del feature store y deben coincidir con el pipeline (idea de la rama `entrega-final`) | `tests/test_serving_feature_store.py` |
| Integración | Carga → limpieza → características → Optuna → Model Registry temporal → recarga del champion → predicción → métricas | `tests/integration/test_pipeline_e2e.py` |
| API (1.1.0) | `POST /predict` con rezagos opcionales (sin ellos los calcula el feature store y responde `lags_source`), `POST /predict/raw`, `GET /health`, `GET /model-info`; Pydantic v2 con `extra="forbid"`; errores 422/404/503 con el mismo cuerpo JSON | `src/serving/`, `docs/api.md`, `docs/openapi.json` |
| Prueba de humo | 27 peticiones por entorno (13 respuestas 200, 11 de 422, 2 de 404, 1 de 405) y 3 de 503 sin modelo; venv: 500/500 idénticas con rezagos y 200/200 sin rezagos; contenedor Linux: 495/500 (1 ulp) y 200/200 | `reports/fase_final/logs/20_api_smoke.log` |
| Pruebas en Linux | Primera corrida en el contenedor Linux con el modelo real: 268 aprobadas y 2 fallidas (dos pruebas exigían igualdad exacta y fallaron por 1 ulp, 8.9e-16, en 5 de 500 filas). Se cambiaron a tolerancia 1e-9 y la corrida vigente da 270 aprobadas | `reproducibilidad/pipeline_linux_pytest.log` (antecedente), `reproducibilidad/pipeline_linux_pytest_v2.log` (vigente), `pipeline_linux_vs_windows.md` |
| CI | `tests.yml`: job de pruebas (Python 3.13, `pytest -q --cov=src`; con el modelo en Git también corren las pruebas que lo usan) y job `docker` que construye la imagen desde el repo, la levanta y la verifica (idea de `entrega-final`); `docker-reproducibilidad.yml`: hadolint y verificación de la imagen publicada | `.github/workflows/`, pestaña Actions |

## Actividades por fase

### Fase 1
- Corregí la descarga para que envíe un `User-Agent` de navegador; sin él el portal de MiBici responde 406
  (commit `9dfcd8d`).
- Agregué el flujo de trabajo reproducible: `Makefile`, `pyproject.toml` y el primer workflow de GitHub Actions
  (`.github/workflows/tests.yml`) que corre las pruebas en cada pull request (`768b809`).
- Ejecuté el notebook de EDA sobre los datos reales de mayo de 2024 y versioné sus figuras (`9fa3232`, `46053a0`).

### Fase 2
- Revisé la migración a Cookiecutter Data Science v2 (`93586e4`): que ningún módulo de `src/`
  conservara rutas, meses, umbrales o semillas fijos y que cada etapa corriera como CLI con `--config`.
- Corregí la prueba de días festivos, que pasaba sin verificar nada, y agregué dos pruebas de `read_month` (CSV en
  latin-1 con una fila mal formada y rechazo de un encabezado inesperado); `ModelTrainer` pasó a registrar en MLflow
  la configuración efectiva (`6213844`).
- Atendí la revisión del PR #2 (`a2a048a`) e integré `main` en `fase2-modelado` con un merge sin reescribir
  historia (`e682de8`), recuperando el `User-Agent` y el backend Agg que se habían perdido al mover el código.
- Detalle en `reports/fase2/erick.md`.

### Fase final
- **Pruebas.** Revisé la migración de `unittest` a pytest (`tests/README.md`, `tests/conftest.py`,
  `pyproject.toml`) y el resultado de `logs/10_pytest_cobertura.log`: 305 aprobadas en 48.1 s. Comprobé que siguen
  presentes las pruebas que protegían el comportamiento desde la Fase 2 (limpieza, rezagos sin fuga, partición
  cronológica, lectura de CSV) y que las que agregué entonces se conservaron.
- **Cobertura.** Revisé la cobertura por módulo y las líneas sin cubrir de `src/monitoring/drift.py` (74 %) y
  `src/serving/app.py` (95 %).
- **API.** Revisé `src/serving/schemas.py` y `docs/api.md` contra `tests/test_serving.py`,
  `tests/test_serving_feature_store.py` y la prueba de humo (`logs/20_api_smoke.log`).
- **CI.** Revisé `tests.yml` (jobs `tests` y `docker`) y `docker-reproducibilidad.yml`, y los dos logs de la suite
  en Linux (`pipeline_linux_pytest.log` y `pipeline_linux_pytest_v2.log`).

## Análisis

**Qué protege la suite y qué falta.** Las 305 pruebas cubren los comportamientos de los que depende que la métrica
sea creíble: las seis reglas de limpieza con anomalías en cantidades conocidas (`test_dataset_cleaning.py`), la
ausencia de fuga en los rezagos (`test_lags_never_use_future_information`, `test_no_short_lags_leak_future_horizon`),
la partición cronológica (`test_temporal_split_is_chronological`), el esquema de la API (422 uniformes,
`extra="forbid"`, `test_openapi_documenta_predict`) y la reproducción de las predicciones de referencia. La
cobertura es de 94 % sobre 2,164 sentencias y 380 ramas. El 100 % de `features.py` frente al 74 % de `drift.py` no
indica descuido: las líneas sin cubrir de `drift.py` (377-438) son la orquestación de la CLI y el registro en
MLflow, mientras que el cálculo que decide una alerta está en `metrics.py`, `rules.py` y `scenarios.py`, con 86 %,
99 % y 99 %. Perseguir 100 % ahí obligaría a simular MLflow para probar código que solo conecta piezas; prefiero un
umbral mínimo en el CI que impida retroceder. Lo que sí falta: ninguna prueba mide latencia o carga del servicio
(solo la prueba de humo registra tiempos) y no hay una prueba que avise cuando el feature store caduca.

**Lo que enseñaron las dos pruebas que fallaron en Linux.** La primera corrida de la suite en el contenedor Linux
con el modelo real dio 268 aprobadas y 2 fallidas. El código no tenía un error: las dos pruebas exigían igualdad
exacta con la referencia y 5 de 500 predicciones difieren en 1 ulp (8.9e-16) porque `exp()` redondea distinto en
Windows y en Linux. Era un defecto de la prueba, que afirmaba más de lo que el sistema promete. Con tolerancia 1e-9,
la misma que usa `scripts/verify_reproducibility.py`, la suite da 270 aprobadas. Una prueba de reproducibilidad debe
declarar su tolerancia y justificarla, y la tolerancia debe vivir en un solo lugar para que la prueba y el
verificador no se separen.

**API 1.1.0, coherencia y CI.** En la 1.0.0 el cliente debía enviar los tres rezagos, es decir, reimplementar
fuera del repositorio una parte de `src/features.py`; cualquier diferencia en ese cálculo produce predicciones
erróneas sin ningún error visible (training-serving skew). En la 1.1.0 el servidor los calcula con el feature store
y la misma función del entrenamiento, y `test_compute_lags_igual_al_pipeline` y
`test_predict_sin_rezagos_coincide_con_el_pipeline` lo convierten en una garantía verificada en cada push. Los
rezagos quedaron opcionales, así que toda petición válida en 1.0.0 lo sigue siendo
(`test_predict_con_rezagos_igual_que_version_1_0_0`). El job `docker` de `tests.yml` aporta algo que la verificación
de la imagen publicada no da: prueba que el commit actual construye una imagen que arranca y reproduce la
referencia, mientras que `docker-reproducibilidad.yml` verifica una imagen que alguien publicó a mano y que puede
no corresponder al código de la rama. De mi propuesta de la Fase 2, la prueba de contrato del encabezado se cumplió
a nivel de lectura (`test_read_month_rejects_unexpected_header`, parametrizada, y
`test_read_month_wrong_number_of_columns`); activar ruff no: sigue configurado en `pyproject.toml` pero ningún
workflow lo ejecuta. Al CI le agregaría ruff como paso previo a las pruebas y `--cov-fail-under=85`, la meta que el
equipo se fijó, para que la cobertura sea una condición y no solo un número en el log.

## Reflexión

En esta fase revisé la calidad del software desde mi rol, y eso me hizo ver la suite de pruebas como la especificación del sistema más que como un requisito: con 305 pruebas pude comprobar qué promete el servicio sin ejecutar el pipeline ni leer cada línea. La lección que más me dejó fue la de las dos pruebas que fallaron en Linux por una diferencia de 8.9e-16: el código estaba bien y la prueba estaba mal, porque exigía igualdad exacta cuando lo que el sistema promete es reproducir la referencia dentro de una tolerancia. Una prueba que afirma más de lo que el sistema garantiza es tan engañosa como una que no verifica nada, como la de días festivos que corregí en la Fase 2. También aprendí que la cobertura hay que leerla por módulo: el 74 % de drift.py corresponde a la orquestación de la línea de comandos, mientras que las reglas que deciden una alerta están cubiertas casi por completo, así que el 94 % global dice menos que saber qué quedó fuera. La prueba de coherencia entre el servicio y el pipeline me pareció la decisión de diseño más importante de la fase, porque elimina el problema en lugar de vigilarlo: el cliente ya no reimplementa los rezagos y el servidor usa la misma función del entrenamiento. Como área de mejora, de lo que propuse en la Fase 2 solo se cumplió la validación del encabezado de los CSV: ruff sigue configurado sin ejecutarse en el CI y la cobertura no tiene un umbral que haga fallar el flujo. Esos dos pasos, junto con una prueba de latencia del servicio, son lo que agregaría primero.
