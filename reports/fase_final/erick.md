# Fase final - Análisis y reflexión de Erick Francisco Siller Ojeda (A01382929)

> Plantilla: escribe solo donde dice `<!-- ESCRIBE AQUÍ -->` (borra el comentario al escribir). No necesitas
> ejecutar nada: todos los resultados ya están en el repositorio. La sección **Reflexión** se copia tal cual
> al reporte final cuando vuelve a generarse, así que redáctala en primera persona y en párrafos.

## Rol y responsabilidad en la fase final

**Rol:** Software Engineer.

En esta fase la ejecución técnica la hizo Gustavo (SRE). Tu responsabilidad como SWE es el **análisis y la
revisión de la calidad de software**: la suite de pruebas (punto 1 de la entrega), la API (punto 2) y la
integración continua.

## Resultados que debes revisar (con su archivo de origen)

| Tema | Resultado | Dónde verlo |
|---|---|---|
| Migración de pruebas | De `unittest` (21 pruebas heredadas, `logs/09_pytest.log`) a pytest con fixtures, marcador `integration` y `--strict-markers` | `tests/README.md`, `tests/conftest.py`, `pyproject.toml` |
| Resultado | `pytest -q`: 270 aprobadas, 0 fallidas, 45.6 s | `reports/fase_final/logs/10_pytest_cobertura.log` |
| Cobertura | 93 % de `src/` (líneas y ramas); el más bajo es `src/monitoring/drift.py` con 74 % | mismo log |
| Integración | Carga → limpieza → características → Optuna → Model Registry temporal → recarga del champion → predicción → métricas | `tests/integration/test_pipeline_e2e.py` |
| API | `POST /predict`, `POST /predict/raw`, `GET /health`, `GET /model-info`; Pydantic v2 con `extra_forbidden`; errores 422/404/503 con el mismo cuerpo JSON | `src/serving/`, `docs/api.md`, `docs/openapi.json` |
| Prueba de humo | 21 peticiones por entorno (9 respuestas 200, 9 de 422, 2 de 404, 1 de 405) y 3 de 503 sin modelo; 500/500 predicciones idénticas en venv completo y en venv solo de servicio | `reports/fase_final/logs/20_api_smoke.log` |
| Pruebas en Linux | En el contenedor Linux, 268 aprobadas y 2 fallidas: las dos pruebas que exigen igualdad exacta con la referencia fallan por 1 ulp (8.9e-16) en 5 de 500 filas | `reports/fase_final/reproducibilidad/pipeline_linux_pytest.log`, `pipeline_linux_vs_windows.md` (sección 5) |
| CI | `tests.yml`: Python 3.13, `pytest -q --cov=src` en push a `main`/`fase-final` y en PR; `docker-reproducibilidad.yml`: hadolint y verificación de la imagen publicada | `.github/workflows/` |

## Preguntas guía

1. ¿Qué comportamientos críticos protege la suite (reglas de limpieza, ausencia de fuga en rezagos, partición
   cronológica, esquema de la API, reproducción de la referencia)? ¿Qué falta por probar?
2. ¿Qué revela el 74 % de `drift.py` frente al 100 % de `features.py`? ¿Vale la pena perseguir 100 %?
3. ¿Qué enseñan las dos pruebas que fallan en Linux por 1 ulp? ¿Cómo deberían escribirse las pruebas de
   reproducibilidad (tolerancia explícita, igual que `scripts/verify_reproducibility.py` con 1e-9)?
4. ¿Por qué `/predict` recibe solo estación, fecha y rezagos y el servidor deriva el resto con `src/features.py`?
   ¿Qué riesgo evita (training-serving skew)?
5. Retoma tu propuesta de la Fase 2 (prueba de contrato del esquema de cada CSV y ruff antes de las pruebas): ¿se
   cumplió? ¿Qué agregarías al CI?

## Actividades por fase

### Fase 1
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (estructura Cookiecutter, POO, unittest, Makefile/pyproject/CI, User-Agent...). -->

### Fase 2
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (revisión de la migración, prueba de festivos, read_month, merge de main, CI...). -->

### Fase final
<!-- ESCRIBE AQUÍ: qué revisaste de pruebas, cobertura, API y CI. -->

## Análisis

<!-- ESCRIBE AQUÍ: 2 o 3 párrafos respondiendo las preguntas guía con datos de la tabla. -->

## Reflexión

<!-- ESCRIBE AQUÍ: un párrafo (8 a 12 líneas): lecciones aprendidas y áreas de mejora desde tu rol. -->
