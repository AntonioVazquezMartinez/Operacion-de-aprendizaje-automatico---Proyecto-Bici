# Declaración de Uso de Herramientas de Inteligencia Artificial - Equipo 19

En apego a los lineamientos de la actividad, se declara el uso de herramientas de inteligencia artificial en el proyecto.

## Fase 1

- Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como herramienta de consulta y de apoyo en la revision de la redaccion, la organizacion, el formato del reporte y la rubrica. El experimento corresponde a la ejecucion real del pipeline sobre los datos abiertos de MiBici, registrada en MLflow. https://claude.ai/

## Fase 2

- Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como apoyo en:
  - la reestructuración del repositorio a la plantilla Cookiecutter Data Science v2;
  - la refactorización del código a módulos configurables mediante `config.yaml`;
  - la generación del código del pipeline de scikit-learn, el ajuste de hiperparámetros con Optuna, el seguimiento de experimentos con MLflow y el registro de modelos en el MLflow Model Registry;
  - las pruebas automatizadas y la documentación del repositorio.

  El equipo revisó, ejecutó y validó el código. Las métricas y capturas reportadas corresponden a la ejecución real del pipeline sobre los datos abiertos de MiBici, registrada en MLflow. https://claude.ai/

## Fase final

- Anthropic. (2026). Claude (Fable 5.2 y Opus 5.5) [Modelo de lenguaje grande], utilizado como apoyo en:
  - las pruebas con pytest (unitarias, de integración y de coherencia servicio-pipeline), su cobertura y la integración continua;
  - el servicio de predicción con FastAPI y Pydantic, el feature store, la exportación del modelo champion y su documentación (OpenAPI y Postman);
  - el Dockerfile, la construcción de las imágenes con Docker y Podman, su publicación en DockerHub, los scripts de verificación de reproducibilidad entre entornos y la configuración de DVC;
  - la simulación de data drift (escenarios, métricas, umbrales y reglas de acción);
  - el contraste e integración de las dos aproximaciones de la fase final (ramas `entrega-final` y `fase-final`);
  - el diagrama de la solución, el generador del reporte final en Word y la redacción y revisión de la documentación.

  El equipo revisó, ejecutó y validó el código. Las pruebas, métricas, tablas, figuras y capturas reportadas corresponden a ejecuciones reales sobre los datos abiertos de MiBici en las computadoras de los integrantes, en los contenedores y en GitHub Actions, registradas en los logs y artefactos del repositorio (`reports/fase_final/`, `reports/drift/`, `models/champion_metadata.json`). https://claude.ai/
