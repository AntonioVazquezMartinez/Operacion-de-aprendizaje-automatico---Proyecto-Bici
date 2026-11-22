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

## Entrega Final

- Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como apoyo en:
  - la generación del código de las pruebas con pytest (unitarias y de integración de extremo a extremo);
  - el servicio de predicción con FastAPI y Pydantic, la exportación del modelo champion desde el MLflow Model Registry y la verificación de reproducibilidad;
  - el `Dockerfile`, la construcción de la imagen con Podman y su publicación en Docker Hub;
  - la simulación de data drift, los umbrales y criterios de alerta;
  - la documentación del repositorio.

  El equipo revisó, ejecutó y validó el código. Las pruebas, métricas, resultados de drift y capturas corresponden a la ejecución real sobre los datos abiertos de MiBici en las computadoras de los integrantes y en el contenedor. https://claude.ai/
