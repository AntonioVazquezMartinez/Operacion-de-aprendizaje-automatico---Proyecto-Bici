#################################################################################
# Comandos del proyecto MiBici (Cookiecutter Data Science)                      #
# En Windows sin `make`, ejecuta los comandos de cada regla directamente.       #
#################################################################################

PYTHON ?= python
IMAGE ?= ml-service:latest
URL ?= http://localhost:8000

.PHONY: requirements data features eda train predict export pipeline test test-cov serve drift \
        docker-build docker-run verify dvc-repro report mlflow-ui clean

## Instalar dependencias (versiones exactas) y el paquete en modo editable
requirements:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

## Descargar y limpiar los datos (registra raw e interim en MLflow)
data:
	$(PYTHON) -m src.dataset

## Construir la matriz de características (registra processed en MLflow)
features:
	$(PYTHON) -m src.features

## Generar las figuras del análisis exploratorio
eda:
	$(PYTHON) -m src.plots

## Entrenar, ajustar con Optuna, comparar y registrar modelos en el Model Registry
train:
	$(PYTHON) -m src.modeling.train

## Predecir el periodo de prueba con el modelo champion del Model Registry
predict:
	$(PYTHON) -m src.modeling.predict

## Exportar el champion a models/champion (metadatos, SHA-256 y predicciones de referencia)
export:
	$(PYTHON) -m src.modeling.export

## Pipeline completo de punta a punta
pipeline: data features eda train predict export

## Pruebas unitarias y de integración (comando único)
test:
	$(PYTHON) -m pytest -q

## Pruebas con cobertura por módulo (lo mismo que corre CI)
test-cov:
	$(PYTHON) -m pytest -q --cov=src --cov-report=term-missing

## Levantar la API de predicción (FastAPI) en el puerto 8000
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

## Simulación de data drift: reports/drift/ y figuras 07-10
drift:
	$(PYTHON) -m src.monitoring.drift

## Construir la imagen del servicio (requiere models/champion)
docker-build:
	docker build -t $(IMAGE) .

## Correr la imagen del servicio
docker-run:
	docker run -p 8000:8000 $(IMAGE)

## Verificar un servicio levantado contra las 500 predicciones de referencia
verify:
	$(PYTHON) scripts/verify_reproducibility.py --url $(URL)

## Reproducir el pipeline con DVC (solo las etapas afectadas por cambios)
dvc-repro:
	dvc repro

## Diagrama de la solución y reporte final en Word
report:
	$(PYTHON) scripts/build_architecture_diagram.py
	$(PYTHON) reports/build_report.py

## Abrir la interfaz de MLflow
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

## Borrar archivos compilados de Python
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
