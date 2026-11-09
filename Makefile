#################################################################################
# Comandos del proyecto MiBici (Cookiecutter Data Science)                      #
# En Windows sin `make`, ejecuta los comandos de cada regla directamente.       #
#################################################################################

PYTHON ?= python

.PHONY: requirements data features eda train predict test mlflow-ui pipeline clean

## Instalar dependencias y el paquete en modo editable
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

## Entrenar, ajustar con Optuna, comparar y registrar modelos
train:
	$(PYTHON) -m src.modeling.train

## Predecir el periodo de prueba con el modelo champion del Model Registry
predict:
	$(PYTHON) -m src.modeling.predict

## Ejecutar las pruebas
test:
	$(PYTHON) -m pytest -v

## Abrir la interfaz de MLflow
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

## Pipeline completo de punta a punta
pipeline: data features eda train predict

## Borrar archivos compilados de Python
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
