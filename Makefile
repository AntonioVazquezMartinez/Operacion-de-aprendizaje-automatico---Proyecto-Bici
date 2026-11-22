#################################################################################
# Comandos del proyecto MiBici (Cookiecutter Data Science)                      #
# En Windows sin `make`, ejecuta los comandos de cada regla directamente.       #
#################################################################################

PYTHON ?= python

.PHONY: requirements data features eda train predict test mlflow-ui pipeline clean export serve verify drift docker-build docker-run

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

## Exportar el modelo champion del Model Registry a models/champion (paquete de despliegue)
export:
	$(PYTHON) -m src.serving.export_model

## Levantar la API localmente (http://localhost:8000/docs)
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

## Verificar la reproducibilidad del modelo desplegable en este entorno
verify:
	$(PYTHON) -m src.serving.verify --full

## Simular data drift y medir la pérdida de desempeño
drift:
	$(PYTHON) -m src.monitoring.drift

IMAGE ?= mibici-demand-api
VERSION ?= 1.0.0
ENGINE ?= podman

## Construir la imagen del servicio (ENGINE=docker para usar Docker)
docker-build:
	$(ENGINE) build $(if $(filter podman,$(ENGINE)),--format docker) -t $(IMAGE):$(VERSION) -t $(IMAGE):latest .

## Ejecutar el contenedor en el puerto 8000
docker-run:
	$(ENGINE) run --rm -p 8000:8000 $(IMAGE):$(VERSION)

## Pipeline completo de punta a punta
pipeline: data features eda train predict export drift

## Borrar archivos compilados de Python
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
