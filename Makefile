# Flujo de trabajo reproducible del pipeline de datos de MiBici (Fase 1).
# Uso: make <target> [PYTHON=ruta/al/python]

PYTHON ?= python

.PHONY: help data clean-data features preprocess baselines pipeline test mlflow-ui

help: ## Muestra los targets disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

data: ## Descarga viajes y catálogo de estaciones a data/raw y registra la versión raw en MLflow
	$(PYTHON) -m src.data.downloader

clean-data: ## Limpia los viajes brutos hacia data/interim y registra la versión interim
	$(PYTHON) -m src.data.cleaner

features: ## Construye la matriz de demanda horaria en data/processed y las figuras del EDA
	$(PYTHON) -m src.features.engineer

preprocess: ## Particiona, escala, codifica y aplica PCA sobre la matriz procesada
	$(PYTHON) -m src.features.preprocessor

baselines: ## Entrena los baselines por versión de datos y los registra en MLflow
	$(PYTHON) -m src.models.train_baseline

pipeline: data clean-data features preprocess baselines ## Ejecuta el pipeline completo en orden

test: ## Corre la suite de pruebas automatizadas
	$(PYTHON) -m unittest tests/test_pipeline.py -v

mlflow-ui: ## Levanta la interfaz de MLflow sobre la base local mlruns.db
	mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
