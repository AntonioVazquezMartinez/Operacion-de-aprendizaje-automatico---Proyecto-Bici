# Sistema MLOps: Prediccion de Demanda en MiBici Guadalajara (Fase 1) - Equipo 19

Este repositorio contiene la entrega de la Fase 1 (Procesamiento y Modelado Inicial) para el proyecto colaborativo del curso Operacion con Modelos de Machine Learning (TC5061) en la Maestria en Inteligencia Artificial Aplicada del Tecnologico de Monterrey.

- Repositorio: https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici
- Equipo Asignado: Equipo 19
- Integrantes:
  - Jose Antonio Vazquez Martinez (A01797208) - Data Engineer & ML Engineer
  - Jesus Angel Rincon Ruiz (A01793960) - Data Scientist
  - Erick Francisco Siller Ojeda (A01382929) - Software Engineer
  - Gustavo Alejandro Vazquez Rivas (A01611559) - Site Reliability Engineer (SRE)

El objetivo es construir la base de un pipeline productivo de Machine Learning con buenas practicas de MLOps. Organizamos el proyecto con Cookiecutter Data Science, programamos codigo modular con programacion orientada a objetos (POO), implementamos versionado y trazabilidad de datos con MLflow y repartimos las tareas entre los 5 roles clave de MLOps.

---

## Caso de Negocio: MiBici Guadalajara

- Problematica: En Guadalajara el sistema de bicicletas presenta desbalances constantes en las estaciones durante las horas pico de entrada y salida laboral. Hay estaciones que se quedan completamente vacias y otras que se llenan al tope, impidiendo que los usuarios puedan anclar su bicicleta.
- Propuesta de Valor: Construir un modelo predictivo que estime la demanda por hora en cada estacion con anticipacion, ayudando a planear las rutas de las camionetas de redistribucion para surtir o liberar bicicletas a tiempo.
- Formulacion ML: Modelo de regresion supervisada para predecir el numero de extracciones de bicicletas por estacion en bloques de una hora ("demand_count").

---

## Estructura del Proyecto (Cookiecutter Data Science)

```text
Tarea_4_Equipo/
│
├── .gitignore               <- Archivos y carpetas ignorados por git
├── README.md                <- Documentacion principal del proyecto
├── requirements.txt         <- Librerias de Python requeridas
├── environment.yml          <- Configuracion del entorno para Conda
│
├── data/
│   ├── raw/                 <- Datos brutos originales descargados de MiBici (inmutables)
│   ├── interim/             <- Datos limpios tras aplicar los filtros de calidad
│   └── processed/           <- Datos con caracteristicas listos para el modelo
│
├── notebooks/
│   └── 01_eda_mibici.ipynb  <- Cuaderno de analisis exploratorio de datos (EDA) y pruebas
│
├── src/                     <- Codigo fuente modular con POO
│   ├── data/
│   │   ├── downloader.py    <- Clase MiBiciDataDownloader: Adquisicion y generacion de datos base
│   │   └── cleaner.py       <- Clase MiBiciDataCleaner: Limpieza y validacion de reglas de negocio
│   ├── features/
│   │   └── engineer.py      <- Clase MiBiciFeatureEngineer: Variables de tiempo, ciclicas y de rezago
│   └── utils/
│       └── mlflow_logger.py <- Clase MLflowDataVersioner: Versionado, metadatos y trazabilidad en MLflow
│
├── tests/
│   └── test_pipeline.py     <- Pruebas unitarias automatizadas con unittest
│
├── models/                  <- Modelos entrenados y serializados (Fase 2)
│
└── reports/
    ├── ML_Canvas_MiBici.md  <- Canvas formal de Machine Learning
    ├── roles_mlops.md       <- Matriz de responsabilidades y flujo entre roles
    ├── Fase 1_Equipo19.docx <- Reporte ejecutivo en Word editable
    └── Fase 1_Equipo19.pdf  <- Reporte ejecutivo oficial en PDF (con figuras integradas)
```

---

## Guia de Inicio Rapido

### 1. Activar Entorno Virtual
```powershell
# En PowerShell
..\Tarea_3\.venv\Scripts\Activate.ps1
```

### 2. Ejecutar Pipeline de Ingesta, Limpieza y Features
```powershell
python -m src.data.downloader
python -m src.data.cleaner
python -m src.features.engineer
```

### 3. Ejecutar Pruebas Automatizadas
```powershell
python -m unittest tests/test_pipeline.py -v
```

### 4. Visualizar Experimentos y Versiones en MLflow UI
```powershell
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
```

---

## Matriz de Roles MLOps

| Rol | Integrante | Responsabilidades en Fase 1 |
|---|---|---|
| Data Engineer | Jose Antonio Vazquez Martinez | Descarga de datos, pipeline de limpieza ETL y registro de versiones raw e interim en MLflow. |
| Data Scientist | Jesus Angel Rincon Ruiz | Definicion del ML Canvas, analisis exploratorio (EDA) y diseno de variables de tiempo y rezagos. |
| Software Engineer | Erick Francisco Siller Ojeda | Estructuracion con Cookiecutter DS, refactorizacion a clases POO y pruebas unitarias. |
| Site Reliability Engineer (SRE) | Gustavo Alejandro Vazquez Rivas | Configuracion de dependencias, base de datos local SQLite para MLflow y entorno reproducible. |
