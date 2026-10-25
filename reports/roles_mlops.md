# Definicion de Roles MLOps e Interacciones Colaborativas (Fase 1) - Equipo 19

En esta primera fase del proyecto establecimos la forma de trabajo y las responsabilidades de cada integrante siguiendo los 5 roles fundamentales de MLOps.

---

## 1. Roles y Responsabilidades en la Fase 1

### 1. Data Engineer (DE) - Jose Antonio Vazquez Martinez (A01797208)
- Responsabilidad: Descarga de datos, limpieza, pipelines de calidad y almacenamiento ordenado.
- Tareas en Fase 1:
  - Disenar y programar la descarga de datos en downloader.py.
  - Implementar la clase MiBiciDataCleaner (cleaner.py) para eliminar duplicados, viajes con duracion negativa, viajes menores a 1 minuto y viajes mayores a 2 horas.
  - Asegurar que todas las estaciones de origen y destino correspondan al catalogo oficial de Guadalajara.
  - Guardar los archivos en data/raw y data/interim sin alterar los datos originales.
  - Registrar las versiones raw e interim en MLflow con sus respectivos hashes SHA-256.

---

### 2. Data Scientist (DS) - Jesus Angel Rincon Ruiz (A01793960)
- Responsabilidad: Planteamiento del problema, analisis exploratorio (EDA) e ingenieria de caracteristicas.
- Tareas en Fase 1:
  - Definir la propuesta de valor y completar el Machine Learning Canvas para la prediccion de demanda horaria.
  - Analizar las distribuciones de viaje en el notebook 01_eda_mibici.ipynb identificando las horas pico (manana y tarde), la diferencia entre semana y fin de semana, y las estaciones mas demandadas.
  - Disenar las variables de tiempo (transformaciones ciclicas con seno y coseno de hora y dia, y rezagos de demanda previa).
  - Programar la clase MiBiciFeatureEngineer (engineer.py) para generar la matriz final en data/processed.

---

### 3. Software Engineer (SWE) - Erick Francisco Siller Ojeda (A01382929)
- Responsabilidad: Calidad de software, arquitectura modular y pruebas unitarias.
- Tareas en Fase 1:
  - Organizar las carpetas del proyecto bajo el estandar Cookiecutter Data Science.
  - Refactorizar los scripts sueltos hacia clases de programacion orientada a objetos (POO) faciles de mantener y reutilizar.
  - Crear la suite de pruebas automatizadas con unittest (tests/test_pipeline.py) para comprobar que no existan duraciones negativas y que las 16 variables requeridas esten presentes.
  - Configurar las reglas de Git para ignorar archivos temporales y bases de datos locales.

---

### 4. Site Reliability Engineer (SRE) - Gustavo Alejandro Vazquez Rivas (A01611559)
- Responsabilidad: Entornos reproducibles, persistencia de configuraciones y seguimiento local.
- Tareas en Fase 1:
  - Configurar los archivos requirements.txt y environment.yml fijando las versiones necesarias de Python y librerias.
  - Configurar la base de datos local SQLite (mlruns.db) para que MLflow guarde los metadatos y artefactos de forma persistente.
  - Probar que el servidor web de MLflow levante correctamente en el puerto local y documentar las instrucciones para abrirlo.

---

### 5. Machine Learning Engineer (MLE) - Jose Antonio Vazquez Martinez (A01797208)
- Responsabilidad: Integracion del pipeline completo, trazabilidad y preparacion para futuros modelos.
- Tareas en Fase 1:
  - Conectar el modulo de versionado MLflowDataVersioner (mlflow_logger.py) para calcular los hashes SHA-256 de cada archivo de datos y guardarlos como parametros y tags.
  - Verificar que las salidas del modulo de datos empaten con las entradas necesarias para los modelos de regresion de la Fase 2.
  - Garantizar que el pipeline pueda correr de forma secuencial sin pasos manuales intermedios.

---

## 2. Matriz de Asignacion de Responsabilidades (RACI)

| Actividad / Entregable | Data Engineer | Data Scientist | Software Engineer | SRE | ML Engineer |
|---|:---:|:---:|:---:|:---:|:---:|
| Estructuracion Cookiecutter DS | I | I | A / R | C | C |
| Definicion del ML Canvas | C | A / R | I | I | C |
| Ingesta y Limpieza de Datos | A / R | C | C | I | C |
| Analisis Exploratorio (EDA) | C | A / R | I | I | C |
| Feature Engineering (POO) | C | R | C | I | A |
| Configuracion de MLflow y Entorno | C | I | I | R | A |
| Versionado de Datos en MLflow | R | I | I | C | A |
| Pruebas Unitarias | C | C | A / R | I | C |
| Compilacion de Reporte | C | C | C | C | A / R |

*Leyenda: R = Responsable (realiza la actividad), A = Accountable (aprueba y responde por el resultado), C = Consultado, I = Informado.*

---

## 3. Flujo de Comunicacion e Interaccion del Equipo

El equipo trabajo de manera conectada y en orden:
1. El SRE y el Software Engineer dejaron listo el entorno virtual, las librerias y la estructura de carpetas en Cookiecutter.
2. Con esa base, el Data Scientist y el Data Engineer definieron en el ML Canvas las reglas de negocio para limpiar y preparar los viajes de MiBici.
3. El Data Engineer descargo, valido y guardo los datos limpios, registrando las versiones en MLflow.
4. Con los datos limpios disponibles, el Data Scientist hizo el analisis exploratorio y armo las variables de tiempo y rezagos en la matriz procesada.
5. Finalmente, el Software Engineer y el ML Engineer programaron las pruebas automatizadas y confirmaron que todos los datos y artefactos quedaron registrados con sus hashes en MLflow, listos para la Fase 2.
