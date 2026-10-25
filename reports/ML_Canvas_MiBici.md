# Machine Learning Canvas: Predicción de Demanda en MiBici Guadalajara

## 1. Propuesta de Valor (Value Proposition)
- **Problema de Negocio:** El sistema de bicicletas públicas MiBici experimenta asimetrías severas en el flujo de viajes. Durante horas pico laborales y estudiantiles (8:00–10:00 y 18:00–20:00), estaciones clave (e.g. Chapultepec, Parque Revolución, Centro Médico, CUCEI) quedan vacías (desabasto) o completamente llenas (bloqueo para devolver unidades), deteriorando la experiencia del usuario y elevando costos de rebalanceo reactivo.
- **Solución Propuesta:** Sistema predictivo de demanda horaria de bicicletas por estación que anticipe los requerimientos de inventario con 1 a 24 horas de antelación.
- **Impacto Esperado:**
  - Reducción del 35% en eventos de estaciones vacías o saturadas.
  - Optimización de rutas para las camionetas de redistribución (rebalanceo proactivo en lugar de reactivo).
  - Aumento en la retención y satisfacción de los usuarios del sistema.

---

## 2. Fuentes de Datos (Data Sources)
- **Datos de Viajes (MiBici Open Data):** Registros históricos de trayectos mensuales (`Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`).
- **Nomenclatura y Catálogo de Estaciones:** Metadatos de infraestructura (`id`, `name`, `obcn`, `location`, `latitude`, `longitude`, `capacity`).
- **Datos de Calendario:** Días festivos oficiales en México, vacaciones escolares, días laborables vs. fines de semana.

---

## 3. Recolección e Ingesta de Datos (Data Collection)
- **Frecuencia de Ingesta:** Carga mensual y agregaciones diarias/horarias automatizadas vía pipeline ETL.
- **Almacenamiento:** Repositorio estructurado `data/raw/` (inmutable), datos depurados en `data/interim/`, datasets agregados para modelado en `data/processed/`.
- **Trazabilidad:** Versionado con **MLflow Tracking**, generando hashes SHA-256 de cada dataset para garantizar reproducibilidad exacta de experimentos.

---

## 4. Ingeniería de Características (Features)
- **Temporales:**
  - `hour`: Hora del día (0 a 23).
  - `day_of_week`: Día de la semana (0 = Lunes a 6 = Domingo).
  - `is_weekend`: Variable binaria (1 para sábado/domingo, 0 en días laborales).
  - `month`: Mes del año.
  - `is_holiday`: Indicador de día inhábil / festivo.
- **Cíclicas:**
  - `sin_hour = sin(2 * π * hour / 24)`
  - `cos_hour = cos(2 * π * hour / 24)`
  - `sin_dayofweek = sin(2 * π * day_of_week / 7)`
  - `cos_dayofweek = cos(2 * π * day_of_week / 7)`
- **Espaciales y de Estación:**
  - `station_id`: Identificador de la estación.
  - `latitude`, `longitude`: Coordenadas geográficas.
  - `capacity`: Capacidad física de anclajes de la estación.
- **Variables Históricas (Lags / Rolling):**
  - `demand_lag_1h`: Demanda de la estación en la hora previa.
  - `demand_lag_24h`: Demanda en el mismo bloque horario el día anterior.
  - `demand_rolling_avg_7d`: Promedio móvil de demanda para el mismo día y hora en las últimas semanas.

---

## 5. Tarea de Aprendizaje Automático (Prediction Task)
- **Tipo de Tarea:** Regresión Supervisada (conteo y series de tiempo tabulares).
- **Variable Objetivo ($y$):** `demand_count`: Número entero no negativo que representa las extracciones de bicicletas en una estación $s$ durante el bloque horario $[t, t+1)$.
- **Familia de Modelos Candidatos (Fase 2):**
  - Baseline: Ridge / Poisson Regression.
  - Avanzados: Random Forest Regressor, XGBoost, LightGBM.

---

## 6. Decisiones y Acciones Operativas (Decisions & Actions)
- **Acción Operativa:** La predicción genera automáticamente una orden de logística de rebalanceo para el equipo operativo de MiBici.
- **Lógica de Decisión:**
  - Si $\text{Demanda\_Predicha}_{s, t+1} > \text{Inventario\_Actual}_s \times 0.8$: Enviar camioneta de rebalanceo con lote de bicicletas antes de la hora pico.
  - Si $\text{Demanda\_Predicha}_{s, t+1} < \text{Inventario\_Actual}_s \times 0.2$ y la estación está al 90% de capacidad: Retirar bicicletas para evitar saturación y permitir recepciones.

---

## 7. Modo de Inferencia (Making Predictions)
- **Frecuencia:** Modo Batch / Micro-batch cada 60 minutos (para programar la logística de la siguiente hora y turno).
- **Latencia Aceptable:** < 5 segundos por corrida completa de inferencia para todas las estaciones de la red.
- **Despliegue Futuro (Fase 3):** API REST implementada en **FastAPI**, empaquetada en contenedor **Podman / Docker**.

---

## 8. Evaluación Offline (Offline Evaluation)
- **Estrategia de Partición:** *Time-based Split* (e.g. 70% entrenamiento cronológico, 15% validación, 15% prueba final) para evitar fuga de información temporal (*data leakage*).
- **Métricas Técnicas:**
  - **MAE (Mean Absolute Error):** Error medio en cantidad absoluta de bicicletas demandadas (interpretable para el operador).
  - **RMSE (Root Mean Squared Error):** Penaliza grandes desviaciones en estaciones críticas.
  - **$R^2$ Score:** Coeficiente de determinación sobre la varianza explicada.

---

## 9. Monitoreo y Evaluación en Vivo (Live Evaluation & Drift)
- **KPIs Operativos:** Tasa de disponibilidad efectiva en estaciones (> 95%), porcentaje de viajes frustrados por estación vacía.
- **Detección de Drift (Fase 3):**
  - **Data Drift:** Prueba Kolmogorov-Smirnov o Wasserstein Distance sobre las variables temporales y de clima.
  - **Concept Drift:** Degradación progresiva del MAE diario frente a los viajes efectivamente registrados.
- **Acción ante Drift:** Disparo de reentrenamiento automático y registro de nuevo modelo en MLflow Model Registry.

---

## 10. Restricciones y Consideraciones Éticas (Constraints & Risks)
- **Privacidad:** No se almacena información identificable de usuarios individuales (`Usuario_Id` se anonimiza y solo se emplean agregaciones demográficas generales).
- **Disponibilidad:** El pipeline debe ser resiliente ante fallas temporales de conexión a internet o inconsistencias en los datos abiertos.
- **Reproducibilidad:** Registro de entorno virtual, versiones de librerías y semillas de aleatoriedad fijas.
