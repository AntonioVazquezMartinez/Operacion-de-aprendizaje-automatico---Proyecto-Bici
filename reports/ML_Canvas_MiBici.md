# Machine Learning Canvas: Predicción de Demanda en MiBici Guadalajara

Versión final (fase final). El Canvas se planteó en la Fase 1; esta versión describe la solución que realmente
construimos. Al final de cada bloque, **Cambio** indica qué se modificó respecto de la Fase 1 y por qué.

## 1. Propuesta de Valor (Value Proposition)
- **Problema de Negocio:** El sistema de bicicletas públicas MiBici experimenta asimetrías severas en el flujo de viajes. Durante horas pico laborales y estudiantiles (picos a las 8:00 y 18:00), estaciones céntricas quedan vacías (desabasto) o completamente llenas (bloqueo para devolver unidades), deteriorando la experiencia del usuario y elevando costos de rebalanceo reactivo.
- **Solución:** Sistema predictivo de las extracciones de bicicletas por estación y hora con 1 a 24 horas de antelación, servido como API REST (FastAPI) en un contenedor Docker.
- **Impacto Esperado:**
  - Menos eventos de estaciones vacías o saturadas (la meta de la Fase 1 era −35 %; no se ha medido porque no hay datos de inventario en tiempo real).
  - Rebalanceo proactivo: priorizar estaciones por riesgo antes de las horas pico.
  - Aumento en la retención y satisfacción de los usuarios del sistema.
- **Cambio:** la meta de −35 % se mantiene como objetivo de negocio, pero se reconoce que no puede validarse sin datos de inventario.

---

## 2. Fuentes de Datos (Data Sources)
- **Datos de Viajes (MiBici Open Data):** 12 CSV mensuales de 2025 (`Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`); 4,532,032 viajes leídos en la fase final.
- **Catálogo de Estaciones:** `nomenclatura_2026_10.csv` (`id`, `name`, `latitude`, `longitude`, `capacity`), 470 estaciones; 370 activas en 2025.
- **Calendario:** días de descanso obligatorio de la Ley Federal del Trabajo (art. 74) y fin de semana.
- **Cambio:** la Fase 1 usaba un mes (mayo de 2024) y `nomenclatura_2026_09.csv`, que el portal retiró. No se incorporaron vacaciones escolares ni clima (área de mejora).

---

## 3. Recolección e Ingesta de Datos (Data Collection)
- **Frecuencia:** lotes mensuales; descarga automática con las URL de `config.yaml` (el portal exige User-Agent de navegador).
- **Almacenamiento:** `data/raw/` (inmutable), `data/interim/` (viajes limpios), `data/processed/` (matriz estación-hora).
- **Trazabilidad:** SHA-256 y `mlflow.log_input` en el experimento `MiBici_Datos`; pipeline y huellas MD5 en `dvc.yaml` / `dvc.lock`.
- **Cambio:** se agregó DVC en la fase final. Se confirmó que el portal republica archivos: con los mismos nombres se leen 11,937 viajes más que en la Fase 2.

---

## 4. Ingeniería de Características (Features)
- **Temporales y cíclicas:** `sin_hour`, `cos_hour`, `sin_dayofweek`, `cos_dayofweek`, `is_weekend`, `is_holiday`.
- **De estación:** `Origen_Id` (codificada con `TargetEncoder`), `latitude`, `longitude`, `capacity`.
- **Históricas (solo información disponible con 24 h de anticipación):** `demand_lag_24h`, `demand_lag_168h`, `demand_same_hour_mean_7d`.
- **Cambio:** se eliminó `demand_lag_1h` (Fase 1) porque no estaría disponible al predecir con más de una hora de anticipación (fuga de información, retroalimentación del profesor). Se agregó el rezago semanal y el festivo; `month` no se usa como entrada. Las primeras 168 h sin historial se descartan en lugar de rellenarlas con ceros.

---

## 5. Tarea de Aprendizaje Automático (Prediction Task)
- **Tipo de Tarea:** regresión supervisada de conteos sobre series de tiempo tabulares.
- **Variable Objetivo ($y$):** `demand_count`, extracciones de bicicletas en la estación $s$ durante $[t, t+1)$.
- **Modelos evaluados:** DummyRegressor (media), PoissonRegressor, RandomForestRegressor y HistGradientBoostingRegressor con pérdida Poisson (champion, `models:/mibici-demand-forecaster/1`).
- **Cambio:** el Canvas de la Fase 1 proponía Ridge, XGBoost y LightGBM. Se usó HistGradientBoosting, la implementación de scikit-learn inspirada en LightGBM, para no agregar dependencias; Ridge se sustituyó por la regresión de Poisson, adecuada para conteos.

---

## 6. Decisiones y Acciones Operativas (Decisions & Actions)
- **Acción Operativa:** la predicción alimenta la planeación del rebalanceo con camionetas.
- **Lógica de Decisión propuesta:**
  - Si $\text{Demanda\_Predicha}_{s, t+1} > \text{Inventario\_Actual}_s \times 0.8$: enviar camioneta con bicicletas antes de la hora pico.
  - Si $\text{Demanda\_Predicha}_{s, t+1} < \text{Inventario\_Actual}_s \times 0.2$ y la estación está al 90 % de capacidad: retirar bicicletas.
- **Cambio:** se mantiene como regla propuesta; no se validó porque no hay datos de inventario en tiempo real. Con un error relativo (WAPE) de 0.60 en prueba, el modelo sirve mejor para ordenar estaciones por riesgo que para fijar cantidades exactas.

---

## 7. Modo de Inferencia (Making Predictions)
- **Frecuencia:** lotes (por ejemplo, cada hora para las siguientes horas); `POST /predict` acepta hasta 1,000 instancias por petición.
- **Despliegue:** API REST con FastAPI (servicio 1.1.0) empaquetada en Docker; imagen `gusrivasmna/mibici-demand-service` (tags `1.1.0`, `1.1.0-model-v1`, `latest`; `1.0.0` sigue publicada). El cliente envía estación y hora; los rezagos son opcionales y, sin ellos, el servicio los calcula con un feature store de las últimas 360 h (ventana predecible 24-dic-2025 a 1-ene-2026).
- **Cambio:** la Fase 1 mencionaba Podman o Docker. La solución integrada usa Docker porque lo pide la entrega (DockerHub) y GitHub Actions lo ejecuta de forma nativa; la rama `entrega-final` construyó su imagen con Podman y esos comandos quedaron documentados. El feature store (idea de esa rama) evita que el cliente calcule los rezagos. La latencia objetivo de < 5 s no se midió para toda la red; las latencias por petición están en `reports/fase_final/logs/20_api_smoke.log`.

---

## 8. Evaluación Offline (Offline Evaluation)
- **Partición:** cronológica 70/15/15 (train hasta el 15-sep-2025, validación hasta el 8-nov, prueba hasta el 31-dic); selección por MAE de validación; prueba usada una sola vez.
- **Métricas:** MAE (decide el champion), RMSE, R² y devianza de Poisson. Champion en prueba: MAE 0.7651, RMSE 1.3018, R² 0.6937.
- **Cambio:** se agregó la devianza de Poisson y, en el monitoreo, el WAPE (MAE / demanda media), porque el MAE baja cuando baja la demanda.

---

## 9. Monitoreo y Evaluación en Vivo (Live Evaluation & Drift)
- **Data drift:** PSI, estadístico D de Kolmogorov-Smirnov y distancia de Wasserstein normalizada sobre las variables del modelo y la predicción, contra el conjunto de validación.
- **Calidad de datos:** faltantes, rezagos fuera de dominio, estaciones no vistas y desfase del perfil horario.
- **Desempeño (concept drift):** degradación del MAE y del WAPE contra el conjunto de prueba.
- **Umbrales y acciones:** OK / AVISO / ALERTA en `config.yaml`; reglas R0-R7 (corregir el pipeline, actualizar catálogo, reentrenar y promover el challenger, vigilar).
- **KPIs operativos (propuestos):** disponibilidad efectiva por estación y viajes frustrados por estación vacía.
- **Cambio:** no hay variables de clima, así que el drift se mide sobre las variables reales del modelo. El reentrenamiento no es automático: la regla R3 lo propone y se ejecuta con `python -m src.modeling.train` (o `dvc repro`).

---

## 10. Restricciones y Consideraciones Éticas (Constraints & Risks)
- **Privacidad:** `Usuario_Id`, género y edad no entran al modelo; la demanda se agrega por estación y hora.
- **Disponibilidad:** el pipeline tolera codificaciones distintas y registros mal formados; los datos versionados con hash permiten detectar cambios del portal.
- **Reproducibilidad:** dependencias con versión exacta, semilla única, imagen con digest fijo, SHA-256 del modelo verificado al arrancar y 500 predicciones de referencia (diferencia 0 en Windows y de 1 ulp en Linux).
- **Cambio:** se documentó que el reentrenamiento no es idéntico bit a bit entre sistemas operativos; por eso se despliega el binario verificado.
