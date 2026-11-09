# Diccionario de datos - MiBici Guadalajara

## Viajes (data/raw/viajes_raw.csv)
Fuente: portal de datos abiertos de MiBici, un CSV por mes.

| Columna | Tipo | Descripción |
|---|---|---|
| Viaje_Id | entero | Identificador único del viaje |
| Usuario_Id | entero | Identificador anónimo del usuario (no se usa en el modelo) |
| Genero | texto | M, F o vacío (se imputa como "Desconocido") |
| Año_de_nacimiento | entero | Año de nacimiento declarado (se imputa con la mediana) |
| Inicio_del_viaje | fecha-hora | Momento en que se extrae la bicicleta |
| Fin_del_viaje | fecha-hora | Momento en que se ancla la bicicleta |
| Origen_Id | entero | Estación donde inicia el viaje |
| Destino_Id | entero | Estación donde termina el viaje |

Columnas derivadas en `data/interim/viajes_cleaned.csv`: `duracion_segundos`, `duracion_minutos`, `Edad`.

## Catálogo de estaciones (data/raw/nomenclatura_estaciones.csv)

| Columna | Descripción |
|---|---|
| id | Identificador de la estación (llave con Origen_Id / Destino_Id) |
| name | Nombre, p. ej. "(GDL-049) López Cotilla / Penitenciaría" |
| latitude, longitude | Coordenadas |
| capacity | Número de anclajes (renombrada desde `dpcapacity`) |

## Matriz de modelado (data/processed/mibici_hourly_demand.csv)
Una fila por estación y hora.

| Columna | Uso | Descripción |
|---|---|---|
| Origen_Id | entrada (TargetEncoder) | Estación |
| fecha_hora | partición | Inicio del bloque horario |
| demand_count | **objetivo** | Extracciones de bicicletas en esa estación y hora |
| hour, day_of_week, month | auxiliares | Componentes de calendario |
| sin_hour, cos_hour, sin_dayofweek, cos_dayofweek | entrada | Codificación cíclica de hora y día |
| is_weekend | entrada | 1 si es sábado o domingo |
| is_holiday | entrada | 1 si es día de descanso obligatorio en México (LFT Art. 74) |
| latitude, longitude, capacity | entrada | Ubicación y tamaño de la estación |
| demand_lag_24h | entrada | Demanda de la misma estación 24 h antes |
| demand_lag_168h | entrada | Demanda de la misma estación una semana antes |
| demand_same_hour_mean_7d | entrada | Promedio de la demanda a la misma hora en los 7 días previos |

Todos los rezagos usan solo información disponible al momento de predecir con hasta 24 h de anticipación.
