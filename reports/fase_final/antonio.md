# Fase final - Análisis y reflexión de José Antonio Vázquez Martínez (A01797208)

> Plantilla: escribe solo donde dice `<!-- ESCRIBE AQUÍ -->` (borra el comentario al escribir). No necesitas
> ejecutar nada: todos los resultados ya están en el repositorio. La sección **Reflexión** se copia tal cual
> al reporte final cuando vuelve a generarse, así que redáctala en primera persona y en párrafos.

## Rol y responsabilidad en la fase final

**Roles:** Data Engineer y ML Engineer.

En esta fase el equipo trabajó dos aproximaciones: servir el binario de la Fase 2 (rama `entrega-final`) y
reentrenar en un entorno limpio para verificar y complementar (rama `fase-final`); se integraron en `fase-final`
(merge `14e4886`).
Tu responsabilidad como DE/MLE es el **análisis y la revisión** de dos entregables:

1. **Etapa de datos:** ingesta de los 12 meses de 2025, limpieza, hashes SHA-256 y versionado con DVC.
2. **Ciclo de vida del modelo:** Model Registry → exportación del champion → servicio (API y contenedor).

Además, como dueño del repositorio, haces el merge del pull request final a `main`.

## Resultados que debes revisar (con su archivo de origen)

| Tema | Resultado | Dónde verlo |
|---|---|---|
| Datos leídos | 4,532,032 viajes en 2025 contra 4,520,095 en la Fase 2 (+11,937): el portal republicó archivos; julio ya no tiene comillas mal cerradas | `reports/fase_final/comparacion_fase2.md` (secciones 2 y 3), `logs/01_dataset.log` |
| Limpieza | 4,412,463 viajes limpios, retención 97.36 % (igual que en la Fase 2) | `comparacion_fase2.md`, tabla de la sección 2 |
| Catálogo | `nomenclatura_2026_09.csv` ya da 404; se usa `nomenclatura_2026_10.csv` (470 estaciones; 370 activas en la matriz) | `config.yaml` (`data.stations_url`) |
| Hash del dataset | Fase 2: empieza con `d00671f9` (captura del Registry); ahora `fec850d2…` | `models/champion_metadata.json` (`dataset_sha256`) |
| Hash y sistema operativo | El SHA-256 de un CSV cambia entre Windows y Linux solo por el fin de línea (CRLF vs LF) | `reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md` (sección 1) |
| Model Registry | `mibici-demand-forecaster` v1 = HistGradientBoosting (`champion`), v2 = RandomForest (`challenger`), v3 = Poisson; misma asignación que en la Fase 2 | `logs/04_train.log` |
| Exportación | `models:/mibici-demand-forecaster/1` → `models/champion/`, SHA-256 `ca4331a4…`, 500 predicciones de referencia | `models/champion_metadata.json`, `logs/06_export.log` |
| Modelo → servicio | La API verifica el SHA-256 del modelo y del feature store al arrancar; las 500 predicciones de referencia salen idénticas en Windows y con diferencia de 1 ulp (8.9e-16) en Linux (contenedor, clon limpio e imagen de DockerHub) | `docs/api.md`, `reports/fase_final/reproducibilidad/*.json` |
| Modelo en Git (desde 1.1.0) | `models/champion/` (unos 6.6 MB, con `feature_store.csv.gz`) está en Git, idea de la rama `entrega-final`: `docker build` funciona en un clon limpio | `reproducibilidad/clon_limpio.json`, `docs/docker.md` |
| Binario de la Fase 2 | `models/fase2_original/` (el `model.joblib` de la Fase 2, entrenado con Python 3.12) reproduce sus 1,000 predicciones de referencia; frente al reentrenado: mismos hiperparámetros, correlación 0.9989, distintos datos del portal | `reproducibilidad/fase2_binario_vs_reentrenado.md` |
| DVC | `dvc.yaml` (5 etapas) + `dvc.lock` sin remoto externo; las salidas de `export` van con `cache: false` porque el modelo ya está en Git (un archivo no puede estar en Git y en la caché de DVC a la vez) | `dvc.yaml`, `dvc.lock`, `docs/dvc.md` |

## Preguntas guía

1. ¿Qué riesgo operativo implica que el portal de MiBici republique CSV y retire catálogos sin avisar? ¿Cómo lo
   detectamos (hashes, conteos) y qué haría falta para no depender de él (remoto de DVC, instantáneas)?
2. ¿Por qué el hash de los CSV intermedios no sirve para comparar entre sistemas operativos y qué usamos en su lugar?
3. ¿Qué gana la operación con el alias `champion` y la exportación con SHA-256 frente a copiar un `.pkl`?
   ¿Cómo se promovería el challenger si el monitoreo aplica la regla R3?
4. ¿Qué ventajas y riesgos tiene versionar el modelo en Git (6.6 MB) frente a dejarlo solo en DVC o en un remoto?
   ¿Por qué el binario de la Fase 2 y el reentrenado no son idénticos aunque tienen los mismos hiperparámetros?
5. Si llegara enero de 2026, ¿qué pasos del ciclo (datos → Registry → servicio) habría que repetir y cuáles ya están
   automatizados?
6. ¿Qué mejorarías de la etapa de datos y del ciclo de vida del modelo si el proyecto siguiera?

## Actividades por fase

### Fase 1
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas con lo que hiciste como DE y MLE (descarga, limpieza, MLflowDataVersioner...). -->

### Fase 2
<!-- ESCRIBE AQUÍ: 2 a 4 viñetas (datos 2025, estructura Cookiecutter c13bc0c, revisión del Registry...). -->

### Fase final
<!-- ESCRIBE AQUÍ: qué revisaste y analizaste de los resultados de la tabla de arriba. -->

## Análisis

<!-- ESCRIBE AQUÍ: 2 o 3 párrafos respondiendo las preguntas guía con datos de la tabla. -->

## Reflexión

<!-- ESCRIBE AQUÍ: un párrafo (8 a 12 líneas): lecciones aprendidas y áreas de mejora desde tu rol. -->
