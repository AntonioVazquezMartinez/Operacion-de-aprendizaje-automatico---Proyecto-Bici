# Fase final - Análisis y reflexión de José Antonio Vázquez Martínez (A01797208)

> Plantilla: escribe solo donde dice `<!-- ESCRIBE AQUÍ -->` (borra el comentario al escribir). No necesitas
> ejecutar nada: todos los resultados ya están en el repositorio. La sección **Reflexión** se copia tal cual
> al reporte final cuando vuelve a generarse, así que redáctala en primera persona y en párrafos.

## Rol y responsabilidad en la fase final

**Roles:** Data Engineer y ML Engineer.

En esta fase la ejecución técnica la hizo Gustavo (SRE) en una sola máquina con los datos, MLflow y Docker.
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
| Modelo → servicio | La API verifica el SHA-256 al arrancar; las 500 predicciones salen idénticas en Windows y con diferencia de 1 ulp (8.9e-16) en Linux | `docs/api.md`, `reports/fase_final/reproducibilidad/*.json` |
| DVC | `dvc.yaml` + `dvc.lock` sin remoto externo (si aún no está en el repo, revísalo cuando se agregue) | `dvc.yaml`, `dvc.lock` |

## Preguntas guía

1. ¿Qué riesgo operativo implica que el portal de MiBici republique CSV y retire catálogos sin avisar? ¿Cómo lo
   detectamos (hashes, conteos) y qué haría falta para no depender de él (remoto de DVC, instantáneas)?
2. ¿Por qué el hash de los CSV intermedios no sirve para comparar entre sistemas operativos y qué usamos en su lugar?
3. ¿Qué gana la operación con el alias `champion` y la exportación con SHA-256 frente a copiar un `.pkl`?
   ¿Cómo se promovería el challenger si el monitoreo aplica la regla R3?
4. Si llegara enero de 2026, ¿qué pasos del ciclo (datos → Registry → servicio) habría que repetir y cuáles ya están
   automatizados?
5. ¿Qué mejorarías de la etapa de datos y del ciclo de vida del modelo si el proyecto siguiera?

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
