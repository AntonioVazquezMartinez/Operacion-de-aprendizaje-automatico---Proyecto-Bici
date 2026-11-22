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
* Como DE hice la descarga y la limpieza de los datos (downloader y cleaner) y dejé las versiones raw e interim en MLflow.
* Como MLE hice el `MLflowDataVersioner`, que guarda el SHA-256 de cada versión de los datos, y el flujo de punta a punta.

### Fase 2
* Pasamos de un mes a todo 2025 (4.52 millones de viajes) y revisé la limpieza con todos los datos: se quedó el 97.36 % de los viajes.
* Conecté el pipeline con MLflow Tracking para comparar los modelos de Optuna.
* Revisé el Model Registry: HistGradientBoosting quedó como v1 @champion (MAE de prueba 0.766, R² 0.69) y RandomForest como v2 @challenger.
* Comprobé que la predicción por lotes carga el champion desde el Registry.

### Fase final
* Datos: revisé que el portal republicó los 12 archivos (4,532,032 viajes, 11,937 más que en la Fase 2), que la retención sigue en 97.36 % y que el catálogo `nomenclatura_2026_09.csv` ya no existe y se usa el de 2026_10.
* Hashes y DVC: revisé que el hash del dataset cambió (`d00671f9` contra `fec850d2…`) y que `dvc.yaml` (5 etapas) y `dvc.lock` dejan registrado con qué datos se entrenó.
* Modelo: revisé que el Registry mantiene el mismo champion (v1) y challenger (v2), que se exporta a `models/champion/` con su SHA-256 (`ca4331a4…`) y 500 predicciones de referencia, y que la API verifica ese hash al arrancar.
* Modelo en Git y binario de la Fase 2: revisé que `models/champion/` (unos 6.6 MB) ya está en Git y que el binario de la Fase 2 reproduce sus 1,000 predicciones de referencia.

## Análisis

El riesgo de que el portal republique archivos y retire catálogos sin avisar es que no podamos explicar por qué cambian las métricas. Esta vez lo detectamos con los conteos (4,532,032 viajes contra 4,520,095) y con el hash del dataset (`d00671f9` contra `fec850d2…`). Para no depender del portal haría falta guardar instantáneas de los CSV en un remoto de DVC, porque hoy `dvc.lock` solo guarda la huella. El hash de los CSV intermedios no sirve para comparar entre Windows y Linux porque cambia por el fin de línea (CRLF contra LF) aunque el contenido sea el mismo. Por eso comparamos el modelo por su SHA-256 y sus 500 predicciones de referencia, que salen idénticas en Windows y con una diferencia de 1 ulp en Linux.

Con el alias `champion` sabemos cuál modelo se usa sin depender de que alguien se acuerde de un archivo. La exportación con SHA-256 hace que la API detecte un modelo alterado y no lo sirva. Meter `models/champion/` en Git (unos 6.6 MB) tiene la ventaja de que `docker build` funciona desde un clon limpio, y el riesgo es que el repositorio crezca si el modelo se reentrena seguido. El binario de la Fase 2 y el reentrenado no son idénticos porque se entrenaron con datos distintos del portal y con otra versión de Python (3.12 contra 3.13), aunque tienen los mismos hiperparámetros y su correlación es de 0.9989.

Si llegara enero de 2026 habría que volver a descargar los datos, correr el pipeline de DVC, revisar el Registry y reexportar el champion. La exportación, la API, la imagen y las pruebas ya están automatizadas. Si el proyecto siguiera, agregaría el remoto de DVC y automatizaría el reentrenamiento en CI.

## Reflexión

Lo que más me quedó de esta fase es que los datos cambian aunque nosotros no toquemos nada. El portal republicó los mismos 12 archivos y pasamos de 4,520,095 a 4,532,032 viajes, y el hash del dataset ya no era el de la Fase 2. Sin los hashes habríamos pensado que cambió el código. Por eso me parece mejor versionar con SHA-256 y con DVC que confiar en que el CSV sea el mismo, y verificar el modelo y no los archivos intermedios, porque su hash cambia entre Windows y Linux por el fin de línea. Del lado del modelo entendí que el Registry solo no basta para llevarlo a producción. En la Fase 2 el champion vivía en la base local de uno de nosotros. Ahora se exporta con su SHA-256 y 500 predicciones de referencia, la API revisa ese hash al arrancar y el modelo ya está en Git, así que cualquiera construye la imagen desde un clon limpio. Como mejora, esta vez Gustavo hizo la parte técnica y yo me quedé con la revisión y el análisis. Nos ayudó a coordinarnos y, aunque practiqué menos mi rol, de igual manera aprendí revisando el trabajo de los demás. Lo que sigue faltando es un remoto de DVC para los datos, porque hoy el modelo está en Git pero los CSV dependen de que el portal siga publicándolos.
