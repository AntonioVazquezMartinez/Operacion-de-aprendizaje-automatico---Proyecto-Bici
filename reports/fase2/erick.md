# Fase 2 - Rol y reflexión: Erick Francisco Siller Ojeda (A01382929)

## Rol: Software Engineer (SWE)

En la Fase 2 mi responsabilidad fue la calidad del software: que el proyecto tuviera una
estructura estándar, que ningún módulo dependiera de valores fijos y que el comportamiento
del pipeline quedara protegido por pruebas automatizadas (evidencia en `reports/fase2/erick.md`).

Actividades realizadas o revisadas en esta fase:

- **Migración a Cookiecutter Data Science v2.** Revisé que la estructura siguiera la
  convención actual de la plantilla: un módulo por etapa (`src/dataset.py`,
  `src/features.py`, `src/plots.py`, `src/tracking.py`, `src/modeling/`), notebook con
  nombre `1.0-eq19-eda-mibici.ipynb` que importa `src/` en lugar de duplicar lógica,
  `pyproject.toml` para instalar el paquete en modo editable y `Makefile` con un comando
  por etapa.
- **Configuración centralizada en `config.yaml`.** Validé que rutas, meses, reglas de
  limpieza, semilla, proporciones de la partición, experimentos de MLflow, modelos,
  hiperparámetros fijos y espacios de búsqueda de Optuna vivan en un solo archivo y se lean
  con `src/config.py`. Cada módulo se ejecuta como CLI y acepta `--config`, por lo que un
  experimento distinto no requiere tocar código.
- **Suite de pruebas (`unittest`).** Revisé las pruebas de limpieza (anomalías
  inyectadas en cantidades conocidas), características (rezagos solo con información
  pasada, sin rezago de 1 h), partición cronológica, preprocesador ajustado solo con
  train, construcción de cada modelo desde `config.yaml`, reproducibilidad con la semilla
  y la prueba de integración que entrena, registra en un Model Registry temporal y vuelve a
  cargar el `champion`.
- **Ajustes incluidos en este commit tras la revisión de código:**
  - `ModelTrainer` registra en MLflow la configuración *efectiva* (con sobrescrituras)
    como artefacto de la corrida comparativa, en lugar de copiar el archivo original;
    `run()` ya no necesita la ruta del YAML como argumento y la prueba de integración
    verifica que el artefacto `config.yaml` exista.
  - La prueba de días festivos verifica `is_holiday` de forma determinista sobre una
    cuadrícula construida a propósito; antes la aserción dependía de que el 1 de mayo
    sobreviviera al descarte de las primeras 168 h de historial y nunca se ejecutaba.
  - Dos pruebas nuevas de `read_month`: lectura de un CSV mensual en latin-1 con una
    fila mal formada (se conserva todo menos esa fila) y rechazo de un encabezado
    inesperado. Con esto la suite pasa de 17 a 20 pruebas.

## Reflexión

Pasar de un mes a un año completo de datos cambió el tipo de problemas que aparecen. Con
un mes, el código "funcionaba" aunque tuviera rutas y umbrales fijos; con doce archivos
mensuales aparecieron casos reales que ningún valor fijo contemplaba: un mes con
codificación distinta, otro con comillas mal cerradas y un catálogo cuya descarga quedó
bloqueada. La lección para mí como Software Engineer es que la configuración externa
(`config.yaml`) y las pruebas automatizadas no son un requisito de la rúbrica, sino la única
forma de que el pipeline siga siendo el mismo cuando los datos cambian: ampliar el periodo
fue agregar meses a una lista, y cada incidencia se convirtió en una regla de lectura con su
prueba.

También aprendí que MLflow y el Model Registry resuelven un problema de ingeniería de
software, no solo de ciencia de datos: el modelo deja de ser un archivo `.pkl` que alguien
tiene en su máquina y pasa a ser una versión con hash del dataset, hiperparámetros,
métricas y un alias (`champion`, `challenger`) que el código de predicción consume por
nombre. Eso desacopla a quien entrena de quien predice, igual que una interfaz desacopla a
dos módulos. Finalmente, el cambio de versión de MLflow (3.18) que rompió la serialización
nos recordó que las dependencias también son parte del sistema y que la prueba de
integración, que entrena y vuelve a cargar el modelo registrado, fue la que detectó el
problema antes de que llegara al reporte. Para la Fase 3, desde el rol de software, propongo
convertir la validación del encabezado de cada CSV mensual en una prueba de contrato sobre el
esquema de entrada, y activar el linter ya configurado en `pyproject.toml` (ruff) como paso
previo a las pruebas, para que el estilo y los errores básicos se detecten antes de ejecutar
el pipeline.
