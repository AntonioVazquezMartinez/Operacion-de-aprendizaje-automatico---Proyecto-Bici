# Fase 2 - Rol y reflexión: Erick Francisco Siller Ojeda (A01382929)

## Rol: Software Engineer (SWE)

Mi responsabilidad en la Fase 2 fue la calidad del software: revisar que la estructura y la
configuración del proyecto cumplieran lo que el reporte afirma y que el comportamiento del
pipeline quedara protegido por pruebas automatizadas. Para que la participación sea
verificable, separo lo que revisé de lo que cambié en el repositorio.

### Lo que revisé

La migración a Cookiecutter Data Science v2, `config.yaml`, `src/config.py`, los módulos por
etapa y la interfaz de línea de comandos con `--config` los implementó Jesús Ángel Rincón Ruiz
(commit `93586e4`). Mi revisión de ese trabajo consistió en comprobar, contra el código, que:

- ningún módulo de `src/` conserva rutas, meses, umbrales, semilla ni hiperparámetros fijos;
  todo se lee de `config.yaml` y las rutas se resuelven desde la raíz del repositorio;
- cada módulo se ejecuta como CLI y acepta otra configuración sin tocar código;
- el notebook `1.0-eq19-eda-mibici.ipynb` solo importa clases de `src/` y muestra resultados;
- la suite existente cubre limpieza, cuadrícula estación-hora, ausencia de rezagos menores a
  24 h, partición cronológica, preprocesador ajustado solo con train, reproducibilidad con
  semilla y la integración con MLflow y el Model Registry.

### Lo que cambié (commit `6213844`, rama `fase2-modelado`)

- **Prueba de días festivos corregida.** La aserción sobre `is_holiday` estaba dentro de un
  `if` que nunca se cumplía, porque el 1 de mayo cae en las 168 h de historial que se
  descartan; la prueba pasaba sin verificar nada. Ahora se comprueba sobre una cuadrícula
  construida a propósito y espera exactamente `[1, 1, 0]`.
- **Dos pruebas nuevas de `read_month`.** Lectura de un CSV mensual en latin-1 con una fila
  mal formada (se conserva todo menos esa fila) y rechazo de un encabezado inesperado. Ese
  camino de lectura, que resolvió las incidencias reales de 2025, no tenía ninguna prueba.
- **`ModelTrainer` registra la configuración efectiva.** La corrida comparativa subía a MLflow
  el archivo `config.yaml` tal cual estaba en disco, a partir de una ruta que `run()` recibía
  como argumento redundante. Ahora serializa la configuración cargada, incluidas las
  sobrescrituras, y `run()` no necesita argumentos. La prueba de integración verifica que el
  artefacto exista.
- Con esto la suite pasa de 17 a 20 pruebas; `README.md` lista la carpeta `reports/fase2/`.

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
dos módulos. Finalmente, la incidencia que el equipo documentó con la serialización en
MLflow 3.18 me recordó que las dependencias también son parte del sistema, y que la prueba
de integración, que entrena y vuelve a cargar el modelo registrado, es el lugar donde ese tipo
de ruptura debe detectarse antes de que llegue al reporte. Para la Fase 3, desde el rol de software, propongo
convertir la validación del encabezado de cada CSV mensual en una prueba de contrato sobre el
esquema de entrada, y activar el linter ya configurado en `pyproject.toml` (ruff) como paso
previo a las pruebas, para que el estilo y los errores básicos se detecten antes de ejecutar
el pipeline.
