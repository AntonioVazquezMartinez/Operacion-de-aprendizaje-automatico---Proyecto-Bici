# Fase 2 - Gustavo Alejandro Vázquez Rivas (A01611559)

## Rol: Site Reliability Engineer (SRE)

**Actividades en la Fase 2:** Entorno reproducible (requirements.txt, environment.yml,
pyproject.toml, Makefile), verificación del pipeline en un clon limpio y gestión de la
base de MLflow.

## Verificación de reproducibilidad en un clon limpio

Fecha: 8 de noviembre de 2026.

Pasos ejecutados desde cero, en un equipo distinto al de desarrollo:

```bash
git clone https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici.git
git checkout fase2-modelado
python -m venv .venv            # entorno virtual nuevo, sin paquetes previos
pip install -r requirements.txt
pip install -e .
python -m pytest -v
```

| Componente   | Versión |
|--------------|---------|
| Sistema      | Windows 11 Pro |
| Python       | 3.13.1 |
| scikit-learn | 1.9.1 |
| MLflow       | 3.18.0 |
| Optuna       | 5.0.0 |
| pandas       | 3.1.1 |
| NumPy        | 2.5.4 |

**Resultado:** `17 passed, 28 warnings in 36.80s`. Se aprobaron todas las pruebas,
incluida la prueba de integración que entrena, registra en un Model Registry
temporal y vuelve a cargar el modelo champion. El reporte las ejecutó con
Python 3.12, y en esta verificación también funcionaron con Python 3.13.

### Hallazgos para la siguiente fase

- `requirements.txt` solo define versiones mínimas (`>=`), así que un clon nuevo
  instala las versiones más recientes. Ya aparecen avisos de obsolescencia:
  `TargetEncoder.random_state` dejará de existir en scikit-learn 1.11, y pandas 3.x
  y SQLAlchemy también muestran advertencias. Propongo generar un archivo de
  versiones exactas (`pip freeze > requirements-lock.txt`) para reproducir el
  entorno con exactitud, y ejecutar las pruebas en cada push con GitHub Actions.
- En Windows, la instalación de `jupyter` falla si el entorno virtual está en una
  ruta muy larga (límite de 260 caracteres). Conviene crear el entorno en una ruta
  corta o habilitar las rutas largas del sistema.

## Reflexión

Esta fase me mostró que la reproducibilidad no se logra con buenas intenciones,
sino con mecanismos concretos. Al pasar de un mes a un año de datos aparecieron
problemas que con un solo archivo eran invisibles: codificaciones distintas entre
meses, comillas mal cerradas en julio y el bloqueo del portal a la descarga
automática. Un pipeline que "funciona en mi máquina" con un mes puede fallar en otro
entorno con doce. Por eso valoro tener las dependencias declaradas en
requirements.txt y environment.yml, una semilla única inyectada desde config.yaml y
pruebas automatizadas que confirman que el mismo código produce las mismas
predicciones. El cambio de formato de serialización en MLflow 3.18 fue un buen
ejemplo de cómo una actualización de una librería puede romper el flujo sin tocar
nuestro código. También entendí el Model Registry como una herramienta de operación
y no solo de registro: los alias champion y challenger permiten cambiar el modelo en
producción o regresar a una versión anterior sin modificar el código que lo consume.
Para la siguiente fase, mi prioridad será automatizar estas verificaciones con
GitHub Actions, fijar las versiones exactas de las dependencias y empezar a
monitorear el data drift, para que la reproducibilidad y la confiabilidad no
dependan de una revisión manual.
