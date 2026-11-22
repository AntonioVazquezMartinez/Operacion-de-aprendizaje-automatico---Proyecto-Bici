# Mensaje para el grupo (fase final)

Copia y pega el bloque de abajo en WhatsApp. Sustituye `[LIGA DEL WORD COMPARTIDO]` y `[LIGA DEL PR]`.

---

🚲 *Fase final MiBici – Equipo 19*

🔀 *Integración hecha:* las dos aproximaciones del equipo (binario de la Fase 2 en `entrega-final` y reentrenamiento de verificación en `fase-final`) ya están integradas en `fase-final`, con lo mejor de cada una (feature store, modelo en Git, registro de predicciones y CI con build vienen de `entrega-final`).

✅ *Ya está listo (todo en `fase-final`)*
• Modelo `models:/mibici-demand-forecaster/1`: MAE de prueba 0.7651 (Fase 2: 0.7656)
• Pruebas: 305 aprobadas, cobertura 94 % (`pytest -q`)
• API 1.1.0: `POST /predict` con rezagos opcionales (feature store), `/health`, `/model-info`
• Imagen 1.1.0 publicada: https://hub.docker.com/r/gusrivasmna/mibici-demand-service (tags 1.1.0, 1.1.0-model-v1, latest; 1.0.0 sigue publicada). Se construye desde un clon limpio y pasó en Windows, Linux, DockerHub y GitHub Actions
• Data drift: 8 escenarios + ventanas semanales; alerta real desde el 20 de diciembre
• DVC: `dvc.yaml` + `dvc.lock`
• Reporte en Word generado con todo lo anterior

⚠️ *NO tienen que ejecutar nada.* Solo leer resultados y escribir.

✍️ *Lo que falta de cada quien*
1. Abrir tu archivo en la rama `fase-final` y llenar donde dice `<!-- ESCRIBE AQUÍ -->` (ya trae tus resultados y preguntas guía):
   • Antonio (DE/MLE) → `reports/fase_final/antonio.md` (datos, DVC, modelo en Git, Registry → servicio, binario de la Fase 2)
   • Jesús (DS) → `reports/fase_final/jesus.md` (aproximación del binario de la Fase 2 en `entrega-final`, feature store, Podman, el contraste de aproximaciones, resultados del modelo y drift)
   • Erick (SWE) → `reports/fase_final/erick.md` (305 pruebas, prueba de coherencia, API 1.1.0, job docker del CI)
2. En el Word compartido: [LIGA DEL WORD COMPARTIDO]
   • Tu celda de la Tabla "Actividades por rol y fase" (sección 3.1, en amarillo)
   • Tu párrafo de reflexión (sección 6.5, en amarillo): pega la sección "Reflexión" de tu .md

💻 *Comandos para subir tu archivo* (en la carpeta del repo):
```
git fetch origin
git checkout fase-final
git pull origin fase-final
git add reports/fase_final/TU_NOMBRE.md
git commit -m "Fase final: análisis y reflexión de TU_NOMBRE"
git pull origin fase-final
git push origin fase-final
```
(`TU_NOMBRE` = antonio, jesus o erick)

📄 *Entrega*
• Gustavo ya no tiene pendientes de contenido: cuando estén sus 3 commits, abre el PR de `fase-final` a `main`, vuelve a generar el Word (toma sus reflexiones de los .md), exporta `Entrega_Final_Equipo19.pdf` (con ligas a GitHub y DockerHub) y lo sube a Canvas.
• PR de `fase-final` a `main`: [LIGA DEL PR] → Antonio hace el merge cuando pase el CI. El PR #5 (`entrega-final`) ya quedó integrado en `fase-final`.

⏰ Fecha límite para su commit: ____
