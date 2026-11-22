# Mensaje para el grupo (fase final)

Copia y pega el bloque de abajo en WhatsApp. Sustituye `[LIGA DEL WORD COMPARTIDO]` y `[LIGA DEL PR]`.

---

🚲 *Fase final MiBici – Equipo 19*

✅ *Ya está listo (todo en la rama `fase-final`)*
• Modelo champion `models:/mibici-demand-forecaster/1` reentrenado en limpio: MAE de prueba 0.7651 (Fase 2: 0.7656, −0.06 %)
• Pruebas con pytest: 270 aprobadas, cobertura 93 % (`pytest -q`)
• API FastAPI: `POST /predict`, `/health`, `/model-info`; las 500 predicciones de referencia salen idénticas
• Docker publicado: https://hub.docker.com/r/gusrivasmna/mibici-demand-service (tags 1.0.0, 1.0.0-model-v1, latest); en Linux difiere solo 1 ulp (8.9e-16) y pasa con tolerancia 1e-9
• Data drift: 8 escenarios + ventanas semanales; alerta real desde el 20 de diciembre (WAPE +38.8 %)
• DVC: `dvc.yaml` + `dvc.lock` (5 etapas)
• Reporte en Word generado con todo lo anterior (≈40 páginas)

⚠️ *NO tienen que ejecutar nada.* Los datos, MLflow y el modelo no están en GitHub; solo van a leer resultados y escribir.

✍️ *Lo que falta de cada quien*
1. Abrir tu archivo en la rama `fase-final` y llenar donde dice `<!-- ESCRIBE AQUÍ -->` (ya trae tus resultados y preguntas guía):
   • Antonio (DE/MLE) → `reports/fase_final/antonio.md` (datos, DVC, Registry → servicio)
   • Jesús (DS) → `reports/fase_final/jesus.md` (resultados del modelo y drift)
   • Erick (SWE) → `reports/fase_final/erick.md` (pruebas, cobertura, API, CI)
   • Gustavo (SRE) → revisar el borrador de `reports/fase_final/gustavo.md`
2. En el Word compartido: [LIGA DEL WORD COMPARTIDO]
   • Tu celda de la Tabla "Actividades por rol y fase" (sección 3.1, en amarillo)
   • Tu párrafo de reflexión (sección 6.5, en amarillo): pega la sección "Reflexión" de tu .md
   • Confirmar la matriz RACI (3.3) y la declaración de IA (sección 8)

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
(`TU_NOMBRE` = antonio, jesus, erick o gustavo)

📄 *Entrega*
• Cuando estén los 4 commits, Gustavo vuelve a generar el Word (toma sus reflexiones de los .md), exporta `Entrega_Final_Equipo19.pdf` (con ligas a GitHub y DockerHub) y lo sube a Canvas.
• PR de `fase-final` a `main`: [LIGA DEL PR] → Antonio hace el merge cuando pase el CI.

⏰ Fecha límite para su commit: ____
