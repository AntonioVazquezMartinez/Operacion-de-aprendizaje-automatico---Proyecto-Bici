# Fase 2, roles y reflexión de José Antonio Vázquez Martínez (A01797208)

## Mis roles en esta fase

### Data Engineer

* Pasamos de usar un solo mes (mayo de 2024) a usar todo 2025, que son 4.52 millones de viajes.
* Revisé la limpieza con todos los datos y se quedó el 97.36 % de los viajes. Las reglas de calidad siguen siendo las mismas. Quitamos los duplicados, los viajes con duración negativa, los de menos de 1 minuto, los de más de 2 horas y los que usan estaciones que no están en el catálogo.
* Los archivos CSV y la base de MLflow no se suben a GitHub porque pesan mucho. En el README explicamos cómo descargar los datos a mano.

### Machine Learning Engineer

* Conecté el pipeline con MLflow Tracking para comparar los modelos que probamos con Optuna.
* Revisé la comparación de modelos y el Model Registry. HistGradientBoosting quedó como v1 @champion, con un MAE de prueba de 0.766 y un R² de 0.69. RandomForest quedó como v2 @challenger.
* Comprobé que la predicción por lotes carga el modelo champion desde el Registry.

## Reflexión

Al pasar de 1 mes a 1 año de datos entendí que cuidar la calidad de los datos importa más que el modelo. Con 4.52 millones de viajes aparecieron casos que antes no se veían, y por eso es mejor dejar las reglas en un archivo de configuración y no escritas dentro del código.

También aprendí que fue buena idea quitar el rezago de 1 hora. Con ese dato el modelo se veía muy bien, pero usaba información que en la vida real no tendríamos a tiempo, así que el resultado no se iba a poder repetir en producción.

Con MLflow me quedó claro que guardar cada corrida, con sus parámetros, sus métricas y la huella de los datos, nos permite repetir un resultado y explicar por qué ganó un modelo. El Model Registry con los nombres champion y challenger deja claro cuál modelo se usa y cuál queda de respaldo, sin depender de que alguien se acuerde de un archivo.
