"""Monitoreo del modelo en producción: métricas de drift, escenarios simulados, reglas de alerta y reporte.

Módulos:
    metrics    métricas puras de drift (PSI, KS, Wasserstein, faltantes, categorías no vistas, perfil horario).
    scenarios  generadores de conjuntos de monitoreo con distribución alterada.
    rules      umbrales, niveles OK / AVISO / ALERTA y acción propuesta.
    figures    gráficas del monitoreo (07 a 10 en reports/figures).
    report     escritura de drift_report.json / .md y drift_summary.csv.
    drift      orquestador y CLI: python -m src.monitoring.drift
"""
