"""Genera el reporte final en Word (reports/Entrega_Final_Equipo19.docx) a partir de los artefactos del repo.

Todas las cifras se leen en tiempo de ejecución (reports/report_data.py): métricas de los CSV y JSON, conteos de
los logs de reports/fase_final/, resultados del drift de reports/drift/drift_report.json, verificación entre
entornos de reports/fase_final/reproducibilidad/*.json y actividad de los integrantes de `git log --all`.
Lo que todavía no existe como artefacto aparece como [PENDIENTE: ...] resaltado en amarillo; al volver a
ejecutar el script, el marcador se sustituye por el valor en cuanto el artefacto exista.

Las reflexiones individuales se toman de la sección "## Reflexión" de reports/fase_final/<nombre>.md cuando el
integrante ya la escribió; si no, se deja un texto guía en amarillo.

Uso:
    python reports/build_report.py                      # genera el .docx (y actualiza el índice con Word si existe)
    python reports/build_report.py --no-word            # sin abrir Word (el índice se actualiza con F9)
    python reports/build_report.py --pdf ruta/al.pdf    # además exporta a PDF con Word (Windows)
    python reports/build_report.py --readme             # además llena la fila de GitHub Actions del README
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_data as rd  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.shared import Inches, Pt  # noqa: E402
from docx_helpers import (Ctx, Guide, Link, Pend, bullet, code, figure, heading, new_document, orientation,  # noqa: E402
                          page_number_footer, para, set_update_fields_on_open, table, toc)

ROOT = rd.ROOT
OUT = ROOT / "reports" / "Entrega_Final_Equipo19.docx"
FIG = rd.FIGS
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
OK_FILL, WARN_FILL, ALERT_FILL = "D9F2E3", "FFF1CC", "F8D7DA"
LEVEL_FILL = {"OK": OK_FILL, "AVISO": WARN_FILL, "ALERTA": ALERT_FILL}


# ------------------------------------------------------------------------------------------- formato
def f(x, d=4):
    return "s/d" if x is None else f"{x:,.{d}f}"


def n(x):
    return "s/d" if x is None else f"{int(x):,}"


def pc(x, d=1, sign=True):
    return "s/d" if x is None else (f"{100 * x:+.{d}f} %" if sign else f"{100 * x:.{d}f} %")


def rel(a, b):
    return None if a is None or b in (None, 0) else a / b - 1


def fecha_larga(d: date) -> str:
    return f"{DIAS[d.weekday()]} {d.day} de {MESES[d.month - 1]} del {d.year}"


def val(D, *path):
    cur = D
    for p in path:
        if cur is None:
            return None
        cur = cur.get(p) if isinstance(cur, dict) else None
    return cur


def member_section(key: str, section: str) -> str | None:
    """Texto de una sección (## ...) del archivo del integrante, sin comentarios; None si sigue vacía."""
    text = rd.read_text(rd.FF / f"{key}.md")
    if not text:
        return None
    m = re.search(rf"^## {section}\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        return None
    body = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S).strip()
    body = re.sub(r"^>.*$", "", body, flags=re.M).strip()
    return body or None


# --------------------------------------------------------------------------------- figuras derivadas
def fase2_image(name: str, tmp: Path) -> Path | None:
    """Extrae una imagen del reporte de la Fase 2 (capturas de MLflow que no están como archivo en el repo)."""
    src = ROOT / "reports" / "Fase2_Equipo19.docx"
    try:
        with zipfile.ZipFile(src) as z:
            data = z.read(f"word/media/{name}")
    except (OSError, KeyError):
        return None
    out = tmp / name
    out.write_bytes(data)
    return out


# ------------------------------------------------------------------------------------------- portada
def cover(ctx: Ctx, D, report_date: date, tmp: Path) -> None:
    doc = ctx.doc
    logo = fase2_image("image1.png", tmp)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if logo:
        p.add_run().add_picture(str(logo), width=Inches(3.2))
    for text, size, bold, space in (
            ("INSTITUTO TECNOLÓGICO Y DE ESTUDIOS SUPERIORES DE MONTERREY", 12, True, 6),
            ("Maestría en Inteligencia Artificial Aplicada", 12, False, 30),
            ("Entrega final: Sistema MLOps para la predicción de demanda horaria por estación en MiBici "
             "Guadalajara - Equipo 19", 18, True, 10),
            ("Integración de las Fases 1, 2 y final: pruebas, servicio, reproducibilidad, contenedor y "
             "monitoreo de data drift", 12, False, 24),
            (fecha_larga(report_date), 12, False, 24)):
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_after = Pt(space)
        r = q.add_run(text)
        r.font.size, r.bold = Pt(size), bold
    t = doc.add_table(rows=len(rd.MEMBERS) + 1, cols=3)
    t.alignment = 1
    for i, (a, b, c) in enumerate([("Integrante", "Matrícula", "Rol MLOps")] +
                                  [(m["name"], m["id"], m["role"]) for m in rd.MEMBERS]):
        for j, v in enumerate((a, b, c)):
            cell = t.rows[i].cells[j]
            cell.width = Inches([2.9, 1.2, 2.4][j])
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(v)
            r.font.size = Pt(11)
            r.bold = i == 0
    for text, link in (("Operaciones de aprendizaje automático | TC5061", None),
                       ("Repositorio: ", rd.REPO_URL), ("Imagen de DockerHub: ", rd.DOCKERHUB_URL)):
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_before = Pt(10 if link is None else 2)
        q.paragraph_format.space_after = Pt(2)
        r = q.add_run(text)
        r.font.size = Pt(11 if link else 12)
        if link:
            from docx_helpers import add_hyperlink
            add_hyperlink(q, link, link, size=9.5)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# -------------------------------------------------------------------------------------- contenido
def build(ctx: Ctx, D, report_date: date, tmp: Path) -> None:
    meta = D["meta"] or {}
    met = meta.get("metrics", {})
    cur = D["models"]["current"] or {}
    f2 = D["models"]["fase2"] or {}
    hgb, rf, poi, dum = (cur.get(k, {}) for k in ("hist_gradient_boosting", "random_forest", "poisson", "dummy_mean"))
    hgb2 = f2.get("hist_gradient_boosting", {})
    logs, pt, api, rep, dock, dvc, drift, eda, req = (D["logs"], D["pytest"], D["api"], D["repro"], D["docker"],
                                                     D["dvc"], D["drift"] or {}, D["eda"] or {}, D["req"])
    model_uri = meta.get("model_uri")
    uri = model_uri or Pend("URI del modelo (models/champion_metadata.json)")
    test_mae, test_mae2 = hgb.get("test_mae"), hgb2.get("test_mae")
    base = val(drift, "baseline", "performance") or {}
    scen = {s["name"]: s for s in drift.get("scenarios", [])}
    windows = drift.get("windows", [])
    envs = rep.get("envs", [])
    ok_envs = [e for e in envs if e.get("verdict") == "PASS"]
    gh = D["git"]
    cfg = D.get("config") or {}
    max_batch = val(cfg, "serving", "max_batch_size")
    raw_row = rd.comp_row(D, "Registros brutos") or ["", "s/d", "s/d", "s/d"]
    hours_row = rd.comp_row(D, "Estaciones activas") or ["", "s/d", "s/d", "s/d"]
    hours = hours_row[2].split("x")[-1].strip() if "x" in hours_row[2] else "s/d"
    rows_train = val(meta, "data_period", "rows_train")
    pyv = val(meta, "library_versions", "python") or "s/d"
    ref_rows = val(drift, "reference", "rows")
    fs = meta.get("feature_store") or {}
    fsw = fs.get("predictable_window") or {}
    svc_ver = val(cfg, "serving", "service_version") or "s/d"
    f2c = rep.get("fase2_cmp") or {}
    f2w, f2l = f2c.get("windows") or {}, f2c.get("linux") or {}
    f2vs = f2w.get("fase2_vs_retrained_on_reference") or {}
    hub_tags = dock.get("hub_tags") or {}
    t110, t100 = hub_tags.get("1.1.0") or {}, hub_tags.get("1.0.0") or {}
    merge = next((x for x in gh.get("commits", []) if x["merge"] and "entrega-final" in x["subject"]), None)
    merge_ref = f"merge {merge['hash']}" if merge else "merge de entrega-final"
    clon = next((e for e in envs if e["_file"].endswith("/clon_limpio.json")), None)
    envs_v1 = rep.get("envs_v1", [])

    def fs_path(e):
        """Camino /predict sin rezagos (feature store) de un JSON de verificación."""
        return val(e, "feature_store", "predict_feature_store") or {}

    cover(ctx, D, report_date, tmp)

    # --------------------------------------------------------------------------------- índice
    p = para(ctx, ("b", "Índice"), align=WD_ALIGN_PARAGRAPH.LEFT, size=14)
    toc(ctx.doc)
    ctx.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # --------------------------------------------------------------------------------- resumen
    heading(ctx, "Resumen ejecutivo", 1)
    para(ctx, "MiBici, el sistema de bicicletas públicas del Área Metropolitana de Guadalajara, rebalancea sus "
              "estaciones de forma reactiva: cuando una estación ya está vacía o llena, una camioneta acude a "
              "corregirla. En este proyecto construimos, a lo largo de tres fases, un sistema de aprendizaje "
              "automático que estima cuántas bicicletas se extraerán de cada estación en cada hora con 1 a 24 horas "
              "de anticipación, y lo operamos con prácticas de MLOps: datos versionados, experimentos trazables, "
              "modelo registrado, pruebas automatizadas, servicio REST, contenedor publicado y monitoreo de drift. "
              "La Tabla ", ctx.ref("t:clave"), " resume los resultados verificables de la entrega final; cada cifra "
              "proviene del artefacto indicado en la última columna.")
    rows = [
        ["Modelo en producción", [("c", model_uri or ""), Pend("metadatos") if not model_uri else None,
                                  f" (HistGradientBoosting, pérdida Poisson, alias {meta.get('alias', '')})"],
         "models/champion_metadata.json"],
        ["Error en prueba (nov-dic 2025, usada una sola vez)",
         f"MAE {f(test_mae)} bicicletas/hora; RMSE {f(hgb.get('test_rmse'))}; R² {f(hgb.get('test_r2'))}",
         "reports/model_comparison.csv"],
        ["Mejora contra la referencia ingenua", f"MAE {pc(-rel(test_mae, dum.get('test_mae')), 1, False)} menor que el de la media "
                                                 f"({f(dum.get('test_mae'))})", "reports/model_comparison.csv"],
        ["Reentrenamiento en entorno limpio vs. Fase 2", f"MAE de prueba {f(test_mae)} vs. {f(test_mae2)} "
                                                        f"({pc(rel(test_mae, test_mae2), 2)})",
         "reports/fase_final/comparacion_fase2.md"],
        ["Pruebas automatizadas", f"{n(pt.get('passed'))} pruebas aprobadas con pytest -q; cobertura de "
                                  f"{pt.get('coverage', 's/d')} % de src/" if pt.get("passed") else
         Pend("log de pytest"), "reports/fase_final/logs/10_pytest_cobertura.log"],
        ["Servicio", f"FastAPI {svc_ver}: POST /predict (rezagos opcionales: sin ellos los calcula el feature "
                     "store), POST /predict/raw, GET /health, GET /model-info; validación con Pydantic v2",
         "docs/api.md, docs/openapi.json"],
        ["Reproducibilidad del artefacto",
         f"{len(ok_envs)} entornos con veredicto PASS (500 predicciones con rezagos y 200 sin rezagos), incluido un "
         "clon limpio y la imagen descargada de DockerHub; el binario de la Fase 2 reproduce sus 1,000 predicciones"
         if envs else Pend("verificación entre entornos"), "reports/fase_final/reproducibilidad/"],
        ["Contenedor", [f"Imagen {rd.DOCKER_IMAGE} (", Link("DockerHub", rd.DOCKERHUB_URL), "): ",
                        ", ".join(sorted(hub_tags)) if hub_tags else Pend("tags publicados"),
                        "; se construye desde un clon limpio porque el modelo está en Git"],
         "docs/docker.md, dockerhub_tags.json"],
        ["Trabajo en equipo", f"Dos aproximaciones complementarias (binario de la Fase 2 en entrega-final y "
                              f"reentrenamiento de verificación en fase-final), contrastadas requisito por requisito e "
                              f"integradas en fase-final ({merge_ref})",
         "sección 2.5, git log"],
        ["Data drift", f"{len(scen)} escenarios y {len(windows)} ventanas semanales; el monitoreo detecta el "
                       f"cambio real de las vacaciones de diciembre", "reports/drift/drift_report.md"],
    ]
    table(ctx, "t:clave", "Resultados clave de la entrega final y artefacto de origen", ["Aspecto", "Resultado",
                                                                                        "Evidencia"],
          rows, widths=[1.7, 3.2, 1.6], size=9)

    # ============================================================================ 1. INTRODUCCIÓN
    heading(ctx, "1. Introducción", 1, page_break=True)
    para(ctx, "Este documento integra las tres fases del proyecto de la materia y presenta la solución completa. "
              "La Fase 1 (25 de octubre de 2026) construyó la base del pipeline con un mes de datos: descarga, "
              "limpieza, ingeniería de características, versionado de datos en MLflow y dos modelos de referencia. "
              "La Fase 2 (8 de noviembre de 2026) amplió los datos a todo 2025, reorganizó el repositorio con "
              "Cookiecutter Data Science v2, parametrizó todo en config.yaml y llevó el entrenamiento a un pipeline "
              "de scikit-learn con Optuna, MLflow Tracking y Model Registry. La fase final, objeto principal de este "
              "reporte, convierte ese modelo en un servicio operable y verificable: pruebas unitarias y de "
              "integración con pytest, una API con FastAPI, reproducibilidad demostrada en otro entorno, una imagen "
              "Docker publicada en DockerHub y un estudio de data drift con umbrales, alertas y acciones. Para verificar "
              "los resultados, el equipo trabajó la fase final con dos aproximaciones complementarias en ramas "
              "separadas y después las integró en una sola solución (sección 2.5).")
    heading(ctx, "1.1 Problemática y Machine Learning Canvas", 2)
    para(ctx, "El problema de negocio no cambió desde la Fase 1: en horas pico (8:00 y 18:00 en días laborales) las "
              "estaciones céntricas se vacían o se saturan, y el rebalanceo con camionetas llega tarde. Si el operador "
              "conoce con anticipación cuántas bicicletas saldrán de cada estación en cada hora, puede programar las "
              "rutas de redistribución antes del desabasto. Estructuramos el problema con el Machine Learning Canvas "
              "(Dorard, 2020). La Tabla ", ctx.ref("t:canvas"), " presenta el Canvas actualizado: la columna central "
              "conserva lo planteado en la Fase 1 y la derecha describe lo que realmente construimos, de modo que se "
              "ve qué decisiones se sostuvieron y cuáles cambiaron con la evidencia.")
    canvas = [
        ["Propuesta de valor", "Predecir la demanda por estación y hora con 1 a 24 h de anticipación para "
                               "rebalancear antes del desabasto.",
         "Igual. El modelo entrega extracciones esperadas por estación-hora; el servicio responde por lote "
         f"(hasta {n(max_batch)} instancias por petición, serving.max_batch_size)."],
        ["Fuentes de datos", "Viajes mensuales de MiBici y catálogo de estaciones.",
         f"12 CSV de 2025 ({n(logs.get('raw_rows'))} viajes leídos) y catálogo nomenclatura_2026_10 "
         f"({n(logs.get('catalog_stations'))} estaciones); calendario de festivos de la LFT (art. 74)."],
        ["Ingesta", "Lotes mensuales; data/raw inmutable; hashes en MLflow.",
         "Descarga automática con las URL de config.yaml (el portal exige User-Agent de navegador); "
         "SHA-256 y mlflow.log_input en el experimento MiBici_Datos; DVC para datos y modelo."],
        ["Variables", "Calendario, cíclicas, estación, rezagos de 1 h y 24 h, promedio móvil.",
         "Calendario cíclico, fin de semana, festivo, ubicación y capacidad, rezagos de 24 h y 168 h y media de la "
         "misma hora en 7 días. Se eliminó el rezago de 1 h (fuga respecto del horizonte)."],
        ["Tarea de ML", "Regresión de conteos (demand_count).", "Igual: regresión de conteos no negativos con "
                                                               "pérdida de Poisson."],
        ["Modelos", "Poisson/Ridge como base; Random Forest, XGBoost, LightGBM.",
         "Dummy (media), PoissonRegressor, RandomForest e HistGradientBoosting (implementación de scikit-learn "
         "inspirada en LightGBM, sin dependencias adicionales). Champion: HistGradientBoosting."],
        ["Decisiones operativas", "Enviar camioneta si la demanda > 80 % del inventario; retirar si la estación "
                                  "está al 90 %.",
         "Se mantiene como regla de uso propuesta; no se dispone de datos de inventario en tiempo real para "
         "validarla (limitación reconocida, sección 6.2)."],
        ["Inferencia", "Lotes cada 60 min; < 5 s para toda la red; FastAPI en Podman/Docker.",
         f"API FastAPI {svc_ver} en contenedor Docker (Podman se usó en la rama entrega-final y sus comandos están "
         "documentados); el cliente puede omitir los rezagos dentro de la ventana del feature store; latencias en "
         "20_api_smoke.log."],
        ["Evaluación offline", "Partición cronológica 70/15/15; MAE, RMSE, R².",
         "Igual, más devianza de Poisson; el conjunto de prueba se usa una sola vez y se agrega WAPE en el "
         "monitoreo."],
        ["Monitoreo", "KS o Wasserstein sobre variables temporales y de clima; degradación del MAE; "
                      "reentrenamiento automático.",
         "PSI, KS y Wasserstein sobre las variables del modelo (no hay variables de clima), calidad de datos, MAE y "
         "WAPE contra línea base; acción propuesta por regla. El reentrenamiento es manual."],
        ["Restricciones", "Privacidad, resiliencia ante fallas del portal, reproducibilidad.",
         "Usuario_Id no se usa; dependencias con versión exacta, semillas, imagen con digest fijo y verificación "
         "del SHA-256 del modelo al arrancar."],
    ]
    table(ctx, "t:canvas", "Machine Learning Canvas: planteamiento de la Fase 1 y solución construida",
          ["Bloque", "Fase 1 (planteamiento)", "Solución final (construida)"], canvas, widths=[1.2, 2.2, 3.1],
          size=8.5, note=["el Canvas completo y los cambios anotados están en reports/ML_Canvas_MiBici.md."])
    para(ctx, "Del análisis de los datos y del Canvas identificamos las características del problema que guiaron la "
              "solución. La Tabla ", ctx.ref("t:caract"), " las relaciona con la decisión técnica que provocaron; "
              "son el hilo conductor de los métodos de la sección 4.", keep=True)
    zero = eda.get("matrix_zero_share_pct")
    caract = [
        ["Conteos no negativos con muchos ceros", f"{zero} % de las combinaciones estación-hora tienen demanda cero; "
                                                 f"media {eda.get('matrix_mean_demand')} viajes/hora"
         if zero is not None else Pend("resumen del EDA"),
         "Pérdida de Poisson (enlace logarítmico): predicciones no negativas y varianza proporcional a la media."],
        ["Horizonte de 1 a 24 h", "Al predecir con 24 h de anticipación no se conoce la demanda de la hora previa",
         "Solo rezagos ≥ 24 h (24 h, 168 h y media de 7 días). El rezago de 1 h se eliminó tras la "
         "retroalimentación del profesor (fuga de información)."],
        ["Serie temporal con estacionalidad", "Picos 8:00 y 18:00, caída en fin de semana y festivos, "
                                             "temporada decembrina",
         "Partición cronológica 70/15/15, codificación cíclica (seno/coseno), indicadores de fin de semana y "
         "festivo; monitoreo por ventanas semanales."],
        ["Cientos de estaciones heterogéneas", f"{n(meta.get('num_stations'))} estaciones activas; unas cuantas del "
                                              "centro concentran la demanda",
         "TargetEncoder con validación cruzada interna (en lugar de one-hot) y variables de ubicación y capacidad."],
        ["Fuente externa que cambia", "Codificaciones distintas, comillas mal cerradas, catálogo que se reemplaza "
                                      "y CSV republicados",
         "Lectura tolerante con validación de encabezado, hashes SHA-256, versionado de datos y verificación del "
         "artefacto en lugar de confiar en reentrenar."],
        ["Etiquetas con retraso", "La demanda real de una hora se conoce después de que ocurre",
         "Reglas de alerta que funcionan sin etiquetas (calidad y drift de entradas) y se confirman con el "
         "desempeño cuando llegan."],
    ]
    table(ctx, "t:caract", "Características del problema y decisiones que guiaron la solución",
          ["Característica", "Evidencia", "Decisión"], caract, widths=[1.5, 2.3, 2.7], size=8.5,
          note=["porcentaje de ceros y media calculados por reports/report_data.py sobre la matriz procesada "
                "(caché en reports/fase_final/eda_resumen.json)."])

    heading(ctx, "1.2 Por qué MLOps en este caso", 2)
    para(ctx, "La justificación de MLOps no es genérica: se desprende de cuatro rasgos concretos del caso, todos "
              "observados en el proyecto.")
    bullet(ctx, ("b", "Los datos llegan cada mes y cambian. "), "MiBici publica un CSV mensual y, además, republica "
           "archivos anteriores: con los mismos doce nombres de archivo leímos ",
           n(logs.get("raw_rows")), f" viajes en la fase final contra {raw_row[1]} en la Fase 2, y el catálogo "
           "nomenclatura_2026_09 dejó de estar publicado. Sin versionado y hashes no podríamos explicar por qué cambian "
           "las métricas (sección 4.4).")
    bullet(ctx, ("b", "Hay drift estacional demostrado. "), "En las ventanas semanales del conjunto de prueba el WAPE "
           "sube de forma abrupta a partir del 20 de diciembre (sección 4.12). Un modelo entrenado una vez y "
           "desplegado sin monitoreo degradaría su utilidad justo en las semanas atípicas.")
    bullet(ctx, ("b", "Las decisiones de rebalanceo deben poder auditarse. "), "Cada predicción del servicio indica "
           "qué versión del modelo la produjo (", ("c", model_uri or "models:/..."), "), y esa versión enlaza con la "
           "corrida de MLflow, sus hiperparámetros y el hash del dataset de entrenamiento.")
    bullet(ctx, ("b", "El modelo debe operar fuera de la máquina de quien entrenó. "), "En la Fase 2 el modelo "
           "champion solo existía en la base local de un integrante. La fase final lo exporta como artefacto "
           "autocontenido con SHA-256, lo versiona en Git, lo sirve con una API y lo empaqueta en una imagen que "
           "cualquiera puede construir desde un clon limpio o descargar, y verificar contra las predicciones de "
           "referencia.")
    para(ctx, "Estos rasgos corresponden a la deuda técnica que describen Sculley et al. (2015): dependencias de "
              "datos inestables, configuración dispersa y ausencia de monitoreo. Las prácticas que adoptamos siguen "
              "el ciclo de vida de MLOps descrito por Treveil et al. (2020) y Kreuzberger et al. (2023).")

    heading(ctx, "1.3 Diagrama de la solución", 2)
    para(ctx, "La Figura ", ctx.ref("f:arq"), " muestra la arquitectura de punta a punta. Se lee en cuatro carriles: "
              "datos, modelado y registro, servicio y despliegue, y monitoreo con retroalimentación; la banda inferior "
              "agrupa las herramientas transversales. Cada bloque indica el módulo del repositorio, la herramienta y "
              "el rol responsable. La Tabla ", ctx.ref("t:herr"), " lista las versiones exactas de las herramientas, "
              "tomadas de requirements.txt.")

    tools = [
        ["Lenguaje y entorno", f"Python {pyv} (venv); dependencias con versión exacta (==) en requirements.txt y "
                               f"requirements-serve.txt", "SRE"],
        ["Datos", f"pandas {req.get('pandas')}, NumPy {req.get('numpy')}, requests {req.get('requests')}", "DE"],
        ["Modelado", f"scikit-learn {req.get('scikit-learn')}, Optuna {req.get('optuna')}, SciPy {req.get('scipy')}",
         "DS / MLE"],
        ["Seguimiento y registro", f"MLflow {req.get('mlflow')} (SQLite + Model Registry), cloudpickle "
                                   f"{req.get('cloudpickle')}", "MLE"],
        ["Versionado", f"Git/GitHub; DVC {req.get('dvc')}", "DE / SWE"],
        ["Pruebas", f"pytest {req.get('pytest')}, pytest-cov {req.get('pytest-cov')}, httpx {req.get('httpx')}",
         "SWE"],
        ["Servicio", f"FastAPI {req.get('fastapi')}, Pydantic {req.get('pydantic')}, uvicorn {req.get('uvicorn')}",
         "SWE"],
        ["Contenedor y CI", "Docker (base python:3.13.9-slim con digest fijo), DockerHub, GitHub Actions", "SRE"],
        ["Monitoreo y reporte", f"SciPy (KS, Wasserstein), Matplotlib {req.get('matplotlib')}, python-docx "
                                f"{req.get('python-docx')}", "DS / SRE"],
    ]
    table(ctx, "t:herr", "Herramientas de software y versiones fijadas", ["Capa", "Herramienta y versión",
                                                                          "Rol"], tools, widths=[1.5, 4.2, 0.8],
          size=9)

    orientation(ctx, True)
    figure(ctx, "f:arq", FIG / "00_arquitectura.png", "Arquitectura de la solución MLOps con componentes, "
           "herramientas y roles", ["generada con scripts/build_architecture_diagram.py. La flecha roja punteada es "
                                    "la retroalimentación propuesta (reentrenar y promover el challenger) cuando el "
                                    "monitoreo aplica la regla R3. ",
                                    "DVC versiona el pipeline completo (dvc.yaml y dvc.lock)." if dvc["yaml"] else
                                    Pend("dvc.yaml y dvc.lock aún no están en el repositorio")], width=9.0)
    orientation(ctx, False)

    # ============================================================================ 2. ACTIVIDADES
    heading(ctx, "2. Actividades realizadas en cada fase", 1, page_break=True)
    para(ctx, "Esta sección describe qué se hizo en cada fase, con qué entregables y cómo se atendió la "
              "retroalimentación. La sección 3 detalla los roles y la Tabla ", ctx.ref("t:fases"),
         " al final de esta sección consolida fase, actividad, rol, herramienta y evidencia.")
    heading(ctx, "2.1 Fase 1: procesamiento y modelado inicial", 2)
    para(ctx, "Trabajamos con un mes de datos (mayo de 2024, 410,806 viajes) para construir el esqueleto del "
              "pipeline. Implementamos la descarga desde el portal, seis reglas de limpieza (duplicados, duraciones "
              "no positivas, viajes menores a 1 minuto y mayores a 2 horas, estaciones fuera del catálogo y edades "
              "fuera de 14 a 90 años) que conservaron el 96.32 % de los registros, el análisis exploratorio, la "
              "cuadrícula estación-hora con rezagos de 1 y 24 horas, un preprocesador con escalado, TargetEncoder y "
              "PCA, y dos modelos de referencia registrados en MLflow (Dummy y regresión de Poisson). Todas las "
              "versiones de datos quedaron en MLflow con su hash SHA-256. Las cifras provienen del reporte "
              "reports/Fase 1_Equipo19.docx.")
    heading(ctx, "2.2 Fase 2: gestión del proyecto de Machine Learning", 2)
    para(ctx, "Ampliamos el periodo a los doce meses de 2025, migramos el repositorio a Cookiecutter Data Science v2, "
              "trasladamos rutas, reglas, semilla, modelos y espacios de búsqueda a config.yaml, y convertimos cada "
              "etapa en un módulo con interfaz de línea de comandos. El entrenamiento pasó a un Pipeline de "
              "scikit-learn por modelo, ajustado con Optuna y registrado en MLflow Tracking; los tres candidatos se "
              "registraron en el Model Registry con alias champion y challenger. El champion de la Fase 2 obtuvo "
              f"MAE de prueba {f(test_mae2)} (reports/fase_final/model_comparison_fase2_referencia.csv).")
    heading(ctx, "2.3 Fase final: del modelo registrado a un servicio verificable", 2)
    para(ctx, "La fase final atendió los cinco puntos de la entrega y los hallazgos que dejamos documentados en la "
              "Fase 2. En orden de ejecución:")
    steps = [
        ("Entorno fijado y reentrenamiento en limpio. ", "Versiones exactas en requirements.txt, semillas globales, "
         "búsqueda de Optuna sin corte por tiempo y descarga automática de los 12 meses; reentrenamiento completo en "
         f"un venv nuevo ({logs.get('train_seconds', 's/d')} s de entrenamiento) y comparación cifra por cifra "
         "contra la Fase 2."),
        ("Exportación del champion. ", "Carpeta MLflow autocontenida (models/champion), metadatos con SHA-256 del "
         "modelo y del dataset, y 500 predicciones de referencia como patrón oro."),
        ("Pruebas. ", f"Migración de unittest a pytest con fixtures, marcador de integración y cobertura: de "
         f"{pt.get('legacy_passed', 's/d')} pruebas heredadas a {n(pt.get('passed'))}."),
        ("Servicio. ", "API FastAPI con validación Pydantic, errores uniformes, OpenAPI y colección de Postman; "
         f"en la versión {svc_ver}, rezagos opcionales calculados con un feature store y registro de predicciones."),
        ("Contenedor y reproducibilidad. ", "Dockerfile multi-etapa, modelo versionado en Git para construir la imagen "
         "desde un clon limpio, verificación del servicio contra la referencia en distintos entornos, reentrenamiento "
         "completo en un contenedor Linux y comparación con el binario original de la Fase 2."),
        ("Dos aproximaciones complementarias. ", "Una sirve el binario original de la Fase 2 (rama entrega-final) y "
         "otra reentrena el modelo en un entorno limpio para verificar la reproducibilidad y complementar la solución "
         "(rama fase-final); se contrastaron y se integraron en fase-final (sección 2.5)."),
        ("Monitoreo de drift. ", "Ocho escenarios (incluido un control), PSI, KS, Wasserstein, calidad de datos, "
         "MAE y WAPE contra línea base, umbrales en config.yaml y acción propuesta por regla."),
        ("Documentación y reporte. ", "README, docs/, diagrama de la solución y este reporte generado por código."),
    ]
    for a, b in steps:
        bullet(ctx, ("b", a), b)
    heading(ctx, "2.4 Retroalimentación atendida entre fases", 2)
    fb = [
        ["Un mes de datos no basta para el modelo (prof., F1)", "Se usan los 12 meses de 2025; agregar un mes es "
                                                               "editar data.months", "F2",
         "config.yaml"],
        ["Rezago de 1 h incompatible con el horizonte (prof., F1)", "Se eliminó; solo rezagos ≥ 24 h", "F2",
         "config.yaml (features.lag_hours), tests/unit/test_features.py"],
        ["Incluir áreas de mejora y estrategias (prof., F1)", "Secciones de mejora en F2 y sección 6 de este "
                                                             "reporte", "F2, FF", "Sección 6"],
        ["requirements.txt con >= (hallazgo SRE, F2)", "Versiones exactas (==) y requirements-lock.txt", "FF",
         "requirements*.txt"],
        ["Corte por tiempo de Optuna no determinista (F2)", "timeout_seconds: null y registro de "
                                                           "tuning_trials_completed", "FF", "config.yaml, 04_train.log"],
        ["Descarga manual por HTTP 406 (F2)", "Descarga automática con User-Agent y URL en config.yaml", "FF",
         "config.yaml (data.trip_urls)"],
        ["Servir con FastAPI en contenedor (estrategia F2)", "API y Dockerfile; Docker en lugar de Podman", "FF",
         "src/serving, Dockerfile"],
        ["Monitoreo con Evidently y umbral de MAE (estrategia F2)", "Implementación propia de PSI/KS/Wasserstein, "
                                                                   "MAE y WAPE, sin dependencia adicional", "FF",
         "src/monitoring"],
        ["Versionar los CSV con DVC (estrategia F2)", ["dvc.yaml + dvc.lock sin remoto externo ",
                                                      None if dvc["yaml"] else Pend("configuración de DVC")], "FF",
         "dvc.yaml, dvc.lock"],
        ["Pruebas en cada push con GitHub Actions (estrategia F2)", "Workflow tests.yml en push a main y "
                                                                   "fase-final", "FF", ".github/workflows/tests.yml"],
    ]
    table(ctx, "t:feedback", "Retroalimentación y hallazgos de fases anteriores, y cómo se atendieron",
          ["Observación (origen)", "Acción", "Fase", "Evidencia"], fb, widths=[2.1, 2.3, 0.6, 1.8], size=8.5)
    para(ctx, "Dos estrategias propuestas en la Fase 2 no se adoptaron tal cual, y lo explicamos: usamos Docker en "
              "lugar de Podman porque la entrega pide Docker y DockerHub, y GitHub Actions ejecuta Docker de forma "
              "nativa; e implementamos las métricas de drift en lugar de usar Evidently para controlar exactamente "
              "los umbrales, los empates y las reglas sin agregar una dependencia pesada a la imagen.")
    fases = [
        ["Fase 1", "Ingesta y limpieza", "DE", "pandas, requests", "src/dataset.py (antes src/data/)"],
        ["Fase 1", "EDA y variables", "DS", "Jupyter, Matplotlib", "notebooks/, reports/figures/01-05"],
        ["Fase 1", "Estructura y pruebas unittest", "SWE", "Cookiecutter, unittest", "tests/"],
        ["Fase 1", "Entorno y MLflow local", "SRE", "venv, SQLite, MLflow", "requirements.txt, environment.yml"],
        ["Fase 1", "Versionado de datos", "MLE / DE", "MLflow, SHA-256", "src/tracking.py"],
        ["Fase 2", "Datos 2025 y lectura robusta", "DE", "pandas", "src/dataset.py"],
        ["Fase 2", "Rediseño de variables y modelos", "DS", "scikit-learn, Optuna", "config.yaml, src/features.py"],
        ["Fase 2", "Cookiecutter v2 y módulos configurables", "SWE", "Cookiecutter v2, config.yaml", "src/"],
        ["Fase 2", "Tracking y Model Registry", "MLE", "MLflow", "src/modeling/train.py"],
        ["Fase 2", "Verificación en clon limpio", "SRE", "venv, pytest", "reports/fase2/gustavo.md"],
        ["Final", "Pruebas unitarias e integración", "SWE (resp.)", "pytest, pytest-cov", "tests/, 10_pytest_cobertura.log"],
        ["Final", "API de predicción", "SWE (resp.)", "FastAPI, Pydantic", "src/serving, docs/api.md"],
        ["Final", "Exportación y ciclo del modelo", "MLE (resp.)", "MLflow", "src/modeling/export.py, models/"],
        ["Final", "Reproducibilidad y contenedor", "SRE", "Docker, GitHub Actions", "Dockerfile, reproducibilidad/"],
        ["Final", "Data drift", "DS / SRE (resp.)", "SciPy, Matplotlib", "src/monitoring, reports/drift/"],
        ["Final", "Versionado de datos y modelo", "DE (resp.)", "DVC", "dvc.yaml, dvc.lock"],
    ]
    fases += [
        ["Final", "Aproximación 1: servicio del binario de la Fase 2 con feature store, Podman y DockerHub, CI con "
                  "build", "DS / SRE", "FastAPI, Podman", "rama entrega-final (PR #5)"],
        ["Final", "Contraste e integración de las dos aproximaciones", "SRE / DS", "Git, pytest, Docker",
         f"{merge_ref}, sección 2.5"],
    ]
    table(ctx, "t:fases", "Resumen de actividades por fase, rol responsable, herramienta y evidencia",
          ["Fase", "Actividad", "Rol", "Herramienta", "Evidencia en el repositorio"], fases,
          widths=[0.6, 2.0, 1.0, 1.4, 2.0], size=8.5,
          note=["“resp.” indica el rol al que corresponde el entregable y que responde por su análisis y revisión "
                "(sección 3). En la fase final el equipo trabajó dos aproximaciones en ramas separadas y las integró "
                "(sección 2.5)."])

    heading(ctx, "2.5 Dos aproximaciones complementarias y su integración", 2)
    para(ctx, "Con una sola implementación no hay contra qué contrastar un resultado. Por eso el equipo trabajó la "
              "fase final con dos aproximaciones en ramas separadas. La primera (rama entrega-final, pull request #5) "
              "parte del binario original de la Fase 2 y lo sirve tal cual. La segunda (rama fase-final) reentrena el "
              "modelo desde los datos del portal en un entorno limpio para comprobar que el resultado de la Fase 2 se "
              f"reproduce (MAE de prueba {f(test_mae)} contra {f(test_mae2)}) y complementa la solución con "
              "verificación entre entornos, monitoreo y publicación de la imagen. Después contrastamos las dos "
              "requisito por requisito, ejecutamos las pruebas de ambas e integramos en fase-final lo que cada una "
              f"resolvía mejor ({merge_ref}, con estrategia ours: los commits de entrega-final quedan en el "
              "historial); las piezas traídas de entrega-final se adaptaron con sus propias pruebas. La Tabla ",
         ctx.ref("t:ramas"), " resume el contraste.", keep=True)
    ramas = [
        ["Rezagos en /predict", "Feature store con las últimas 360 h; rezagos opcionales y lags_source",
         "El cliente envía los tres rezagos",
         "Feature store, con rezagos opcionales y la misma función del entrenamiento"],
        ["Modelo en la imagen", "Binario de la Fase 2 dentro de Git en models/champion",
         "Modelo reentrenado fuera de Git (DVC); build con el modelo exportado localmente",
         "Modelo reentrenado en Git: docker build funciona en un clon limpio"],
        ["Contenedor", "Podman, Python 3.12, imagen angrinconr/mibici-demand-api",
         "Docker, python:3.13.9-slim por digest, imagen gusrivasmna/mibici-demand-service",
         "Imagen gusrivasmna/mibici-demand-service 1.1.0 y comandos de Podman documentados"],
        ["Integración continua", "Pruebas y construcción de la imagen desde el repositorio",
         "Pruebas con cobertura y verificación de la imagen publicada",
         "Ambas: job docker que construye y prueba la imagen, más la verificación de la publicada"],
        ["Pruebas", "Suite pytest con prueba de coherencia servicio-pipeline",
         "Suite pytest unitaria y de integración, cobertura por módulo",
         "Suite de fase-final más la prueba de coherencia (sin training-serving skew)"],
        ["Reproducibilidad", "1,000 predicciones del binario en Windows, Linux y contenedor Podman",
         "Reentrenamiento en limpio comparado con la Fase 2; 500 predicciones en varios entornos; reentrenamiento "
         "en Linux", "Ambas; el binario de la Fase 2 se conservó y se comparó con el reentrenado"],
        ["Monitoreo", "Escenarios simulados y monitoreo semanal con PSI/KS",
         "8 escenarios, PSI/KS/Wasserstein, calidad de datos, MAE y WAPE", "Monitoreo de fase-final"],
        ["Auditoría de predicciones", "PREDICTION_LOG (registro JSONL)", "No tenía",
         "Registro opcional de predicciones"],
    ]
    table(ctx, "t:ramas", "Contraste de las dos aproximaciones y lo que se integró en fase-final",
          ["Aspecto", "Aproximación 1: binario de la Fase 2 (entrega-final)",
           "Aproximación 2: reentrenamiento de verificación (fase-final)", "Solución integrada"], ramas,
          widths=[1.2, 1.7, 1.7, 1.9], size=8,
          note=["descripción de entrega-final tomada de su README (rama origin/entrega-final, sección Entrega Final) "
                "y de sus commits acf3df3 y 9110f58; la integración, en los commits posteriores al merge (git log)."])
    para(ctx, ("b", "Por qué fue útil. "), "Contrastar dos aproximaciones funcionó como verificación cruzada. El "
              "reentrenamiento confirmó que el resultado de la Fase 2 se reproduce y que el binario original da "
              "predicciones casi idénticas al reentrenado (sección 4.10). La primera aproximación mostró que el "
              "servicio puede calcular los rezagos con la misma función del entrenamiento en lugar de pedirlos al "
              "cliente, y que con el modelo en Git la imagen se construye en un clon limpio; la segunda aportó la "
              "verificación entre entornos con tolerancia justificada, el monitoreo más completo y la imagen fijada "
              "por digest. Integrarlas requirió adaptar las piezas de entrega-final al formato del modelo exportado y "
              "al servicio de fase-final, lo que anotamos como área de mejora (sección 6.2).")

    # ============================================================================ 3. ROLES
    heading(ctx, "3. Roles y metodología MLOps", 1, page_break=True)
    para(ctx, "Desde la Fase 1 adoptamos los cinco roles de MLOps con cuatro integrantes: José Antonio Vázquez cubre "
              "Data Engineer y ML Engineer. En las dos primeras fases cada integrante ejecutó las actividades de su "
              "rol. En la fase final el equipo trabajó dos aproximaciones en ramas separadas y las integró en "
              "fase-final (sección 2.5); cada rol responde por los entregables de su ámbito y documenta su análisis "
              "en su archivo reports/fase_final/<nombre>.md.")
    heading(ctx, "3.1 Actividades de cada rol por fase", 2)
    roles_rows = [
        ["Data Engineer\n(Antonio)", "Descarga y limpieza (downloader, cleaner); versiones raw/interim en MLflow.",
         "Ingesta de los 12 meses de 2025, codificaciones y registros mal formados; revisión de la limpieza "
         "(97.36 % de retención en la Fase 2).",
         ["Responsable del análisis y revisión de la etapa de datos: cambios del portal, retención, hashes, DVC. ",
          Guide("[Antonio: ajusta esta celda con lo que revisaste.]")]],
        ["Data Scientist\n(Jesús)", "ML Canvas, EDA, variables cíclicas y rezagos.",
         "Rediseño de variables (sin rezago de 1 h), algoritmos y espacios de Optuna; integró la migración a "
         "Cookiecutter v2 y config.yaml (commit 93586e4).",
         ["Diseño del servicio con feature store, simulación de drift y análisis de resultados del modelo. ",
          Guide("[Jesús: ajusta esta celda con tu análisis.]")]],
        ["Software Engineer\n(Erick)", "Estructura Cookiecutter, refactorización a POO, suite unittest, CI.",
         "Revisión de la migración, corrección de la prueba de festivos, pruebas de read_month, integración de main "
         "y CI.",
         ["Responsable del análisis y revisión de la calidad de software: pruebas, cobertura, API y CI. ",
          Guide("[Erick: ajusta esta celda con lo que revisaste.]")]],
        ["Site Reliability Engineer\n(Gustavo)", "requirements.txt, environment.yml, MLflow con SQLite.",
         "Entorno reproducible y verificación en un clon limpio; hallazgos de dependencias sin versión fija.",
         "Entorno fijado, reentrenamiento de verificación, contenedor, publicación de la imagen y verificación "
         "entre entornos; integración de las dos aproximaciones."],
        ["ML Engineer\n(Antonio)", "MLflowDataVersioner con SHA-256; flujo de punta a punta.",
         "Pipeline con MLflow Tracking, Model Registry con champion/challenger y predicción por lotes.",
         ["Responsable del análisis del ciclo de vida del modelo: Registry → exportación → servicio. ",
          Guide("[Antonio: ajusta esta celda.]")]],
    ]
    table(ctx, "t:roles", "Actividades por rol y fase", ["Rol (integrante)", "Fase 1", "Fase 2", "Fase final"],
          roles_rows, widths=[1.3, 1.6, 1.8, 1.8], size=8.5,
          note=["Fases 1 y 2 según reports/roles_mlops.md, los reportes de cada fase, reports/fase2/*.md y el historial "
                "de Git. Las celdas en amarillo las ajusta cada integrante."])
    heading(ctx, "3.2 Alineación de los roles con el ciclo de vida MLOps", 2)
    para(ctx, "La Figura ", ctx.ref("f:ciclo"), " ubica cada etapa del ciclo de vida (datos, experimentación, "
              "empaquetado, despliegue, monitoreo y retroalimentación) con los roles responsables y la fase en que se "
              "construyó. La Tabla ", ctx.ref("t:ciclo"), " detalla la responsabilidad de cada rol en cada etapa y "
              "los entregables que la evidencian.", keep=True)
    figure(ctx, "f:ciclo", FIG / "00b_ciclo_mlops.png", "Ciclo de vida MLOps del proyecto con roles y fases",
           ["generada con scripts/build_architecture_diagram.py. El primer rol de cada etapa es quien rinde cuentas "
            "(A en la matriz RACI)."], width=5.2)
    ciclo = [
        ["Datos", "DE (A/R), DS (C)", "Ingesta reproducible, limpieza, cuadrícula, hashes y DVC",
         "src/dataset.py, src/features.py, MiBici_Datos"],
        ["Experimentación", "DS (A/R), MLE (R)", "Variables, algoritmos, espacios de búsqueda, métricas, Optuna",
         "config.yaml, src/modeling/train.py"],
        ["Empaquetado", "MLE (A/R), SWE (R)", "Registry, exportación con SHA-256, pruebas, CI",
         "models/, tests/, .github/workflows/"],
        ["Despliegue", "SRE (A/R), SWE (R)", "API, contenedor, imagen versionada, verificación entre entornos",
         "src/serving, Dockerfile, DockerHub"],
        ["Monitoreo", "DS (A), SRE (R)", "Escenarios, métricas de drift, umbrales y alertas",
         "src/monitoring, reports/drift/"],
        ["Retroalimentación", "MLE (A), DE (R), SRE (C)", "Reentrenar, evaluar challenger y mover el alias; "
                                                           "corregir el pipeline de datos",
         "Reglas R1-R7 en reports/drift/"],
    ]
    table(ctx, "t:ciclo", "Responsabilidad de cada rol en las etapas del ciclo de vida",
          ["Etapa", "Roles", "Responsabilidad", "Evidencia"], ciclo, widths=[1.1, 1.3, 2.3, 1.8], size=8.5)
    heading(ctx, "3.3 Matriz RACI de la fase final", 2)
    raci = [
        ["Datos, hashes y DVC", "A", "I", "I", "R", "C"],
        ["Reentrenamiento y comparación con Fase 2", "C", "C", "I", "R", "A"],
        ["Exportación del champion y feature store", "I", "R", "C", "R", "A"],
        ["Pruebas pytest y cobertura", "I", "R", "A", "R", "C"],
        ["API FastAPI", "I", "R", "A", "R", "C"],
        ["Contenedor, DockerHub y reproducibilidad", "I", "R", "C", "A/R", "C"],
        ["Simulación de drift y umbrales", "C", "A/R", "I", "R", "C"],
        ["Contraste e integración de las dos aproximaciones", "I", "R", "C", "A/R", "C"],
        ["Análisis por rol (archivo .md)", "R", "R", "R", "R", "R"],
        ["Reporte final, PDF y PR a main", "C", "C", "C", "R", "A"],
    ]
    table(ctx, "t:raci", "Matriz RACI de la fase final", ["Actividad", "DE (Antonio)", "DS (Jesús)", "SWE (Erick)",
                                                          "SRE (Gustavo)", "MLE (Antonio)"], raci,
          widths=[2.3, 0.85, 0.85, 0.85, 0.85, 0.85], size=8.5, align=["l", "c", "c", "c", "c", "c"],
          note=["R = responsable de ejecutar, A = rinde cuentas y revisa el resultado, C = consultado, I = informado. "
                "En las filas técnicas hay dos R porque cada aproximación tuvo su propia implementación; quien tiene A "
                "revisa y analiza el resultado desde su rol. El dueño del repositorio (MLE) hace el merge del pull "
                "request final."])
    heading(ctx, "3.4 Flujo de colaboración", 2)
    para(ctx, "El trabajo se organiza en GitHub con una rama por fase (fase1-datos-reales, fase2-modelado, fase-final) "
              "y pull requests hacia main. Git no admite espacios en nombres de rama, por eso la rama de esta entrega "
              "es fase-final. Cada pull request ejecuta el workflow de pruebas (.github/workflows/tests.yml) en un "
              "runner Linux con Python 3.13; un segundo workflow (docker-reproducibilidad.yml) revisa el Dockerfile "
              "con hadolint y verifica la imagen publicada contra la referencia. Para evitar conflictos, cada "
              "integrante escribe su análisis en su propio archivo (reports/fase_final/antonio.md, jesus.md, "
              "erick.md y gustavo.md) y lo integra con un commit; José Antonio Vázquez, como dueño del repositorio, "
              "hace el merge del pull request final. Las dos aproximaciones de la fase final vivieron en ramas "
              f"separadas (entrega-final, con su pull request #5, y fase-final) y se integraron con un merge en "
              f"fase-final ({merge_ref}) que conserva el historial de ambas.")
    heading(ctx, "3.5 Evidencia de actividad en GitHub", 2)
    para(ctx, "La actividad de los cuatro integrantes queda registrada en el repositorio de GitHub: el historial de "
              "commits de cada rama, los pull requests de cada fase con su revisión e integración continua, y los "
              "archivos donde cada integrante documenta las actividades y el análisis de su rol en la Fase 2 y en la "
              "fase final. Las ligas son:", keep=True)
    for lab, url in (("Historial de commits de la rama de la entrega final", f"{rd.REPO_URL}/commits/fase-final"),
                     ("Pull requests del proyecto", f"{rd.REPO_URL}/pulls?q=is%3Apr"),
                     ("Análisis por integrante en la fase final (reports/fase_final/)",
                      f"{rd.REPO_URL}/tree/fase-final/reports/fase_final"),
                     ("Roles y reflexiones por integrante en la Fase 2 (reports/fase2/)",
                      f"{rd.REPO_URL}/tree/fase-final/reports/fase2")):
        bullet(ctx, f"{lab}: ", Link(url, url)).alignment = WD_ALIGN_PARAGRAPH.LEFT
    para(ctx, "La Tabla ", ctx.ref("t:prs"), " resume los pull requests de cada fase.", keep=True)
    prs = [[f"#{num}", Link(f"{rd.REPO_URL}/pull/{num}", f"{rd.REPO_URL}/pull/{num}"), branch, desc]
           for num, branch, desc in rd.PULL_REQUESTS]
    prs = [[a, [Link(f"pull/{a[1:]}", b.url)], c, d] for a, b, c, d in prs]
    final_pr = next((num for num, branch, _ in rd.PULL_REQUESTS if branch == "fase-final"), None)
    if final_pr is None:  # se resuelve al agregar el PR de fase-final a rd.PULL_REQUESTS
        prs.append(["final", [Pend("liga del PR de fase-final a main")], "fase-final",
                    "Entrega final: pruebas, API, Docker, reproducibilidad, drift y reporte"])
    table(ctx, "t:prs", "Pull requests del proyecto", ["PR", "Liga", "Rama", "Contenido"], prs,
          widths=[0.6, 1.4, 1.6, 2.9], size=8.5)

    # ============================================================================ 4. MÉTODOS
    heading(ctx, "4. Métodos, técnicas y resultados por etapa", 1, page_break=True)
    para(ctx, "Para cada etapa describimos el método, las alternativas que consideramos y por qué elegimos el "
              "método, el resultado con su artefacto de origen y un análisis de lo que significa para la operación "
              "de MiBici. Salvo que se indique lo contrario, las cifras corresponden al reentrenamiento de la fase "
              f"final (Windows 11, Python {pyv}, entorno limpio).")

    # 4.1 Ingesta y limpieza
    heading(ctx, "4.1 Ingesta y limpieza de datos", 2)
    table(ctx, "t:m-ingesta", "Ingesta y limpieza: método, alternativas y justificación",
          ["Decisión", "Alternativas", "Justificación"],
          [["Descarga programática con URL en config.yaml y User-Agent de navegador", "Descarga manual (Fase 2)",
            "Repetible por cualquiera; el portal responde 406 sin User-Agent. Se usan primero los CSV presentes."],
           ["Lectura tolerante: UTF-8 y latin-1, respaldo ante comillas mal cerradas, validación del encabezado",
            "Fallar ante el primer error; limpiar a mano", "Los meses reales traen codificaciones distintas; validar "
                                                           "el esquema evita mezclar columnas."],
           ["Seis reglas de negocio parametrizadas (60 s, 120 min, 14-90 años, catálogo)",
            "Eliminación de atípicos por percentiles", "Reglas explicables al operador; el p99 de duración es 31 min, "
                                                       "120 min solo quita anomalías."]],
          widths=[2.4, 1.6, 2.5], size=8.5)
    data_t = D["comp"].get("data")
    if data_t:
        table(ctx, "t:limpieza", "Limpieza de datos y matriz de modelado: Fase 2 contra reentrenamiento de la fase "
                                 "final", data_t["header"], data_t["rows"], widths=[2.6, 1.3, 1.5, 1.2], size=8.5,
              align=["l", "r", "r", "r"], note=["reports/fase_final/comparacion_fase2.md (sección 2), "
                                                "01_dataset.log y 02_features.log."])
    else:
        para(ctx, Pend("tabla de limpieza (reports/fase_final/comparacion_fase2.md)"))
    para(ctx, ("b", "Análisis. "), f"La retención se mantuvo en {logs.get('retention', 's/d')} % y casi todo lo descartado son "
              "viajes menores a un minuto (desanclajes fallidos), igual que en la Fase 1. El hallazgo relevante es "
              "que los mismos doce archivos del portal produjeron ",
         n(logs.get("raw_rows")), f" viajes ({raw_row[3]} respecto de la Fase 2), y que julio, que antes tenía comillas mal "
         "cerradas, hoy se lee con el lector estándar: el portal republica archivos sin avisar. Para la operación "
         "significa que un modelo no puede auditarse sin saber exactamente qué versión de los datos lo produjo; de ahí "
         "los hashes y el versionado de la sección 4.4. La matriz conserva las mismas 370 estaciones y el mismo número "
         "de filas, por lo que el efecto en el modelo es pequeño (sección 4.10).")

    # 4.2 EDA
    heading(ctx, "4.2 Análisis exploratorio", 2)
    para(ctx, "El EDA se hace con funciones de src/plots.py que regeneran las figuras 01 a 05 sobre los viajes limpios "
              "de 2025 (python -m src.plots); el notebook solo importa esas funciones. Elegimos figuras estáticas "
              "versionadas en lugar de un tablero interactivo porque son reproducibles y se pueden citar en el "
              "reporte. Las cifras de los pies se calculan sobre los mismos datos (reports/fase_final/eda_resumen.json).")
    ph = eda.get("peak_hours") or []
    figure(ctx, "f:eda1", FIG / "01_distribucion_horaria.png", "Distribución de viajes por hora del día (2025)",
           [f"picos a las {ph[0][0]}:00 ({n(ph[0][1])} viajes en el año) y {ph[1][0]}:00 ({n(ph[1][1])}); entre la 1:00 "
            f"y las 4:59 solo hay {eda.get('night_trips_1_to_4')} viajes porque el servicio no opera. Esta forma "
            "bimodal justifica la codificación cíclica de la hora." if len(ph) >= 2 else Pend("resumen del EDA")],
           width=5.0)
    figure(ctx, "f:eda2", FIG / "02_viajes_dia_semana.png", "Volumen de viajes por día de la semana (2025)",
           [f"en promedio {n(eda.get('weekday_avg_per_day'))} viajes por día laboral y {n(eda.get('weekend_avg_per_day'))} "
            f"en fin de semana ({eda.get('weekend_drop_pct')} % menos): el sistema se usa sobre todo para ir al "
            "trabajo o a la escuela, de ahí las variables de día de la semana, fin de semana y festivo."
            if eda else Pend("resumen del EDA")], width=5.0)
    top = eda.get("top_stations") or []
    figure(ctx, "f:eda3", FIG / "03_top_estaciones_demanda.png", "Las 10 estaciones con más extracciones (2025)",
           ["; ".join(f"{name} ({n(cnt)})" for _, name, cnt in top) + ". La demanda se concentra en el centro de "
            "Guadalajara; esta heterogeneidad motiva el TargetEncoder de la estación." if top else Pend("EDA")],
           width=5.0)
    figure(ctx, "f:eda4", FIG / "04_distribucion_duracion.png", "Duración de los viajes",
           [f"mediana de {eda.get('duration_median_min')} min y media de {eda.get('duration_mean_min')} min: "
            "trayectos cortos de última milla, coherentes con el límite de 120 min de la limpieza."
            if eda else Pend("EDA")], width=4.6)
    gs = eda.get("gender_share_pct") or {}
    figure(ctx, "f:eda5", FIG / "05_demografia_usuarios.png", "Perfil demográfico de los viajes",
           [f"hombres {gs.get('M')} %, mujeres {gs.get('F')} %, sin dato {gs.get('Desconocido')} %; edad mediana "
            f"{eda.get('age_median', 0):.0f} años (rango intercuartil {eda.get('age_q1', 0):.0f}-"
            f"{eda.get('age_q3', 0):.0f}). Estas variables no entran al modelo: la demanda se agrega por estación y "
            "hora y no se usan datos de usuarios individuales (privacidad)." if gs else Pend("EDA")], width=5.6)

    # 4.3 Características
    heading(ctx, "4.3 Ingeniería de características", 2)
    table(ctx, "t:m-feat", "Ingeniería de características: método, alternativas y justificación",
          ["Decisión", "Alternativas", "Justificación"],
          [["Cuadrícula completa estación × hora con ceros explícitos", "Solo horas con viajes",
            "El modelo debe aprender también las horas sin demanda (más de la mitad de la matriz)."],
           ["Rezagos de 24 h y 168 h y media de la misma hora en 7 días", "Rezago de 1 h (Fase 1), medias móviles "
                                                                         "recientes",
            "Disponibles con 24 h de anticipación: no hay fuga de información respecto del horizonte del Canvas."],
           ["Descartar las primeras 168 h sin historial", "Rellenar con ceros (Fase 1)",
            "Los ceros artificiales introducían demanda falsa al inicio de la serie."],
           ["Seno/coseno de hora y día; fin de semana; festivos de la LFT art. 74", "One-hot de la hora; sin "
                                                                                    "festivos",
            "Continuidad 23:00-0:00 con dos variables; los festivos explican caídas que el día de la semana no."],
           ["TargetEncoder para Origen_Id con validación cruzada interna", "One-hot (370 columnas); id numérico",
            "Captura el nivel de cada estación sin cientos de columnas y sin sobreajuste (Micci-Barreca, 2001)."],
           ["Sin PCA", "PCA al 95 % de varianza (Fase 1)", "En la Fase 1 empeoró el RMSE de prueba (3.659 vs. "
                                                          "2.595) y con 13 variables no aporta."]],
          widths=[2.2, 1.6, 2.7], size=8.5)
    para(ctx, ("b", "Análisis. "), f"La matriz final tiene {n(logs.get('matrix_rows'))} filas y {logs.get('matrix_cols', 's/d')} "
              f"columnas ({n(meta.get('num_stations'))} estaciones × {hours} horas). La decisión con más impacto fue "
              "eliminar el rezago de 1 hora: con él la Fase 1 lograba un MAE de 1.344 en un mes, pero ese dato no "
              "existe al momento de predecir con varias horas de anticipación, así que la métrica no se sostendría "
              "en operación. La prueba tests/unit/test_features.py verifica que ningún rezago use información "
              "posterior a t − 24 h, de modo que la regla queda protegida contra cambios futuros. El servicio deriva "
              "el calendario y las variables de estación con las mismas funciones de src/features.py, lo que elimina "
              "el riesgo de diferencias entre entrenamiento y servicio.")

    # 4.4 Versionado de datos
    heading(ctx, "4.4 Versionado de datos y trazabilidad", 2)
    table(ctx, "t:m-version", "Versionado: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                                 "Justificación"],
          [["SHA-256 de cada versión (raw, interim, processed) y mlflow.log_input", "Fechas o nombres de archivo",
            "El hash detecta cualquier cambio de contenido; vincula cada corrida al dataset exacto."],
           ["No subir los CSV como artefactos de MLflow (log_dataset_artifacts: false)", "Subir los CSV completos",
            "Con un año de datos duplicaba gigabytes; el hash ya garantiza la trazabilidad."],
           [["DVC con dvc.yaml y dvc.lock, sin remoto externo ", None if dvc["yaml"] else
             Pend("configuración de DVC")], "Git LFS; remoto en la nube",
            "dvc.lock registra la huella de datos y modelo por etapa en Git sin subir binarios ni exponer "
            "credenciales; un remoto queda como trabajo futuro."],
           ["Modelo exportado dentro de Git (models/champion, unos 6.6 MB) y salidas de export con cache: false en "
            "DVC", "Solo DVC sin remoto; remoto de DVC",
            "Con el modelo solo en DVC un clon limpio no puede construir la imagen; en Git es pequeño, auditable y "
            "llega al CI. Idea tomada de la rama entrega-final."]],
          widths=[2.4, 1.5, 2.6], size=8.5)
    para(ctx, ("b", "Análisis. "), "El versionado mostró su valor en esta fase: el SHA-256 del dataset procesado de la Fase 2 "
              "(visible en la captura del Model Registry, Figura ", ctx.ref("f:registry"), ", comienza con "
              "d00671f9) difiere del actual (", ("c", (meta.get("dataset_sha256") or "")[:16] + "…"), "), lo que "
              "confirmó con evidencia que los datos del portal cambiaron y no el código. Además, el reentrenamiento en "
              "Linux mostró que el hash de un CSV depende del fin de línea del sistema operativo (CRLF frente a LF) "
              "aunque el contenido sea idéntico; por eso verificamos el artefacto del modelo por su SHA-256 y sus "
              "predicciones, no por el hash de los CSV intermedios.",
         [" DVC (docs/dvc.md) declara el pipeline en dvc.yaml con ", str(len(dvc["stages"])), " etapas (",
          " → ".join(dvc["stages"]), "), cada una con su comando, código, datos y secciones de config.yaml como "
          "dependencias; dvc.lock guarda ", str(dvc.get("lock_entries") or "s/d"), " huellas MD5 de dependencias y "
          "salidas, incluidos los 12 CSV crudos y la carpeta del modelo, y ", ("c", "dvc status"), " responde: «",
          dvc.get("status") or "s/d", "». No configuramos un remoto: los datos son públicos y se descargan con las "
          "URL de config.yaml, el hash permite comprobar que son los mismos y el modelo se distribuye dentro de la "
          "imagen de DockerHub."] if dvc["stages"] else [" ", Pend("etapas de dvc.yaml y huellas de dvc.lock")])

    # 4.5 Modelado
    heading(ctx, "4.5 Modelado y ajuste de hiperparámetros", 2)
    table(ctx, "t:m-model", "Modelado: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                             "Justificación"],
          [["Cuatro modelos de complejidad creciente: Dummy (media), PoissonRegressor, RandomForest, "
            "HistGradientBoosting con pérdida Poisson", "XGBoost, LightGBM (Canvas F1); redes neuronales",
            "HGB implementa en scikit-learn la idea de LightGBM (Ke et al., 2017) sin dependencias nuevas; Dummy "
            "es el piso; RF contrasta bagging contra boosting (Breiman, 2001; Friedman, 2001)."],
           ["Pérdida y GLM de Poisson para conteos", "Error cuadrático", "Predicciones no negativas y varianza que "
                                                                        "crece con la media (Nelder y Wedderburn, "
                                                                        "1972)."],
           ["Optuna con TPE y semilla fija; 20 trials (5 en RF) sobre 300,000 filas", "Búsqueda en malla o "
                                                                                      "aleatoria",
            "TPE concentra la búsqueda en regiones prometedoras (Bergstra et al., 2011; Akiba et al., 2019); sin "
            "corte por tiempo para que sea determinista."],
           ["Selección por MAE de validación; reentrenar con train + val y evaluar test una sola vez",
            "Elegir con test; validación cruzada aleatoria", "Evita sesgo optimista y fuga temporal; MAE es "
                                                             "interpretable en bicicletas por hora."]],
          widths=[2.4, 1.5, 2.6], size=8.5)
    mrows, fills = [], {}
    labels = {"hist_gradient_boosting": "HistGradientBoosting (Poisson)", "random_forest": "RandomForest",
              "poisson": "PoissonRegressor", "dummy_mean": "Dummy (media)"}
    reg = {"hist_gradient_boosting": "v1 · champion", "random_forest": "v2 · challenger", "poisson": "v3",
           "dummy_mean": "no se registra"}
    for k, lab in labels.items():
        r = cur.get(k)
        if r:
            mrows.append([lab, f(r["val_mae"]), f(r["test_mae"]), f(r["test_rmse"]), f(r["test_r2"]),
                          f(r["test_poisson_deviance"]), reg[k]])
    fills = {(0, i): OK_FILL for i in range(7)}
    table(ctx, "t:modelos", "Comparación de modelos (validación y prueba)", ["Modelo", "MAE val.", "MAE prueba",
                                                                            "RMSE prueba", "R² prueba",
                                                                            "Devianza Poisson", "Registry"],
          mrows, widths=[1.8, 0.75, 0.8, 0.8, 0.75, 0.85, 1.05], size=8.5,
          align=["l", "r", "r", "r", "r", "r", "l"], fills=fills,
          note=["reports/model_comparison.csv (corrida 00_Comparativa_Modelos de MiBici_Fase2_Modelos). Trials "
                "completados: " + ", ".join(f"{m} {a}/{b}" for m, a, b in logs.get("trials", [])) + " (04_train.log)."])
    figure(ctx, "f:mae", FIG / "06_comparacion_modelos.png", "MAE de prueba por modelo",
           ["generada por src/modeling/train.py y registrada en MLflow."], width=5.0)
    hp = meta.get("hyperparameters", {})
    para(ctx, ("b", "Resultados y fundamento metodológico. "), f"El champion, HistGradientBoosting con pérdida Poisson, tiene "
              f"el menor MAE de validación ({f(hgb.get('val_mae'))}) y por eso se eligió; solo después se evaluó una vez "
              f"en prueba: MAE {f(test_mae)}, {pc(-rel(test_mae, dum.get('test_mae')), 1, False)} menor que la referencia Dummy "
              f"({f(dum.get('test_mae'))}), con R² de {f(hgb.get('test_r2'))}. RandomForest queda a "
              f"{pc(rel(rf.get('test_mae'), test_mae))} y se registró como challenger. Que dos familias de ensambles "
              "lleguen a errores tan parecidos indica que el límite está en la información de las variables más que "
              "en el algoritmo. Optuna eligió learning_rate ", f"{hp.get('learning_rate', 0):.3f}", ", max_iter ",
         str(hp.get("max_iter")), ", max_leaf_nodes ", str(hp.get("max_leaf_nodes")), " y min_samples_leaf ",
         str(hp.get("min_samples_leaf")), ": muchas iteraciones con tasa baja y hojas grandes, un modelo flexible pero "
                                          f"estable con {n(rows_train)} filas de entrenamiento.")
    para(ctx, ("b", "Por qué falla la regresión de Poisson lineal. "), f"Reduce el MAE en {pc(-rel(poi.get('test_mae'), dum.get('test_mae')), 1, False)} "
              f"contra la media, pero su RMSE de prueba ({f(poi.get('test_rmse'))}) es mayor que el de la propia media "
              f"({f(dum.get('test_rmse'))}) y su R² es negativo ({f(poi.get('test_r2'))}; en validación "
              f"{f(poi.get('val_r2'), 1)}). Con enlace logarítmico las variables actúan de forma lineal en la escala del "
              "logaritmo, es decir, multiplicativa en la demanda: en estaciones y horas de alta demanda la combinación "
              "de rezagos altos y TargetEncoder produce predicciones exponencialmente grandes. Unas pocas predicciones "
              "extremas dominan el error cuadrático. Los árboles no tienen ese problema porque cada hoja predice un "
              "valor acotado por los datos y capturan interacciones estación × hora sin especificarlas.")
    para(ctx, "La Figura ", ctx.ref("f:perfil"), " compara la demanda real y la predicha por hora del día en el periodo "
              "de prueba.", keep=True)
    prof = ROOT / "mlartifacts" / (meta.get("run_id") or "x") / "artifacts" / "plots" / "perfil_horario_hist_gradient_boosting.png"
    prof_note = "artefacto plots/perfil_horario_hist_gradient_boosting.png de la corrida del champion en MLflow."
    if not prof.exists():
        prof = fase2_image("fig4.png", tmp)
        prof_note = "figura del reporte de la Fase 2 (la corrida local de MLflow no está disponible en esta máquina)."
    figure(ctx, "f:perfil", prof, "Demanda media real contra predicha por hora del día en prueba (champion)",
           [prof_note, " El modelo reproduce la forma del día (sin demanda de 1:00 a 4:00, picos de 8:00 y 18:00) y "
                       "sobrestima ligeramente los picos, lo que es preferible a subestimarlos para el rebalanceo; el "
                       "sesgo medio en prueba es ", pc(base.get("bias")) if base else "s/d", " (drift_report.json)."],
           width=5.4)
    para(ctx, ("b", "Análisis reflexivo. "), f"Un MAE de {f(test_mae, 2)} bicicletas por hora parece pequeño, pero la demanda "
              f"media en prueba es {f(base.get('mean_demand'), 2) if base else 's/d'}, así que el error relativo (WAPE) "
              f"es {pc(base.get('wape'), 1, False) if base else 's/d'}. Para el operador esto significa que el modelo "
              "sirve para ordenar estaciones por riesgo y anticipar los picos, no para fijar el número exacto de "
              "bicicletas por estación y hora. Esta lectura nos llevó a vigilar el WAPE además del MAE en el monitoreo "
              "(sección 4.12).")

    # 4.6 Tracking y Registry
    heading(ctx, "4.6 Seguimiento de experimentos y Model Registry", 2)
    table(ctx, "t:m-mlflow", "Seguimiento y registro: método, alternativas y justificación",
          ["Decisión", "Alternativas", "Justificación"],
          [["MLflow Tracking con SQLite local; corrida padre por modelo y trials anidados",
            "Hojas de cálculo; Weights & Biases", "Gratuito, local, con Model Registry integrado (Zaharia et al., "
                                                  "2018)."],
           ["Model Registry con alias champion/challenger", "Etapas Staging/Production; nombres de archivo",
            "Los alias permiten promover o revertir sin cambiar el código que consume el modelo."],
           ["Exportar el champion a carpeta MLflow autocontenida con metadatos y SHA-256",
            "Servir desde el Registry; copiar un .pkl suelto", "El servicio no depende de mlruns.db; el hash "
                                                               "detecta un modelo alterado al arrancar."]],
          widths=[2.4, 1.6, 2.5], size=8.5)
    reg_img = fase2_image("fig8.png", tmp)
    figure(ctx, "f:registry", reg_img, "Versiones del modelo mibici-demand-forecaster en el Model Registry (Fase 2)",
           ["captura del reporte de la Fase 2: tres versiones con etiquetas de algoritmo, hiperparámetros, hash del "
            "dataset y métricas; champion en v1 y challenger en v2. El reentrenamiento de la fase final reprodujo la "
            "misma asignación (04_train.log)."], width=6.2)
    para(ctx, ("b", "Resultado. "), "src/modeling/export.py exportó ", uri, " a models/champion junto con "
              "models/champion_metadata.json (run_id ", ("c", meta.get("run_id", "")), ", SHA-256 del modelo ",
         ("c", (meta.get("model_sha256") or "")[:16] + "…"), ", métricas, columnas y versiones de librerías) y "
         f"models/reference_predictions.csv con {n(val(meta, 'reference_predictions', 'rows'))} filas de prueba "
         f"(semilla {val(meta, 'reference_predictions', 'seed')}). ",
         f"La recarga desde el Registry reprodujo MAE {logs['predict'][0]} / RMSE {logs['predict'][1]} / R² "
         f"{logs['predict'][2]} sobre {n(logs.get('predict_rows'))} filas (05_predict.log)." if logs.get("predict")
         else Pend("05_predict.log"), align=WD_ALIGN_PARAGRAPH.LEFT)

    # 4.7 Estructura
    heading(ctx, "4.7 Estructura del repositorio y refactorización", 2)
    para(ctx, "El repositorio sigue Cookiecutter Data Science v2 (DrivenData, 2024). Cada etapa es un módulo con "
              "interfaz de línea de comandos que lee config.yaml; ninguna ruta, semilla, umbral o hiperparámetro está "
              "escrito en el código. La Tabla ", ctx.ref("t:modulos"), " resume los módulos. Elegimos un archivo de "
              "configuración único en lugar de argumentos dispersos o variables de entorno porque es versionable, "
              "legible y lo consumen por igual el entrenamiento, la API, el monitoreo y las pruebas.", keep=True)
    mods = [["src/config.py", "Carga de config.yaml, rutas, semillas globales"],
            ["src/dataset.py", "Descarga, lectura robusta y limpieza"],
            ["src/features.py", "Cuadrícula, calendario, rezagos, partición y preprocesador"],
            ["src/tracking.py", "MLflow, SHA-256 y versionado de datos"],
            ["src/modeling/", "pipeline.py, train.py (Optuna + Registry), predict.py, export.py"],
            ["src/serving/", "app.py (FastAPI), schemas.py (Pydantic v2) y feature_store.py (rezagos en línea)"],
            ["models/", "champion/ (modelo, estaciones y feature store), metadatos, patrones oro y fase2_original/"],
            ["src/monitoring/", "scenarios, metrics, rules, figures, report y drift (CLI)"],
            ["tests/unit, tests/integration", "Pruebas con pytest y fixtures compartidas"],
            ["docs/, scripts/, docker/", "Documentación de la API, OpenAPI, Postman, scripts de verificación"]]
    table(ctx, "t:modulos", "Módulos del repositorio", ["Ruta", "Responsabilidad"], mods, widths=[2.0, 4.5],
          size=8.5)

    # 4.8 Pruebas
    heading(ctx, "4.8 Pruebas unitarias y de integración", 2)
    table(ctx, "t:m-tests", "Pruebas: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                            "Justificación"],
          [["pytest con fixtures de sesión y datos sintéticos con anomalías en cantidades conocidas",
            "unittest (Fases 1 y 2)", "Fixtures reutilizables, parametrización, marcadores y cobertura; las "
                                      "pruebas no usan red ni datos reales."],
           ["Marcador integration para el flujo de punta a punta con un Registry temporal", "Solo pruebas unitarias",
            "Detecta rupturas entre etapas (p. ej. cambios de serialización de MLflow)."],
           ["Cobertura de ramas con pytest-cov y CI en cada push", "Medir solo líneas; correr a mano",
            "La cobertura de ramas muestra caminos de error sin probar (Breck et al., 2017)."],
           ["Prueba de coherencia servicio-pipeline: el servicio recalcula rezagos y predicciones del feature store "
            "y deben coincidir con el pipeline", "Confiar en que el servicio usa las mismas funciones",
            "Detecta training-serving skew; idea tomada de la rama entrega-final (tests/test_serving_feature_store.py)."]],
          widths=[2.4, 1.4, 2.7], size=8.5)
    para(ctx, "El comando único es ", ("c", "pytest -q"), " (configuración en pyproject.toml). Resultado: ",
         f"{n(pt.get('passed'))} pruebas aprobadas, {pt.get('failed', 0)} fallidas, en {pt.get('seconds', 's/d')} s, "
         f"con cobertura de {pt.get('coverage', 's/d')} % de src/ ({n(pt.get('statements'))} sentencias y "
         f"{n(pt.get('branches'))} ramas)" if pt.get("passed") else Pend("10_pytest_cobertura.log"),
         ". La Tabla ", ctx.ref("t:testfiles"), " muestra las funciones de prueba por archivo y la Tabla ",
         ctx.ref("t:cov"), " la cobertura por módulo.")
    tf = [[p_, str(c), "integración" if "integration" in p_ else ("servicio" if "serving" in p_ else
                                                                  ("monitoreo" if "drift" in p_ else "unitaria"))]
          for p_, c in pt.get("files", [])]
    table(ctx, "t:testfiles", "Funciones de prueba por archivo", ["Archivo", "Funciones test_", "Tipo"], tf,
          widths=[3.6, 1.2, 1.4], size=8.5, align=["l", "r", "l"],
          note=["conteo de definiciones def test_ (una función parametrizada genera varias pruebas, por eso el total "
                "ejecutado es mayor). Detalle por archivo en tests/README.md."])
    cov_rows = [[m_, str(s), str(mi), f"{c} %"] for m_, s, mi, c in pt.get("modules", [])]
    cfills = {(i, 3): (OK_FILL if int(r[3][:-2]) >= 90 else WARN_FILL) for i, r in enumerate(cov_rows)}
    table(ctx, "t:cov", "Cobertura de pruebas por módulo (líneas y ramas)", ["Módulo", "Sentencias", "Sin cubrir",
                                                                           "Cobertura"], cov_rows,
          widths=[2.8, 1.1, 1.1, 1.0], size=8.5, align=["l", "r", "r", "r"], fills=cfills,
          note=["reports/fase_final/logs/10_pytest_cobertura.log."])
    para(ctx, ("b", "Análisis. "), "La cobertura global supera la meta de 85 % que nos fijamos. El módulo menos cubierto es "
              "src/monitoring/drift.py: las líneas sin cubrir son la orquestación de la CLI y el registro en MLflow, "
              "cuya lógica de cálculo sí está probada en metrics.py, rules.py y scenarios.py. Las pruebas también "
              "revelaron límites reales: en una primera corrida en Linux, dos pruebas que exigían igualdad exacta con la "
              "referencia fallaron por una diferencia de 1 ulp (sección 4.10) y se corrigieron para usar tolerancia "
              "1e-9; con ese ajuste la suite completa pasa también en Linux con el modelo real. La lección es que una "
              "prueba de reproducibilidad debe declarar su tolerancia en lugar de suponer igualdad bit a bit entre "
              "sistemas operativos. En GitHub Actions el flujo tests.yml corre la suite con cobertura en Python 3.13 y un "
         "runner Linux en cada push (", Link("pestaña Actions", rd.ACTIONS_URL), "; corrida del push final: ",
         Link(rd.TESTS_RUN_URL.rsplit("/", 1)[-1], dock.get("gha_build_run_url") or rd.TESTS_RUN_URL),
         "); como el modelo ya está en Git, "
         "ahí corren también las pruebas que usan el modelo real, y un segundo job (docker) construye la imagen "
         "desde el repositorio, la levanta y la verifica contra los patrones oro.")

    # 4.9 Serving
    heading(ctx, "4.9 Servicio de predicción con FastAPI", 2)
    table(ctx, "t:m-api", "Servicio: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                           "Justificación"],
          [["FastAPI + Pydantic v2 + uvicorn", "Flask; mlflow models serve", "Validación declarativa, OpenAPI "
                                                                            "automático y errores 422 detallados; "
                                                                            "control total del esquema."],
           ["POST /predict recibe estación y fecha; el servidor deriva calendario y estación",
            "Exigir las 13 columnas del modelo", "Contrato simple para el cliente y sin diferencias entre "
                                                 "entrenamiento y servicio (mismas funciones de src/features.py)."],
           ["Rezagos opcionales: sin ellos se calculan con un feature store de las últimas 360 h por estación "
            "(1.1.0)", "Obligar al cliente a enviarlos (1.0.0); consultar una base en línea",
            "El cliente no reimplementa el cálculo; se usa add_lag_features del entrenamiento. Idea de entrega-final. "
            "Limitación: la ventana predecible es fija."],
           ["Subir a 1.1.0 (MINOR) y conservar 1.0.0 publicada", "Reemplazar 1.0.0",
            "Toda petición válida en 1.0.0 sigue siendo válida y da la misma predicción; quien fijó 1.0.0 no se rompe."],
           ["Registro opcional de predicciones (PREDICTION_LOG, JSONL)", "Solo logs del servidor",
            "Auditoría de entradas, salidas, versión y latencia para cerrar el ciclo con las etiquetas."],
           ["POST /predict/raw con las 13 columnas", "Solo /predict", "Permite verificar la reproducibilidad "
                                                                      "contra la referencia sin pasar por la "
                                                                      "derivación de variables."],
           ["Verificación del SHA-256 al arrancar; 503 si el modelo no es el esperado", "Cargar cualquier archivo",
            "Evita servir un modelo alterado o de otra versión."]],
          widths=[2.4, 1.5, 2.6], size=8.5)
    ep = [["GET /health", "Estado, modelo cargado y SHA-256 verificado", "200, 503"],
          ["GET /model-info", "Nombre, versión, URI, alias, run_id, hiperparámetros, métricas, columnas y versiones",
           "200, 503"],
          ["POST /predict", f"Lote de 1 a {n(max_batch)} instancias {{station_id, datetime}} con los tres rezagos "
                            "opcionales (los tres o ninguno); responde lags_source y los rezagos usados",
           "200, 404, 422, 503"],
          ["POST /predict/raw", "Filas con las 13 columnas del modelo", "200, 422, 503"]]
    table(ctx, "t:endpoints", "Endpoints de la API", ["Endpoint", "Descripción", "Códigos"], ep,
          widths=[1.4, 4.0, 1.1], size=8.5, note=["esquema completo en docs/api.md y docs/openapi.json; colección de "
                                                  "Postman en docs/postman_collection.json."])
    if api.get("example_request"):
        para(ctx, "Petición y respuesta reales (fila 1 de la referencia, 20_api_smoke.log):", keep=True)
        req_compact = re.sub(r"\s+", " ", api["example_request"]).replace("{ ", "{").replace(" }", "}")
        resp_compact = re.sub(r"\s+", " ", api["example_response"]).replace("{ ", "{").replace(" }", "}")
        code(ctx, "POST /predict\n" + req_compact + f"\n\nHTTP {api['example_status']}\n" + resp_compact, size=8)
    if api.get("fs_request"):
        para(ctx, "Sin rezagos (los calcula el servicio con el feature store; ventana predecible ",
             fsw.get("start", "s/d"), " a ", fsw.get("end", "s/d"), "):", keep=True)
        req_compact = re.sub(r"\s+", " ", api["fs_request"]).replace("{ ", "{").replace(" }", "}")
        resp_compact = re.sub(r"\s+", " ", api["fs_response"]).replace("{ ", "{").replace(" }", "}")
        code(ctx, "POST /predict\n" + req_compact + f"\n\nHTTP {api['fs_status']}\n" + resp_compact, size=8)
    srows = []
    for s in api.get("sections", []):
        codes = ", ".join(f"{k}: {v}" for k, v in sorted(s["codes"].items()))
        rep_txt = "; ".join(f"{p_}{' sin rezagos' if 'sin rezagos' in (ref or '') else ''}: {a}/{b} iguales bit a "
                            f"bit, dif. máx. {float(d):.2g}" for p_, ref, _, d, a, b in s["repro"]) or "—"
        srows.append([s["title"], str(s["requests"]), codes, rep_txt])
    table(ctx, "t:smoke", "Prueba de humo de la API contra uvicorn", ["Entorno", "Peticiones", "Códigos HTTP",
                                                                     "Reproducción de la referencia"], srows,
          widths=[2.1, 0.8, 1.4, 2.2], size=8.5, note=["reports/fase_final/logs/20_api_smoke.log (scripts/api_smoke.py)."])
    para(ctx, ("b", "Análisis. "), "La prueba de humo cubre el camino feliz, cada familia de error (tipos, negativos, NaN, "
              "fechas inválidas, campos faltantes o extra, lotes vacíos o mayores al límite, estaciones desconocidas) "
              "y el arranque sin modelo. Que el venv limpio con solo requirements-serve.txt reproduzca exactamente "
              "las 500 predicciones demuestra que el servicio no depende de la base de MLflow ni del código de "
              "entrenamiento. Con el feature store, el cliente solo envía estación y hora dentro de la ventana predecible "
              f"({fsw.get('start', 's/d')} a {fsw.get('end', 's/d')}); fuera de ella responde 422 con la ventana válida. "
              "Limitación: el feature store es una foto de las últimas "
              f"{fs.get('hours', 's/d')} horas de datos que caduca; en operación habría que actualizarlo con cada "
              "ingesta. Las Figuras ", ctx.ref("f:swagger"), " a ", ctx.ref("f:resp"), " muestran la documentación "
              "interactiva, el esquema de /predict y una respuesta real.")
    figure(ctx, "f:swagger", FIG / "12_api_swagger_docs.png", "Documentación interactiva de la API (Swagger UI, /docs)",
           ["captura del servicio 1.1.0: endpoints de predicción y de operación, y los esquemas Pydantic."], width=4.6)
    figure(ctx, "f:esquema", FIG / "13_api_esquema_predict.png", "Esquema y ejemplos de POST /predict en Swagger UI",
           ["ejemplo con una instancia con rezagos y otra sin ellos (lags_source = request y feature_store)."],
           width=3.6)
    figure(ctx, "f:resp", FIG / "14_api_respuesta_200.png", "Respuesta real de POST /predict del contenedor 1.1.0",
           ["figura generada por scripts/build_api_response_figure.py a partir de la petición y respuesta registradas "
            "en 20_api_smoke.log (sección B); no es una captura del navegador."], width=6.3)

    # 4.10 Reproducibilidad
    heading(ctx, "4.10 Reproducibilidad en otro entorno", 2)
    table(ctx, "t:m-repro", "Reproducibilidad: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                                     "Justificación"],
          [["Versiones exactas (==) y requirements-lock.txt; imagen base con digest", ">= (Fase 2); Conda",
            "Un clon nuevo instala exactamente lo mismo; el digest fija también el sistema base."],
           ["Semilla única (seed: 42) en estimadores, TargetEncoder, Optuna, NumPy, random y PYTHONHASHSEED",
            "Semillas por módulo", "Una sola fuente de verdad en config.yaml (src/config.py: set_global_seeds)."],
           ["Patrón oro: 500 predicciones de referencia y comparación con tolerancia explícita",
            "Comparar solo métricas agregadas", "Una métrica igual puede ocultar predicciones distintas; la "
                                                "comparación fila a fila no."],
           ["Desplegar el binario verificado, no reentrenar en cada entorno", "Reentrenar en el destino",
            "El reentrenamiento entre sistemas operativos no es idéntico bit a bit (ver abajo)."]],
          widths=[2.4, 1.4, 2.7], size=8.5)
    para(ctx, ("b", "Primera verificación: reentrenar en un entorno limpio y comparar con la Fase 2. "), "La Tabla ",
         ctx.ref("t:f2vs"), " compara las métricas del champion; las de los cuatro modelos están en "
                            "reports/fase_final/comparacion_fase2.md.", keep=True)
    f2rows = []
    for metric, lab in (("val_mae", "MAE validación"), ("test_mae", "MAE prueba"), ("test_rmse", "RMSE prueba"),
                        ("test_r2", "R² prueba")):
        a, b = hgb2.get(metric), hgb.get(metric)
        f2rows.append([lab, f(a), f(b), f"{b - a:+.4f}" if a is not None and b is not None else "s/d",
                       pc(rel(b, a), 2) if a else "s/d"])
    table(ctx, "t:f2vs", "Champion de la Fase 2 contra el reentrenamiento en entorno limpio",
          ["Métrica", "Fase 2", "Fase final", "Diferencia", "Diferencia %"], f2rows,
          widths=[1.8, 1.1, 1.1, 1.1, 1.1], size=9, align=["l", "r", "r", "r", "r"],
          note=["model_comparison_fase2_referencia.csv y model_comparison.csv. Criterio de aceptación: ±2 % del MAE "
                "de prueba de la Fase 2."])
    para(ctx, f"La diferencia del MAE de prueba es de {pc(rel(test_mae, test_mae2), 2)} y se explica por los datos: el portal republicó archivos y el modelo "
              "Dummy, que no tiene nada aleatorio, cambia en la misma magnitud. Optuna recuperó los mismos "
              "hiperparámetros del champion y la asignación de alias no cambió.")
    para(ctx, ("b", "Segunda verificación: el mismo artefacto en otros entornos. "), "scripts/verify_reproducibility.py envía "
              "a un servicio levantado las 500 filas de la referencia con rezagos (por /predict y /predict/raw) y las "
              f"200 filas de la referencia del feature store sin rezagos, y compara fila por fila. La Tabla ",
         ctx.ref("t:envs"), f" resume cada entorno verificado con el servicio {svc_ver}.", keep=True)
    erows, efills = [], {}
    envs = sorted(envs, key=lambda e: ("tolerancia0" in e["_file"], e["_file"]))
    for i, e in enumerate(envs):
        epd = val(e, "endpoints", "predict") or {}
        sv = val(e, "server_environment", "runtime_library_versions") or {}
        fp = fs_path(e)
        erows.append([e.get("label"), f"Python {sv.get('python', 's/d')}, scikit-learn {sv.get('scikit-learn', 's/d')}",
                      f"{e.get('tolerance')}", f"{epd.get('max_abs_diff', 's/d'):.3g}"
                      if isinstance(epd.get('max_abs_diff'), float) else str(epd.get("max_abs_diff")),
                      f"{epd.get('rows_bitwise_identical', 's/d')}/{epd.get('rows', 's/d')}",
                      f"{fp.get('max_abs_diff'):.2g} ({fp.get('rows_bitwise_identical')}/{fp.get('rows')})" if fp else "—",
                      e.get("verdict"), e["_file"].split("/")[-1]])
        efills[(i, 6)] = OK_FILL if e.get("verdict") == "PASS" else ALERT_FILL
    extra = []
    if rep.get("verify_diff") is not None:
        extra.append(["proceso nuevo, venv de entrenamiento", f"Python {pyv} (Windows)", "0", rep["verify_diff"],
                      "500/500", "—", "PASS", "07_verify_reference.log"])
    if rep.get("standalone_diff") is not None:
        extra.append(["venv solo con requirements-serve.txt", f"Python {pyv} (Windows)", "0", rep["standalone_diff"],
                      "500/500", "—", "PASS", "08_verify_standalone…log"])
    for j, row in enumerate(extra):
        efills[(len(erows) + j, 6)] = OK_FILL
    gha_row = []
    if not dock.get("gha"):
        gha_row = [["runner de GitHub Actions (imagen 1.1.0 publicada)", "Linux", "1e-09", "—", "—", "—", "—",
                    [Pend("reproducibilidad/github_actions.json (imagen 1.1.0)")]]]
    table(ctx, "t:envs", "Verificación de las predicciones de referencia por entorno (servicio vigente)",
          ["Entorno", "Servidor", "Tol.", "Dif. máx. con rezagos", "Idénticas (500)", "Sin rezagos: dif. máx. "
                                                                                  "(idénticas/200)",
           "Veredicto", "Archivo"],
          erows + extra + gha_row, widths=[1.35, 1.15, 0.5, 0.75, 0.65, 0.95, 0.65, 1.2], size=7.5,
          align=["l", "l", "r", "r", "r", "r", "c", "l"], fills=efills,
          note=["archivos en reports/fase_final/reproducibilidad/ y reports/fase_final/logs/. La fila con tolerancia "
                "0 es un control deliberadamente estricto: falla por la diferencia de 1 ulp que se explica abajo. "
                f"Historial de la versión 1.0.0 (en reproducibilidad/v1.0.0/): {len(envs_v1)} verificaciones, "
                f"{sum(1 for e in envs_v1 if e.get('verdict') == 'PASS')} con PASS, incluida la del runner de GitHub "
                "Actions (", Link(f"corrida {dock['gha_v1_run_url'].rsplit('/', 1)[-1]}", dock["gha_v1_run_url"]),
                ")."])
    if clon:
        cep = val(clon, "endpoints", "predict") or {}
        para(ctx, ("b", "Clon limpio. "), "Copiamos a una carpeta fuera del repositorio solo los archivos de Git (sin "
                  "datos, sin mlruns.db y sin entorno virtual) y ahí corrieron docker build -t ml-service:latest . y "
                  "la verificación: veredicto ", ("b", str(clon.get("verdict"))), ", diferencia máxima ",
             f"{cep.get('max_abs_diff', 0):.2g}", " con rezagos y ", f"{fs_path(clon).get('max_abs_diff', 0):.2g}",
             " sin rezagos (reproducibilidad/clon_limpio.json y clon_limpio_build.log). Esto demuestra que el "
             "repositorio por sí solo basta para reconstruir el servicio.")
    lin_env = next((e for e in envs if e["_file"].endswith("/contenedor_local.json")), None)
    lin_ep = val(lin_env, "endpoints", "predict") or {}
    lin_rows = lin_ep.get("rows", "s/d")
    lin_diff_rows = (lin_rows - lin_ep["rows_bitwise_identical"]) if isinstance(lin_rows, int) else "s/d"
    lin_max = f"{lin_ep['max_abs_diff']:.1e}" if isinstance(lin_ep.get("max_abs_diff"), float) else "s/d"
    para(ctx, ("b", "Hallazgo: 1 ulp entre Windows y Linux. "), "En el contenedor Linux, ", str(lin_diff_rows), " de las ",
         str(lin_rows), " predicciones difieren en la última cifra binaria (máximo ", lin_max, "). El diagnóstico (reproducibilidad/diagnostico_exp/"
              "resumen.txt) muestra que la suma de los árboles en escala logarítmica es idéntica bit a bit y que la "
              "diferencia aparece solo en el exp() final de la función de enlace, porque la biblioteca matemática de "
              "C de Windows (UCRT) y la de Linux (glibc) redondean distinto esos casos. Por eso la tolerancia por "
              "omisión es 1e-9, seis órdenes de magnitud por encima de lo observado y muy por debajo de cualquier "
              "diferencia con significado para un conteo de bicicletas.")
    pcsv = rep.get("pipeline_csv")
    if pcsv:
        hg = [r for r in pcsv if r["model"] == "hist_gradient_boosting" and r["metric"] in ("val_mae", "test_mae",
                                                                                           "test_rmse", "test_r2")]
        ident = sorted({r["model"] for r in pcsv if r["identical"] == "True"} -
                       {r["model"] for r in pcsv if r["identical"] == "False"})
        para(ctx, ("b", "Tercera verificación: reentrenar todo el pipeline en un contenedor Linux. "), "Con los mismos CSV, "
                  "semillas y versiones, ", ", ".join(ident) or "ningún modelo",
             " salen idénticos bit a bit, pero el champion difiere ligeramente: ",
             "; ".join(f"{r['metric']} {float(r['windows']):.4f} (Windows) vs {float(r['linux']):.4f} (Linux)" for r in hg),
             ". La causa es la misma de 1 ulp: la pérdida Poisson usa exp() en cada iteración, Optuna recibe "
             "valores de validación ligeramente distintos y a partir de cierto trial propone otros hiperparámetros. "
             "La conclusión del experimento (ranking, algoritmo y nivel de error) no cambia. Detalle en "
             "reports/fase_final/reproducibilidad/pipeline_linux_vs_windows.md.")
    else:
        para(ctx, ("b", "Tercera verificación: reentrenamiento en Linux. "), Pend("pipeline_linux_vs_windows.csv"))
    if f2vs:
        own_w = f2w.get("fase2_reproduces_own_reference") or {}
        own_l = f2l.get("fase2_reproduces_own_reference") or {}
        exp_env = val(f2w, "fase2_binary", "export_environment") or {}
        para(ctx, ("b", "Cuarta verificación: el binario original de la Fase 2. "), "La rama entrega-final conservaba el "
                  "modelo que se entrenó en la Fase 2 en otra máquina (Python ", exp_env.get("python", "s/d"),
             "); lo guardamos en models/fase2_original/. Ese mismo binario reproduce aquí sus ",
             n(own_w.get("rows")), " predicciones de referencia con diferencia máxima ", f"{own_w.get('max_abs_diff', 0):.2g}",
             " en Windows (Python ", val(f2w, "environment", "python") or "s/d", ") y ",
             f"{own_l.get('max_abs_diff', 0):.2g}", f" en el contenedor Linux ({own_l.get('rows_bitwise_identical')}/"
             f"{own_l.get('rows')} idénticas). La Tabla ", ctx.ref("t:f2bin"), " lo compara con el champion "
                                                                              "reentrenado.", keep=True)
        mva = f2vs.get("metrics_vs_actual") or {}
        table(ctx, "t:f2bin", "Binario de la Fase 2 contra el champion reentrenado (500 filas de la referencia)",
              ["Medida", "Valor"],
              [["Mismos hiperparámetros", "sí" if f2w.get("same_hyperparameters") else "no"],
               ["Diferencia máxima entre predicciones", f"{f2vs.get('max_abs_diff'):.4f}"],
               ["Diferencia media absoluta", f"{f2vs.get('mean_abs_diff'):.4f}"],
               ["Filas con diferencia > 0.1 viajes", f"{f2vs.get('rows_abs_diff_gt_0.1')} de {f2vs.get('rows')}"],
               ["Correlación de Pearson", f"{f2vs.get('pearson_correlation'):.5f}"],
               ["MAE contra la demanda real (Fase 2 / reentrenado)",
                f"{val(mva, 'fase2_binary', 'mae'):.4f} / {val(mva, 'retrained_champion', 'mae'):.4f}"],
               ["RMSE (Fase 2 / reentrenado)",
                f"{val(mva, 'fase2_binary', 'rmse'):.4f} / {val(mva, 'retrained_champion', 'rmse'):.4f}"]],
              widths=[3.6, 2.9], size=9, align=["l", "r"],
              note=["reproducibilidad/fase2_binario_vs_reentrenado.json y .md (scripts/compare_fase2_binary.py). El MAE "
                    "es sobre la muestra de 500 filas, no sobre el conjunto de prueba completo."])
        para(ctx, "Interpretación: no son el mismo binario aunque tienen el mismo algoritmo, la misma semilla y los "
                  "mismos hiperparámetros, porque se entrenaron con datos del portal descargados en fechas distintas "
                  "(sección 4.1). Sus predicciones casi coinciden (correlación cercana a 1, diferencia media de "
                  "centésimas de viaje) y su error es prácticamente el mismo; la diferencia mide el efecto de los "
                  "datos, no del código.")
    lin, lin_prev = rep.get("linux_pytest"), rep.get("linux_pytest_prev")
    if lin or lin_prev:
        prev_txt = (f"La primera corrida de la suite en el contenedor Linux con el modelo real dio {lin_prev['passed']} "
                    f"aprobadas y {lin_prev['failed']} fallidas ({lin_prev['file']}): las dos pruebas que exigían "
                    "igualdad exacta con la referencia fallaron por el ulp descrito arriba. Se cambiaron a tolerancia "
                    "1e-9 (tests/test_serving.py y tests/integration/test_versioned_artifacts.py), igual que la "
                    "verificación del servicio. " if lin_prev and lin_prev["failed"] else "")
        cur_txt = (f"Con el código actual, la suite completa en Linux (Python 3.13.9, modelo real) da "
                   f"{lin['passed']} aprobadas y {lin['failed']} fallidas en {lin['seconds']:.1f} s ({lin['file']})."
                   if lin else Pend("corrida vigente de la suite en Linux"))
        para(ctx, ("b", "Pruebas en Linux. "), prev_txt, cur_txt)
    para(ctx, ("b", "Análisis reflexivo. "), "La reproducibilidad tiene niveles. El artefacto desplegado se reproduce entre "
              "entornos con una diferencia máxima de 1 ulp; el reentrenamiento se reproduce exactamente dentro de un "
              "mismo sistema operativo, pero no bit a bit entre sistemas cuando el algoritmo usa funciones "
              "trascendentes. Por eso lo que se versiona y despliega es el binario con su SHA-256, y lo que se pide "
              "al reentrenar en otro entorno es que la conclusión sea la misma, no que los bits lo sean.")

    # 4.11 Contenedor
    heading(ctx, "4.11 Contenedor Docker y publicación", 2)
    table(ctx, "t:m-docker", "Contenedor: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                                 "Justificación"],
          [["Dockerfile multi-etapa sobre python:3.13.9-slim fijado por digest", "Imagen completa de Python; "
                                                                                 "Podman",
            "Misma versión de Python que el entrenamiento; imagen pequeña; la entrega pide Docker y DockerHub."],
           ["Solo requirements-serve.txt (mlflow-skinny) y los módulos que importa el servicio",
            "Copiar todo el repositorio", "Menos superficie y menos dependencias; el entrenamiento no viaja en la "
                                          "imagen."],
           ["Usuario sin privilegios, HEALTHCHECK a /health y verificación del SHA-256 durante el build",
            "Correr como root sin chequeo", "Buenas prácticas de seguridad y un build que falla con mensaje claro si "
                                           "falta el modelo."],
           ["Tags <servicio> y <servicio>-model-v<N>, más latest; 1.1.0 se publicó sin retirar 1.0.0", "Solo latest",
            "Versión del servicio y del modelo explícitas e inmutables; latest para pruebas rápidas."],
           ["Modelo dentro del contexto de build desde Git", "Generarlo antes del build; descargarlo con DVC",
            "docker build -t ml-service:latest . funciona en un clon limpio y en el CI sin pasos previos."]],
          widths=[2.4, 1.4, 2.7], size=8.5)
    para(ctx, "Comandos de la entrega (desde la raíz de un clon del repositorio, sin pasos previos):", keep=True)
    code(ctx, "docker build -t ml-service:latest .\ndocker run -p 8000:8000 ml-service:latest\n"
              f"docker pull {rd.DOCKER_IMAGE}:{svc_ver}\ndocker run -p 8000:8000 {rd.DOCKER_IMAGE}:{svc_ver}\n"
              "# Podman (comandos de la rama entrega-final): podman build --format docker -t ml-service:latest .")
    meaning = {"1.1.0": "Servicio 1.1.0: rezagos opcionales con feature store",
               "1.1.0-model-v1": "Servicio 1.1.0 con models:/mibici-demand-forecaster/1",
               "1.0.0": "Servicio 1.0.0: /predict exige los tres rezagos",
               "1.0.0-model-v1": "Servicio 1.0.0 con models:/mibici-demand-forecaster/1",
               "latest": "Última imagen publicada"}
    trows = [[[("c", t)], meaning.get(t, ""), [("c", v["digest"][:19] + "…")], n(v["full_size_bytes"]),
              v["last_updated"][:10]] for t, v in sorted(hub_tags.items())]
    if not trows:
        trows = [["—", "—", [Pend("reproducibilidad/dockerhub_tags.json")], "—", "—"]]
    table(ctx, "t:tags", f"Tags publicados de {rd.DOCKER_IMAGE} en DockerHub",
          ["Tag", "Significado", "Digest", "Bytes comprimida", "Publicada"], trows,
          widths=[1.2, 2.4, 1.4, 0.9, 0.8], size=8, align=["l", "l", "l", "r", "l"],
          note=["consulta de la API pública de DockerHub guardada en reproducibilidad/dockerhub_tags.json; liga: ",
                Link(rd.DOCKERHUB_URL, rd.DOCKERHUB_URL), "."])
    hub_fs = fs_path(dock.get("hub") or {})
    drows = [["Imagen vigente", [("c", f"{rd.DOCKER_IMAGE}:{svc_ver}")], "dockerhub_tags.json"],
             ["Digest (1.1.0, 1.1.0-model-v1, latest)", [("c", t110.get("digest", ""))] if t110 else
              [Pend("digest publicado")], "dockerhub_tags.json"],
             ["Tamaño", dock.get("size") or Pend("tamaño de la imagen"), "docs/docker.md"],
             ["Tiempo de build", dock.get("build_time") or "s/d", "docs/docker.md"],
             ["Base", "python:3.13.9-slim fijada por digest", "Dockerfile"],
             ["Verificación tras docker pull", f"{val(dock, 'hub', 'verdict') or 's/d'}: dif. máx. "
              f"{val(dock, 'hub', 'endpoints', 'predict', 'max_abs_diff'):.2g} con rezagos "
              f"({val(dock, 'hub', 'endpoints', 'predict', 'rows_bitwise_identical')}/500 idénticas) y "
              f"{hub_fs.get('max_abs_diff', 0):.2g} sin rezagos ({hub_fs.get('rows_bitwise_identical')}/200)"
              if dock.get("hub") else Pend("imagen_dockerhub.json"), "reproducibilidad/imagen_dockerhub.json"]]
    table(ctx, "t:imagen", "Detalles de la imagen publicada", ["Concepto", "Valor", "Evidencia"], drows,
          widths=[1.6, 3.4, 1.5], size=8.5)
    gha = dock.get("gha")
    gha_txt = ([" Para la versión ", svc_ver, " el flujo verificó la imagen publicada con veredicto ",
                ("b", str(gha.get("verdict"))), ", diferencia máxima ",
                f"{val(gha, 'endpoints', 'predict', 'max_abs_diff'):.2g}", " con rezagos y ",
                f"{fs_path(gha).get('max_abs_diff', 0):.2g}" if fs_path(gha) else "s/d", " sin rezagos (",
                Link("corrida", dock["gha_run_url"]) if dock.get("gha_run_url") else "reproducibilidad/",
                "; reproducibilidad/github_actions.json)."] if gha else
               [" ", Pend("resultado del flujo docker-reproducibilidad para la imagen 1.1.0 "
                          "(reproducibilidad/github_actions.json)")])
    gb = dock.get("gha_build")
    if gb:
        gha_txt = gha_txt + [" El job docker de tests.yml, que construye la imagen desde el repositorio en el runner, "
                             "dio ", ("b", str(gb.get("verdict"))), " con las mismas diferencias (",
                             f"{val(gb, 'endpoints', 'predict', 'max_abs_diff'):.2g}", " con rezagos y ",
                             f"{fs_path(gb).get('max_abs_diff', 0):.2g}", " sin rezagos; ",
                             Link("corrida", dock.get("gha_build_run_url") or rd.TESTS_RUN_URL),
                             "; reproducibilidad/github_actions_build.json). Ambos flujos publican el JSON como "
                             "anotaciones del job, de donde se recuperó."]
    para(ctx, ("b", "Análisis. "), "Con el modelo en Git, la imagen se reconstruye en cualquier clon y el CI la "
              "construye en cada push (job docker de tests.yml) además de verificar la imagen publicada "
              "(docker-reproducibilidad.yml). Así la prueba de \"otro entorno\" es literal: máquinas que nunca vieron "
              "los datos ni la base de MLflow reproducen las predicciones del modelo versionado. La versión 1.0.0 se "
              "verificó en el runner de GitHub Actions con PASS (",
         Link(f"corrida {dock['gha_v1_run_url'].rsplit('/', 1)[-1]}", dock["gha_v1_run_url"]), ").", gha_txt,
         " La Figura ", ctx.ref("f:hub"), " muestra el repositorio en DockerHub y la Figura ", ctx.ref("f:gha"),
         " las corridas de GitHub Actions de ambas ramas.")
    figure(ctx, "f:hub", FIG / "15_dockerhub_tags.png", "Repositorio gusrivasmna/mibici-demand-service en DockerHub",
           ["captura tomada cuando latest apuntaba a 1.0.0 (121.06 MB en unidades binarias = 126,937,001 bytes); los "
            "tags vigentes y sus digests están en la Tabla ", ctx.ref("t:tags"), "."], width=4.4)
    figure(ctx, "f:gha", FIG / "16_github_actions.png", "Corridas de GitHub Actions (tests y docker-reproducibilidad)",
           ["incluye las corridas de la rama entrega-final (PR #5) y de fase-final; todas en verde al momento de la "
            "captura."], width=4.2)

    # 4.12 Drift
    heading(ctx, "4.12 Monitoreo de data drift", 2)
    table(ctx, "t:m-drift", "Monitoreo: método, alternativas y justificación", ["Decisión", "Alternativas",
                                                                               "Justificación"],
          [["Referencia de distribución = validación; línea base de desempeño = prueba", "Usar entrenamiento como "
                                                                                         "referencia",
            "Validación es el periodo más reciente sin mezclar estaciones del año; el modelo final vio validación, "
            "por eso el desempeño se compara contra prueba."],
           ["PSI, estadístico D de Kolmogorov-Smirnov y Wasserstein normalizada", "Solo p-valores; Evidently",
            f"Con {n(ref_rows)} filas cualquier p-valor es ~0; se decide con tamaños de efecto (Massey, 1951). PSI es la "
            "convención de la industria."],
           ["Dos familias de escenarios: corrupción de entradas y cambio real (recalculando variables y objetivo "
            "desde los viajes)", "Solo perturbar columnas", "Distinguir si hay que arreglar el pipeline o "
                                                            "reentrenar; reentrenar con datos corruptos aprendería el "
                                                            "error."],
           ["Escenario de control sin alteración", "Sin control", "Demuestra que los umbrales no generan falsas "
                                                                  "alarmas."],
           ["MAE y WAPE con el mismo umbral; reglas con y sin etiquetas", "Solo MAE", "El MAE baja cuando baja la "
                                                                                      "demanda; en producción la "
                                                                                      "etiqueta llega tarde (Gama et "
                                                                                      "al., 2014)."]],
          widths=[2.4, 1.4, 2.7], size=8.5)
    th = drift.get("thresholds", {})
    th_labels = {"psi": "PSI por variable", "ks_statistic": "Estadístico D de KS", "wasserstein_norm":
                 "Wasserstein / desv. est.", "missing_rate": "Aumento de faltantes", "count_domain_violation_rate":
                 "Rezagos fuera de dominio", "unseen_category_rate": "Estaciones no vistas",
                 "hour_profile_shift_hours": "Desfase del perfil horario (h)", "mae_degradation":
                 "Degradación del MAE", "wape_degradation": "Degradación del WAPE"}
    table(ctx, "t:umbrales", "Umbrales de aviso y alerta (config.yaml, sección monitoring.thresholds)",
          ["Señal", "Aviso (≥)", "Alerta (≥)"], [[th_labels.get(k, k), str(v.get("warning")), str(v.get("alert"))]
                                                for k, v in th.items()], widths=[3.4, 1.4, 1.4], size=9,
          align=["l", "r", "r"], note=["se fijaron antes de ver los resultados y no se recalibraron; justificación de "
                                       "cada uno en reports/drift/drift_report.md."])
    para(ctx, "La Tabla ", ctx.ref("t:escenarios"), " resume los resultados contra la línea base de prueba (MAE ",
         f(base.get("mae")), ", WAPE ", f(base.get("wape")), ").", keep=True)
    srows, sfills = [], {}
    fam = {"control": "Control", "corrupcion_entradas": "Corrupción", "cambio_real": "Cambio real"}
    for i, s in enumerate(drift.get("scenarios", [])):
        lvl = s["decision"]["level"]
        srows.append([s["name"], fam.get(s["family"], s["family"]), f(s["performance"]["mae"]),
                      pc(s["degradation"]["mae"]), pc(s["degradation"]["wape"]),
                      f"{s['levels']['quality']} / {s['levels']['drift']} / {s['levels']['performance']}",
                      lvl, s["decision_without_labels"]["level"], s["decision"]["rule"]])
        sfills[(i, 6)] = LEVEL_FILL.get(lvl, "FFFFFF")
        sfills[(i, 7)] = LEVEL_FILL.get(s["decision_without_labels"]["level"], "FFFFFF")
    table(ctx, "t:escenarios", "Resultados de los escenarios de monitoreo", ["Escenario", "Familia", "MAE", "ΔMAE",
                                                                            "ΔWAPE", "Calidad / drift / desempeño",
                                                                            "Nivel", "Sin etiquetas", "Regla"],
          srows, widths=[1.25, 0.75, 0.5, 0.55, 0.62, 1.2, 0.68, 0.7, 0.9], size=7,
          align=["l", "l", "r", "r", "r", "c", "c", "c", "l"], fills=sfills,
          note=["reports/drift/drift_summary.csv y drift_report.json. “Sin etiquetas” es lo que diría el monitoreo "
                "antes de conocer la demanda real (solo calidad y drift de entradas)."])
    figure(ctx, "f:dist", FIG / "07_drift_distribuciones.png", "Distribuciones de referencia contra escenarios",
           ["el panel inferior derecho muestra el perfil horario del rezago de 24 h: con 6 h de desfase el pico se "
            "corre y la correlación con la referencia pasa de "
            f"{val(scen, 'control', 'quality', 'hour_profile', 'corr_at_zero') or 0:.3f} a "
            f"{val(scen, 'reloj_desfasado', 'quality', 'hour_profile', 'corr_at_zero') or 0:.3f}."], width=6.2)
    orientation(ctx, True)
    figure(ctx, "f:psi", FIG / "08_drift_psi_por_variable.png", "PSI por variable y escenario contra la referencia",
           ["las variables de calendario (rayadas) son informativas y no deciden el nivel: son funciones de la fecha. "
            "El reloj desfasado no altera ninguna distribución marginal y solo se detecta en la predicción y en el "
            "desfase del perfil horario."], width=9.2)
    ctx.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    figure(ctx, "f:perf", FIG / "09_drift_desempeno.png", "Desempeño del champion por escenario contra la línea base",
           ["bandas de aviso y alerta según config.yaml. En la caída de demanda y en vacaciones el MAE baja pero el "
            "WAPE sube: el error relativo empeora aunque el absoluto mejore."], width=9.0)
    orientation(ctx, False)
    fw = windows[6] if len(windows) > 6 else None
    figure(ctx, "f:win", FIG / "10_drift_ventanas_tiempo.png", "Monitoreo continuo del conjunto de prueba en ventanas "
                                                              "de 7 días",
           ["sin alteración: es el drift natural de los datos reales. La ventana que inicia el ",
            fw["start"][:10] if fw else "s/d", f" pasa a ALERTA con WAPE {pc(val(fw, 'degradation', 'wape'))} y sesgo "
                                               f"{pc(val(fw, 'performance', 'bias'))}." if fw else ""], width=5.2)
    wrows, wfills = [], {}
    for i, w in enumerate(windows):
        wrows.append([str(w["window"]), w["start"][:10], n(w["rows"]), f(w["performance"]["mae"]),
                      pc(w["degradation"]["mae"]), pc(w["degradation"]["wape"]), pc(w["performance"]["bias"]),
                      w["decision"]["level"], w["decision"]["rule"]])
        wfills[(i, 7)] = LEVEL_FILL.get(w["decision"]["level"], "FFFFFF")
    table(ctx, "t:ventanas", "Ventanas semanales del conjunto de prueba", ["#", "Inicio", "Filas", "MAE", "ΔMAE",
                                                                          "ΔWAPE", "Sesgo", "Nivel", "Regla"],
          wrows, widths=[0.3, 0.9, 0.7, 0.6, 0.7, 0.7, 0.7, 0.7, 1.2], size=8,
          align=["c", "l", "r", "r", "r", "r", "r", "c", "l"], fills=wfills,
          note=["drift_report.json (windows); la última ventana está incompleta."])
    sc = scen
    para(ctx, ("b", "Análisis de resultados. "), "El escenario de control queda en OK en las tres dimensiones, lo que valida "
              "que los umbrales no producen falsas alarmas con datos normales. Los tres escenarios de corrupción se "
              "detectan sin necesidad de etiquetas y la regla R1 propone revisar el pipeline antes de reentrenar. Los "
              "escenarios de cambio real muestran por qué el MAE solo no basta: con 40 % menos viajes el MAE cambia ",
         pc(val(sc, "caida_demanda", "degradation", "mae")), " pero el WAPE cambia ",
         pc(val(sc, "caida_demanda", "degradation", "wape")), ", y en las vacaciones de diciembre el MAE cambia ",
         pc(val(sc, "vacaciones_diciembre", "degradation", "mae")), " mientras el WAPE cambia ",
         pc(val(sc, "vacaciones_diciembre", "degradation", "wape")), " con un sesgo de ",
         pc(val(sc, "vacaciones_diciembre", "performance", "bias")), ": el modelo sigue esperando la demanda de "
                                                                      "semanas normales. Con estaciones nuevas, el "
                                                                      "drift de las variables es mínimo, pero la regla "
                                                                      "de calidad detecta ",
         pc(val(sc, "estaciones_nuevas", "quality", "unseen_category_rate"), 1, False), " de filas con estaciones no "
                                                                                        "vistas.")
    para(ctx, ("b", "Acción propuesta. "), "Cada nivel se traduce en una acción concreta (reglas R0 a R7 en "
              "drift_report.md): revisar el pipeline de variables ante fallas de calidad (R1); actualizar el catálogo "
              "y vigilar aparte las estaciones nuevas (R2); reentrenar con datos recientes, evaluar al challenger en "
              "la misma ventana y mover el alias si mejora cuando hay drift con degradación (R3); y vigilar o preparar "
              "un challenger cuando hay drift sin etiquetas (R6, R7). Para MiBici, la recomendación operativa es "
              "reentrenar antes del 20 de diciembre incorporando el calendario de vacaciones, porque la ventana real "
              "de esas fechas ya cruza el umbral de alerta.")
    para(ctx, ("b", "Limitaciones. "), "Los umbrales son convenciones (PSI) o criterios del equipo, no se calibraron con "
              "el costo de un error de rebalanceo; el escenario de escala no se detectaría con un factor entero (×2) "
              "porque no rompe el dominio de los conteos; y la simulación es por lotes sobre datos históricos, no un "
              "monitoreo en línea.")

    # ============================================================================ 5. VERIFICACIÓN
    heading(ctx, "5. Verificación de los requisitos de la entrega final", 1, page_break=True)
    para(ctx, "La Tabla ", ctx.ref("t:req"), " recorre los cinco puntos de la entrega inciso por inciso con la evidencia "
              "que permite verificarlos.")
    reqs = [
        ["1. Pruebas", "Pruebas unitarias de funciones clave", "tests/unit (preprocesamiento, métricas, inferencia, "
                                                               "limpieza, características)", "pytest -q -m \"not integration\""],
        ["1. Pruebas", "Pruebas de integración del pipeline", "tests/integration: carga → limpieza → características → "
                                                              "entrenamiento → Registry → predicción → métricas",
         "pytest -q -m integration"],
        ["1. Pruebas", "Comando único y documentación", f"pytest -q: {n(pt.get('passed'))} aprobadas, cobertura "
                                                        f"{pt.get('coverage')} %", "README (Pruebas), tests/README.md"],
        ["2. API", "Endpoint POST /predict con FastAPI", "src/serving/app.py", "docs/api.md, 20_api_smoke.log"],
        ["2. API", "Validación de entrada y salida con Pydantic", "src/serving/schemas.py; errores 422 uniformes",
         "tests/test_serving.py"],
        ["2. API", "Esquema documentado", "OpenAPI exportado, Swagger en /docs, colección de Postman",
         "docs/openapi.json, docs/postman_collection.json"],
        ["2. API", "Ruta y versión del artefacto en el README", [("c", model_uri or ""), ", alias champion, carpeta "
                                                                                         "models/champion, SHA-256"],
         "README (Artefacto del modelo)"],
        ["3. Reproducibilidad", "Dependencias fijadas y semillas", "requirements*.txt con ==; seed 42 y "
                                                                  "set_global_seeds", "requirements.txt, config.yaml"],
        ["3. Reproducibilidad", "Ejecución en entorno limpio y comparación", f"MAE {f(test_mae)} vs {f(test_mae2)}; "
                                                                             f"{len(ok_envs)} entornos PASS",
         "comparacion_fase2.md, reproducibilidad/*.json"],
        ["3. Reproducibilidad", "Artefactos versionados (DVC/MLflow)", ["MLflow Registry + SHA-256; DVC con "
                                                                       "dvc.lock; modelo exportado en Git ",
                                                                       None if dvc["lock"] else Pend("dvc.lock")],
         "mlruns.db (local), dvc.yaml, dvc.lock, models/"],
        ["4. Docker", "Dockerfile del servicio", "Multi-etapa, no root, HEALTHCHECK", "Dockerfile, docs/docker.md"],
        ["4. Docker", "docker build -t ml-service:latest . y docker run -p 8000:8000 ml-service:latest",
         "Construcción y ejecución verificadas, también desde un clon limpio y en el CI (job docker)",
         "docker_build_v1.1.0.log, clon_limpio.json, tests.yml"],
        ["4. Docker", "Publicación en DockerHub con tags versionados",
         [f"{rd.DOCKER_IMAGE}: ", ", ".join(sorted(hub_tags)) if hub_tags else Pend("tags publicados")],
         [Link("DockerHub", rd.DOCKERHUB_URL), ", dockerhub_tags.json"]],
        ["5. Drift", "Datos de monitoreo con distribución alterada", f"{len(scen)} escenarios en dos familias + "
                                                                    f"{len(windows)} ventanas reales",
         "src/monitoring/scenarios.py"],
        ["5. Drift", "Métricas contra la línea base", "MAE, RMSE, R², devianza, WAPE y sesgo vs. prueba",
         "drift_summary.csv"],
        ["5. Drift", "Gráficos", "Figuras 07 a 10", "reports/figures/"],
        ["5. Drift", "Umbrales, criterios de alerta y acción propuesta", "OK/AVISO/ALERTA y reglas R0-R7",
         "config.yaml, drift_report.md"],
        ["Entrega", "PDF con ligas a GitHub y DockerHub", "Portada y sección 7 de este documento",
         "Entrega_Final_Equipo19.pdf"],
    ]
    table(ctx, "t:req", "Requisitos de la entrega final, cumplimiento y evidencia", ["Punto", "Requisito",
                                                                                     "Cómo se cumplió", "Evidencia"],
          reqs, widths=[1.0, 1.9, 2.0, 1.6], size=8)

    # ============================================================================ 6. CONCLUSIONES
    heading(ctx, "6. Conclusiones generales", 1, page_break=True)
    para(ctx, "El proyecto pasó de un notebook con un mes de datos a un sistema que puede auditarse y operarse: el "
              "modelo champion (", uri, f") reduce a la mitad el error de la referencia ingenua usando solo información "
              "disponible con 24 horas de anticipación; se sirve con una API validada, se empaqueta en una imagen "
              "versionada que se reconstruye desde un clon limpio, se reproduce en otros entornos con diferencias del orden "
              "de 1e-16, y cuenta con un monitoreo que distingue fallas del pipeline de cambios reales y propone qué "
              "hacer en cada caso. La solución final integra dos aproximaciones complementarias que se verificaron "
              "entre sí.")
    heading(ctx, "6.1 Lecciones aprendidas", 2)
    for a, b in [
        ("La validez de una métrica depende de cómo se obtuvo. ", "El MAE de 1.344 de la Fase 1 era mejor en "
         "apariencia que cualquier resultado posterior, pero usaba un rezago que no existe en operación. Una métrica "
         "honesta y peor vale más que una buena e irrepetible."),
        ("Los datos cambian aunque el código no. ", "El portal republicó archivos y retiró el catálogo. Sin hashes "
         "habríamos atribuido la diferencia al modelo."),
        ("Reproducible no siempre significa idéntico bit a bit. ", "El artefacto se reproduce entre sistemas; el "
         "reentrenamiento no. Hay que declarar qué se compara y con qué tolerancia."),
        ("Una sola métrica puede engañar. ", "El MAE mejora cuando la demanda baja; el WAPE revela que el error "
         "relativo empeoró. El monitoreo debe vigilar métricas complementarias."),
        ("El monitoreo necesita controles. ", "El escenario sin alteración evitó que los umbrales dispararan falsas "
         "alarmas y obligó a separar las variables de calendario, que cambian por construcción."),
        ("Contrastar dos aproximaciones funciona como verificación cruzada. ", "Servir el binario de la Fase 2 y "
         "reentrenar en limpio confirmó que el resultado se reproduce, y el contraste resolvió decisiones de diseño "
         "(quién calcula los rezagos, si el modelo debe estar en Git) con la mejor respuesta de cada aproximación, "
         "verificada con las pruebas de ambas."),
    ]:
        bullet(ctx, ("b", a), b)
    heading(ctx, "6.2 Áreas de mejora reconocidas en el análisis del caso", 2)
    mejoras = [
        ["Faltan variables externas (clima, eventos, vacaciones escolares)", "Diciembre: WAPE "
         f"{pc(val(sc, 'vacaciones_diciembre', 'degradation', 'wape'))} y sesgo "
         f"{pc(val(sc, 'vacaciones_diciembre', 'performance', 'bias'))}",
         "Incorporar lluvia y temperatura por hora y un calendario escolar y de eventos."],
        ["El modelo se ancla al nivel histórico de cada estación", "Demanda ±40 %: alerta de desempeño en ambos "
                                                                   "sentidos",
         "Variables de tendencia reciente y reentrenamiento programado."],
        ["Un solo horizonte y solo extracciones", "Rezagos fijos de 24 h; el rebalanceo depende del flujo neto",
         "Modelos por horizonte y un segundo objetivo de llegadas por estación-hora."],
        ["Un solo corte temporal de validación", "Selección con un periodo (sep-nov)",
         "Validación cruzada temporal con ventanas móviles."],
        ["Umbrales por convención, sin costo de negocio", "PSI 0.10/0.25 y degradación 10/25 %",
         "Calibrar con el costo de un viaje frustrado y de un traslado de camioneta."],
        ["Sin despliegue continuo ni monitoreo en línea", "Drift por lotes sobre datos históricos",
         "Pipeline programado que publique imagen y reporte de drift con cada mes nuevo."],
        ["Etiquetas con retraso", "La regla de desempeño confirma tarde", "Mantener reglas sin etiquetas y medir el "
                                                                         "tiempo de detección."],
        ["Detección de errores de escala limitada", "Un factor entero no rompe el dominio de conteos",
         "Conciliar los rezagos con los viajes registrados del día anterior."],
        ["Dependencia de una fuente que cambia", "CSV republicados y catálogo retirado",
         "Remoto de DVC con instantáneas de cada descarga y validación de esquema por mes."],
        ["Regla operativa del Canvas sin validar", "No hay datos de inventario en tiempo real",
         "Integrar el inventario por estación para medir el impacto real en desabasto."],
        ["Feature store estático", f"Ventana predecible sin rezagos fija ({fsw.get('start', 's/d')[:10]} a "
                                   f"{fsw.get('end', 's/d')[:10]})",
         "Actualizar el feature store con cada ingesta o leer el histórico reciente en línea."],
        ["Interfaz común entre aproximaciones", "La integración requirió adaptar el formato del modelo exportado y "
                                                "del servicio", "Definir desde el inicio la interfaz común (artefacto, "
                                                               "esquema de la API) para integrar con menos adaptación."],
    ]
    table(ctx, "t:mejoras", "Áreas de mejora, evidencia y propuesta", ["Área de mejora", "Evidencia", "Propuesta"],
          mejoras, widths=[2.2, 2.0, 2.3], size=8.5)
    heading(ctx, "6.3 Estrategias empleadas en el análisis y la solución", 2)
    estr = [
        ["Partir del ML Canvas", "Las características del problema (conteos, horizonte, serie temporal) "
                                 "decidieron pérdida, variables y partición", "1.1"],
        ["Línea base primero", "Dummy como piso de toda comparación", "4.5"],
        ["Evitar fuga de información", "Rezagos ≥ 24 h, partición cronológica, test usado una vez", "4.3, 4.5"],
        ["Configuración única", "config.yaml para rutas, reglas, modelos, umbrales y escenarios", "4.7"],
        ["Automatizar y probar", f"{n(pt.get('passed'))} pruebas, CI en cada push", "4.8"],
        ["Fijar el entorno", "Versiones exactas, semillas, digest de la imagen", "4.10, 4.11"],
        ["Verificar entre entornos", "500 predicciones de referencia con tolerancia explícita", "4.10"],
        ["Escenario de control", "Validar que los umbrales no den falsas alarmas", "4.12"],
        ["Separar causas", "Corrupción de entradas frente a cambio real, con acciones distintas", "4.12"],
        ["Documentar decisiones", "Tablas de alternativas, logs de cada ejecución, comparación con la fase anterior",
         "4, reports/fase_final/"],
        ["Segunda aproximación de verificación", "Trabajar una segunda aproximación de verificación e integrar lo "
                                                 "mejor de ambas", "2.5"],
    ]
    table(ctx, "t:estrategias", "Estrategias empleadas", ["Estrategia", "Cómo se aplicó", "Sección"], estr,
          widths=[1.8, 3.9, 0.8], size=8.5)
    heading(ctx, "6.4 Trabajo futuro y recomendaciones", 2)
    for t_ in ["Programar la ingesta mensual y el reporte de drift como un flujo automático que abra un pull request "
               "con el reentrenamiento cuando aplique la regla R3.",
               "Configurar un remoto de DVC para los datos crudos y procesados (el modelo ya está en Git), de modo "
               "que se recuperen con dvc pull aunque el portal cambie o retire archivos.",
               "Agregar clima y calendario escolar, y evaluar modelos por horizonte (1, 6 y 24 h).",
               "Actualizar el feature store con cada ingesta para que la ventana predecible sin rezagos avance con "
               "los datos.",
               "Exponer métricas del servicio (latencia, errores 4xx/5xx) y registrar las predicciones para cerrar el "
               "ciclo con las etiquetas reales.",
               "Recomendación a MiBici: usar el modelo para priorizar estaciones por riesgo en las horas pico y "
               "reentrenarlo antes de periodos vacacionales."]:
        bullet(ctx, t_)
    heading(ctx, "6.5 Reflexiones individuales", 2)
    guides = {
        "antonio": "Antonio (DE/MLE): escribe un párrafo sobre la etapa de datos y el ciclo de vida del modelo. "
                   "Sugerencias: qué implica que el portal republique CSV, el valor de los hashes y de DVC, y cómo el "
                   "modelo pasa del Registry al servicio con su SHA-256. Pega aquí el texto de la sección Reflexión de "
                   "reports/fase_final/antonio.md.",
        "jesus": "Jesús (DS): escribe un párrafo sobre la aproximación del binario de la Fase 2 (rama entrega-final) "
                 "y los resultados. Sugerencias: por qué el feature store y Podman, qué aportó contrastar las dos "
                 f"aproximaciones y qué significa un WAPE de {pc(base.get('wape'), 0, False)} para la operación. Pega "
                 "aquí la sección Reflexión de reports/fase_final/jesus.md.",
        "erick": "Erick (SWE): escribe un párrafo sobre la calidad de software. Sugerencias: qué protege la suite de "
                 "pruebas, qué revela la cobertura, la prueba de coherencia servicio-pipeline, el job docker del CI y "
                 "la lección de las pruebas de igualdad exacta en Linux. Pega aquí la sección Reflexión de "
                 "reports/fase_final/erick.md.",
    }
    for m in rd.MEMBERS:
        text = member_section(m["key"], "Reflexión")
        p = para(ctx, ("b", f"{m['name']} ({m['role']}). "))
        if text and m["key"] != "gustavo":
            from docx_helpers import _add_parts
            _add_parts(ctx, p, [re.sub(r"\s*\n\s*", " ", text)])
        elif text:
            from docx_helpers import _add_parts
            _add_parts(ctx, p, [re.sub(r"\s*\n\s*", " ", text)])
        else:
            from docx_helpers import _add_parts
            _add_parts(ctx, p, [Guide(guides.get(m["key"], "Escribe tu reflexión."))])

    # ============================================================================ 7. REFERENCIAS
    heading(ctx, "7. Referencias y ligas", 1, page_break=True)
    heading(ctx, "7.1 Ligas del proyecto", 2)
    links = [("Repositorio de GitHub", rd.REPO_URL), ("Imagen en DockerHub", rd.DOCKERHUB_URL),
             ("Rama de la entrega final", f"{rd.REPO_URL}/tree/fase-final"),
             ("Rama entrega-final (solución paralela, PR #5)", f"{rd.REPO_URL}/tree/entrega-final"),
             ("Ejecuciones de GitHub Actions", rd.ACTIONS_URL),
             ("Historial de commits (todos los integrantes)", f"{rd.REPO_URL}/commits/fase-final"),
             ("Documentación de la API", f"{rd.REPO_URL}/blob/fase-final/docs/api.md"),
             ("Reporte de data drift", f"{rd.REPO_URL}/blob/fase-final/reports/drift/drift_report.md"),
             ("Datos abiertos de MiBici", "https://www.mibici.net/es/datos-abiertos/")]
    for lab, url in links:
        bullet(ctx, f"{lab}: ", Link(url, url)).alignment = WD_ALIGN_PARAGRAPH.LEFT
    bullet(ctx, "Pull request de la entrega final: ",
           Link(f"{rd.REPO_URL}/pull/{final_pr}", f"{rd.REPO_URL}/pull/{final_pr}") if final_pr else
           Pend("liga del PR de fase-final a main")).alignment = WD_ALIGN_PARAGRAPH.LEFT
    heading(ctx, "7.2 Referencias bibliográficas", 2)
    refs = [
        "Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter "
        "optimization framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery "
        "& Data Mining, 2623-2631.",
        "Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. "
        "Advances in Neural Information Processing Systems, 24, 2546-2554.",
        "Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML test score: A rubric for ML "
        "production readiness and technical debt reduction. 2017 IEEE International Conference on Big Data, "
        "1123-1132.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
        "Dorard, L. (2020). Machine Learning Canvas. https://www.ownml.co/machine-learning-canvas",
        "DrivenData. (2024). Cookiecutter Data Science (v2). https://cookiecutter-data-science.drivendata.org/",
        "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of "
        "Statistics, 29(5), 1189-1232.",
        "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift "
        "adaptation. ACM Computing Surveys, 46(4), Artículo 44.",
        "Huyen, C. (2022). Designing machine learning systems: An iterative process for production-ready "
        "applications. O'Reilly Media.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly "
        "efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, "
        "3146-3154.",
        "Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine learning operations (MLOps): Overview, definition, "
        "and architecture. IEEE Access, 11, 31866-31879.",
        "Ley Federal del Trabajo, Artículo 74. Diario Oficial de la Federación, México.",
        "Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. Journal of the American Statistical "
        "Association, 46(253), 68-78.",
        "Merkel, D. (2014). Docker: Lightweight Linux containers for consistent development and deployment. Linux "
        "Journal, 2014(239), Artículo 2.",
        "Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in "
        "classification and prediction problems. ACM SIGKDD Explorations Newsletter, 3(1), 27-32.",
        "MiBici. (2025). Datos abiertos del Sistema de Bicicletas Públicas del Área Metropolitana de Guadalajara. "
        "https://www.mibici.net/es/datos-abiertos/",
        "Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear models. Journal of the Royal Statistical "
        "Society: Series A, 135(3), 370-384.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, "
        "P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & "
        "Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, "
        "2825-2830.",
        "Ramírez, S. (2018). FastAPI [Software]. https://fastapi.tiangolo.com/",
        "Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, "
        "J. F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. Advances in Neural "
        "Information Processing Systems, 28, 2503-2511.",
        "Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & "
        "Heidmann, L. (2020). Introducing MLOps. O'Reilly Media.",
        "Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., "
        "Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with "
        "MLflow. IEEE Data Engineering Bulletin, 41(4), 39-45.",
    ]
    for r_ in refs:
        p = para(ctx, r_, align=WD_ALIGN_PARAGRAPH.LEFT, size=11, space_after=4)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)

    # ============================================================================ 8. IA
    heading(ctx, "8. Declaración de uso de herramientas de inteligencia artificial", 1)
    para(ctx, "En apego a los lineamientos de la actividad, se declara el uso de herramientas de inteligencia "
              "artificial en el proyecto (detalle por fase en reports/declaracion_ia.md).")
    for txt in [
        "Fase 1: Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como herramienta de "
        "consulta y de apoyo en la revisión de la redacción, la organización, el formato del reporte y la rúbrica.",
        "Fase 2: Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande], utilizado como apoyo en la "
        "reestructuración del repositorio a Cookiecutter Data Science v2, la refactorización a módulos configurables, "
        "el código del pipeline con Optuna, MLflow Tracking y Model Registry, las pruebas y la documentación.",
        "Fase final: Anthropic. (2026). Claude (Fable 5.2 y Opus 5.5) [Modelo de lenguaje grande], utilizado como "
        "apoyo en las pruebas con pytest, el servicio con FastAPI y el feature store, el Dockerfile y la publicación "
        "de las imágenes, los scripts de verificación de reproducibilidad, la configuración de DVC, la simulación de "
        "data drift, la comparación e integración de las ramas fase-final y entrega-final, el diagrama de la "
        "solución, el generador de este reporte y la redacción y revisión de la documentación. https://claude.ai/"]:
        bullet(ctx, txt)
    para(ctx, "El equipo revisó, ejecutó y validó el código. Todas las métricas, tablas y figuras de este reporte "
              "corresponden a ejecuciones reales registradas en los logs y artefactos del repositorio, y se generan "
              "por código a partir de ellos (reports/build_report.py).")


# --------------------------------------------------------------------------------------------- main
def word_postprocess(docx: Path, pdf: Path | None) -> str:
    """Actualiza el índice y los campos con Word (Windows) y, opcionalmente, exporta a PDF."""
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return "Word no disponible: abre el documento y actualiza el índice con F9."
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        d = word.Documents.Open(str(docx.resolve()))
        for t in d.TablesOfContents:
            t.Update()
        d.Fields.Update()
        for t in d.TablesOfContents:
            t.Update()
        d.Save()
        if pdf:
            d.SaveAs2(str(pdf.resolve()), FileFormat=17)
        pages = d.ComputeStatistics(2)
        d.Close(False)
        return f"Índice actualizado con Word ({pages} páginas)" + (f"; PDF: {pdf}" if pdf else "")
    finally:
        word.Quit()


def update_readme_gha(D) -> str:
    """Llena las filas de GitHub Actions de la tabla de reproducibilidad del README con los JSON del runner."""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    dock = D["docker"]
    done = []
    for prefix, data, url, fname in (
            ("| GitHub Actions, imagen construida desde el repo", dock.get("gha_build"), dock.get("gha_build_run_url"),
             "github_actions_build.json"),
            ("| GitHub Actions, imagen publicada", dock.get("gha"), dock.get("gha_run_url"), "github_actions.json")):
        if not data:
            continue
        ep = val(data, "endpoints", "predict") or {}
        fp = val(data, "feature_store", "predict_feature_store") or {}
        run = f", [corrida {url.rsplit('/', 1)[-1]}]({url})" if url else ""
        row = (f"{prefix} | {ep.get('max_abs_diff'):.3g} | {ep.get('rows_bitwise_identical')}/{ep.get('rows')} | "
               f"{fp.get('max_abs_diff', 0):.3g} ({fp.get('rows_bitwise_identical')}/{fp.get('rows')}) | "
               f"{data.get('verdict')} | `reproducibilidad/{fname}`{run} |")
        new_text, k = re.subn(rf"^{re.escape(prefix)}.*$", lambda _: row, text, flags=re.M)
        if k:
            text = new_text
            done.append(fname)
    readme.write_text(text, encoding="utf-8", newline="\n")
    return f"README: filas de GitHub Actions actualizadas con {', '.join(done) or 'ningún JSON'}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--date", help="Fecha de la portada (AAAA-MM-DD); por omisión, hoy")
    ap.add_argument("--no-word", action="store_true", help="No abrir Word para actualizar el índice")
    ap.add_argument("--pdf", type=Path, help="Exportar también a PDF con Word")
    ap.add_argument("--readme", action="store_true", help="Actualizar la fila de GitHub Actions del README")
    args = ap.parse_args()
    report_date = date.fromisoformat(args.date) if args.date else date.today()
    D = rd.load_all()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nums = {}
        for _ in range(2):  # dos pasadas: la segunda conoce el número de cada tabla y figura citada antes
            doc = new_document()
            page_number_footer(doc, "Entrega final · Equipo 19 · TC5061")
            ctx = Ctx(doc=doc, nums=nums)
            build(ctx, D, report_date, tmp)
            nums = dict(ctx.assigned)
        if args.no_word:  # sin Word, el índice se genera al abrir el documento
            set_update_fields_on_open(doc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        doc.save(args.output)
    print(f"Reporte: {args.output}")
    print(f"Figuras: {ctx.counters['fig']} | Tablas: {ctx.counters['tab']}")
    print(f"Marcadores [PENDIENTE] ({len(ctx.pending)}):")
    for p_ in dict.fromkeys(ctx.pending):
        print(f"  - {p_}")
    print(f"Textos guía en amarillo: {len(ctx.guides)}")
    if args.readme:
        print(update_readme_gha(D))
    if not args.no_word:
        try:
            print(word_postprocess(args.output, args.pdf))
        except Exception as exc:  # noqa: BLE001 - Word es opcional
            print(f"No se pudo actualizar con Word ({exc}); actualiza el índice con F9.")


if __name__ == "__main__":
    main()
