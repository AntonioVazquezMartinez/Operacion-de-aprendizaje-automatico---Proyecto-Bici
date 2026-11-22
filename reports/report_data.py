"""Carga de las cifras del reporte final desde los artefactos del repositorio.

Ninguna cifra del reporte se escribe a mano: todo se lee aquí de CSV, JSON, Markdown y logs. Si un artefacto
no existe, el valor queda en None y el generador (`reports/build_report.py`) inserta un marcador
[PENDIENTE: ...] resaltado.

El resumen del EDA (`reports/fase_final/eda_resumen.json`) se calcula una sola vez a partir de
`data/interim` y `data/processed` (que no están en Git) y se reutiliza en las siguientes ejecuciones.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FF = REPORTS / "fase_final"
LOGS = FF / "logs"
REPRO = FF / "reproducibilidad"
FIGS = REPORTS / "figures"

REPO_URL = "https://github.com/AntonioVazquezMartinez/Operacion-de-aprendizaje-automatico---Proyecto-Bici"
DOCKERHUB_URL = "https://hub.docker.com/r/gusrivasmna/mibici-demand-service"
DOCKER_IMAGE = "gusrivasmna/mibici-demand-service"
ACTIONS_URL = f"{REPO_URL}/actions"
# Corrida del flujo docker-reproducibilidad en GitHub Actions (hadolint + verificación de la imagen publicada).
# Corrida que verificó la imagen 1.0.0 (su JSON está en reproducibilidad/v1.0.0/github_actions.json). La de 1.1.0 se
# toma de _fuente.run de reproducibilidad/github_actions.json cuando existe.
GHA_RUN_URL = f"{REPO_URL}/actions/runs/45060105120"
# Corrida del flujo tests (job de pruebas con cobertura y job docker que construye la imagen) del push final.
TESTS_RUN_URL = f"{REPO_URL}/actions/runs/45066506318"

# Integrantes: nombre, matrícula, rol, archivo de la fase final y alias con los que aparecen en Git.
MEMBERS = [
    {"key": "antonio", "name": "José Antonio Vázquez Martínez", "short": "Antonio", "id": "A01797208",
     "role": "Data Engineer y ML Engineer", "roles": ["DE", "MLE"],
     "aliases": ["josantoniovazquezmartinez", "joseantoniovazquezmartinez", "antoniovazquezmartinez-tec",
                 "a01797208"]},
    {"key": "jesus", "name": "Jesús Ángel Rincón Ruiz", "short": "Jesús", "id": "A01793960",
     "role": "Data Scientist", "roles": ["DS"], "aliases": ["jesús rincón", "jesus rincon", "a01793960"]},
    {"key": "erick", "name": "Erick Francisco Siller Ojeda", "short": "Erick", "id": "A01382929",
     "role": "Software Engineer", "roles": ["SWE"], "aliases": ["erick siller", "erick2929", "a01382929"]},
    {"key": "gustavo", "name": "Gustavo Alejandro Vázquez Rivas", "short": "Gustavo", "id": "A01611559",
     "role": "Site Reliability Engineer (SRE)", "roles": ["SRE"],
     "aliases": ["gavrm2d", "gusrivastec", "a01611559", "gavr@m2d.mx"]},
]

# Fechas de corte de cada fase (entregas del 25-oct y 8-nov de 2026).
PHASES = [("Fase 1", date(2000, 1, 1), date(2026, 10, 25)),
          ("Fase 2", date(2026, 10, 26), date(2026, 11, 8)),
          ("Fase final", date(2026, 11, 9), date(2100, 1, 1))]

# Para la entrega final agregar: (N, "fase-final", "Entrega final: ...") con el número del PR.
PULL_REQUESTS = [
    (1, "fase1-datos-reales", "Fase 1: descarga de datos reales, preprocesamiento y baselines en MLflow"),
    (2, "fase1-swe-reproducibilidad", "Fase 1: Makefile, pyproject, CI y notebook de EDA ejecutado"),
    (3, "jesus/declaracion-ia", "Fase 1: declaración de uso de IA"),
    (4, "fase2-modelado", "Fase 2: Cookiecutter v2, config.yaml, Optuna, MLflow Tracking y Model Registry"),
    (5, "entrega-final", "Fase final en paralelo (Jesús Rincón): feature store, Podman/DockerHub, CI con build; abierto"),
]
REPRO_V1 = REPRO / "v1.0.0"


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_json(path: Path) -> Optional[Any]:
    text = read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_csv(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def md_tables(text: str) -> List[Dict[str, Any]]:
    """Tablas de un Markdown con el encabezado de sección (##) bajo el que aparecen."""
    tables, section, lines = [], "", text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            tables.append({"section": section, "header": header, "rows": rows})
            continue
        i += 1
    return tables


# ----------------------------------------------------------------------------------------------- loaders
def load_models() -> Dict[str, Any]:
    cur = read_csv(REPORTS / "model_comparison.csv")
    ref = read_csv(FF / "model_comparison_fase2_referencia.csv")
    conv = lambda rows: {r["model"]: {k: (float(v) if k not in ("model", "run_id") else v) for k, v in r.items()}
                         for r in rows} if rows else None  # noqa: E731
    return {"current": conv(cur), "fase2": conv(ref)}


def load_comparacion() -> Dict[str, Any]:
    text = read_text(FF / "comparacion_fase2.md")
    if not text:
        return {}
    out = {"tables": md_tables(text)}
    for t in out["tables"]:
        if t["section"].startswith("2."):
            out["data"] = t
        if t["section"].startswith("1."):
            out["metrics"] = t
    m = re.search(r"tard[oó] (\d+) s", text)
    out["train_seconds"] = int(m.group(1)) if m else None
    return out


def load_pytest() -> Dict[str, Any]:
    text = read_text(LOGS / "10_pytest_cobertura.log")
    out: Dict[str, Any] = {"available": text is not None}
    if not text:
        return out
    m = re.search(r"(\d+) passed(?:, (\d+) warnings?)? in ([\d.]+)s", text)
    if m:
        out["passed"] = int(m.group(1))
        out["seconds"] = float(m.group(3))
    out["failed"] = int(re.search(r"(\d+) failed", text).group(1)) if re.search(r"(\d+) failed", text) else 0
    tot = re.search(r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%", text, re.M)
    if tot:
        out["statements"], out["missed"], out["branches"] = int(tot.group(1)), int(tot.group(2)), int(tot.group(3))
        out["coverage"] = int(tot.group(5))
    out["modules"] = [(m.group(1).replace("\\", "/"), int(m.group(2)), int(m.group(3)), int(m.group(6)))
                      for m in re.finditer(r"^(src\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%", text, re.M)]
    legacy = read_text(LOGS / "09_pytest.log") or ""
    lm = re.search(r"(\d+) passed", legacy)
    out["legacy_passed"] = int(lm.group(1)) if lm else None
    # Funciones de prueba por archivo (def test_...), sin contar parametrizaciones.
    files = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        n = len(re.findall(r"^\s*def test_", path.read_text(encoding="utf-8"), re.M))
        files.append((path.relative_to(ROOT).as_posix(), n))
    out["files"] = files
    return out


def load_api() -> Dict[str, Any]:
    text = read_text(LOGS / "20_api_smoke.log")
    out: Dict[str, Any] = {"available": text is not None}
    if not text:
        return out
    sections = re.split(r"^##### ", text, flags=re.M)
    out["sections"] = []
    for sec in sections[1:]:
        title = sec.splitlines()[0].strip()
        codes = Counter(re.findall(r"<<< HTTP (\d+)", sec))
        repro = re.findall(r"\*\*\* (/predict(?:/raw)?)(?: \[([^\]]+)\])?: (\d+) predicciones; diferencia máxima "
                           r"absoluta = ([\d.eE+-]+); iguales bit a bit = (\d+)/(\d+)", sec)
        out["sections"].append({"title": title, "codes": dict(codes), "requests": sum(codes.values()),
                                "repro": repro})
    m = re.search(r"--- predict: una instancia.*?>>> POST /predict\n(\{.*?\n\})\n<<< HTTP (\d+)[^\n]*\n(\{.*?\n\})\n",
                  text, re.S)
    if m:
        out["example_request"], out["example_status"], out["example_response"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"--- predict SIN rezagos: el servicio.*?>>> POST /predict\n(\{.*?\n\})\n<<< HTTP (\d+)[^\n]*\n"
                  r"(\{.*?\n\})\n", text, re.S)
    if m:
        out["fs_request"], out["fs_status"], out["fs_response"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"--- 422: rezago negativo.*?<<< HTTP (\d+)[^\n]*\n(\{.*?\n\})\n", text, re.S)
    if m:
        out["error_example"] = m.group(2)
    return out


def _verify_jsons(folder: Path) -> List[Dict[str, Any]]:
    """JSON de scripts/verify_reproducibility.py (tienen `verdict`) de una carpeta."""
    envs = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        data = read_json(path)
        if isinstance(data, dict) and "verdict" in data and "endpoints" in data:
            data["_file"] = path.relative_to(ROOT).as_posix()
            envs.append(data)
    return envs


def load_repro() -> Dict[str, Any]:
    envs = _verify_jsons(REPRO)          # servicio vigente (1.1.0)
    envs_v1 = _verify_jsons(REPRO_V1)    # historial de la versión 1.0.0
    def pytest_summary(name: str) -> Optional[Dict[str, Any]]:
        text = read_text(REPRO / name)
        m = re.search(r"(?:(\d+) failed, )?(\d+) passed[^\n]*in ([\d.]+)s", text or "")
        if not m:
            return None
        return {"failed": int(m.group(1) or 0), "passed": int(m.group(2)), "seconds": float(m.group(3)),
                "file": f"reports/fase_final/reproducibilidad/{name}"}

    # Vigente: código actual + modelo real en Linux (v2). Antecedente: la corrida que motivó la tolerancia 1e-9.
    lin = pytest_summary("pipeline_linux_pytest_v2.log")
    lin_prev = pytest_summary("pipeline_linux_pytest.log")
    diag = read_text(REPRO / "diagnostico_exp" / "resumen.txt")
    standalone = read_text(LOGS / "08_verify_standalone_serve_env.log") or ""
    sm = re.search(r"max_abs_diff ([\d.eE+-]+)", standalone)
    verify = read_text(LOGS / "07_verify_reference.log") or ""
    vm = re.search(r"diferencia máxima absoluta = ([\d.eE+-]+)", verify)
    pipeline_cmp = None
    for name in ("pipeline_linux.json", "pipeline_linux_comparacion.json", "comparacion_pipeline_linux.json"):
        if (REPRO / name).exists():
            pipeline_cmp = read_json(REPRO / name)
            pipeline_cmp = {"file": f"reports/fase_final/reproducibilidad/{name}", "data": pipeline_cmp}
    pipeline_csv = read_csv(REPRO / "pipeline_linux_vs_windows.csv") if REPRO.exists() else None
    pipeline_md = (REPRO / "pipeline_linux_vs_windows.md").exists() if REPRO.exists() else False
    fase2_cmp = {k: read_json(REPRO / f"fase2_binario_vs_reentrenado{s}.json") for k, s in (("windows", ""),
                                                                                              ("linux", "_linux"))}
    return {"envs": envs, "envs_v1": envs_v1, "fase2_cmp": fase2_cmp, "linux_pytest": lin, "linux_pytest_prev": lin_prev, "diag": diag, "pipeline_csv": pipeline_csv, "pipeline_md": pipeline_md,
            "standalone_diff": sm.group(1) if sm else None, "verify_diff": vm.group(1) if vm else None,
            "pipeline_cmp": pipeline_cmp}


def load_docker() -> Dict[str, Any]:
    out: Dict[str, Any] = {"dockerfile": (ROOT / "Dockerfile").exists(), "docs": (ROOT / "docs" / "docker.md").exists()}
    docs = read_text(ROOT / "docs" / "docker.md") or ""
    out["tags_documented"] = sorted(set(re.findall(re.escape(DOCKER_IMAGE) + r":([\w.\-]+)", docs)))
    gha = read_json(REPRO / "github_actions.json") if REPRO.exists() else None
    out["gha"] = gha                                         # flujo docker-reproducibilidad, imagen publicada 1.1.0
    out["gha_run_url"] = (gha or {}).get("_fuente", {}).get("run")
    gha_v1 = read_json(REPRO_V1 / "github_actions.json")
    out["gha_v1"] = gha_v1
    out["gha_v1_run_url"] = (gha_v1 or {}).get("_fuente", {}).get("run") or GHA_RUN_URL
    gha_build = read_json(REPRO / "github_actions_build.json") if REPRO.exists() else None
    out["gha_build"] = gha_build                             # job docker de tests.yml (imagen construida del repo)
    out["gha_build_run_url"] = (gha_build or {}).get("_fuente", {}).get("run")
    tags = read_json(REPRO / "dockerhub_tags.json") or {}
    out["hub_tags"] = {t["tag"]: t for t in tags.get("tags", [])}
    digest = read_text(REPRO / "github_actions_image_digest.json") if REPRO.exists() else None
    out["digest"] = digest.strip() if digest else None
    push_logs = [p for p in REPRO.glob("*") if re.search(r"push|dockerhub|publicacion", p.name, re.I)] \
        if REPRO.exists() else []
    out["push_logs"] = [p.relative_to(ROOT).as_posix() for p in push_logs]
    hub = read_json(REPRO / "imagen_dockerhub.json") if REPRO.exists() else None
    out["hub"] = hub
    out["published_evidence"] = bool(push_logs) or bool(gha and gha.get("verdict") == "PASS") or \
        bool(hub and hub.get("verdict") == "PASS")
    m = re.search(r"Digest publicado[^`]*`(sha256:[0-9a-f]+)`", docs)
    out["published_digest"] = m.group(1) if m else None
    m = re.search(r"\| Tamaño \| ([^|]+) \|", docs)
    out["size"] = m.group(1).strip().replace("`", "") if m else None
    m = re.search(r"\| Tiempo de build \| ([^|]+) \|", docs)
    out["build_time"] = m.group(1).strip().replace("`", "") if m else None
    build = read_text(REPRO / "docker_build_v1.1.0.log") or read_text(REPRO_V1 / "docker_build.log") or ""
    m = re.search(r"naming to docker\.io/library/([\w.\-:/]+)", build)
    out["local_build_tag"] = m.group(1) if m else None
    return out


def load_dvc() -> Dict[str, Any]:
    out = {"yaml": (ROOT / "dvc.yaml").exists(), "lock": (ROOT / "dvc.lock").exists(), "stages": [],
           "doc": (ROOT / "docs" / "dvc.md").exists(), "status": None, "lock_entries": None}
    text = read_text(ROOT / "dvc.yaml")
    if text:
        out["stages"] = re.findall(r"^  ([\w\-]+):\s*$", text, re.M)
    lock = read_text(ROOT / "dvc.lock")
    if lock:
        out["lock_entries"] = len(re.findall(r"^\s*md5: [0-9a-f]+", lock, re.M))
    exe = ROOT / ".venv" / ("Scripts" if (ROOT / ".venv" / "Scripts").exists() else "bin") / "dvc"
    for cmd in ([str(exe)], ["dvc"]):
        try:
            res = subprocess.run(cmd + ["status"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                 timeout=120)
            if res.returncode == 0:
                out["status"] = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
                break
        except (OSError, subprocess.SubprocessError):
            continue
    return out


def load_requirements() -> Dict[str, str]:
    text = read_text(ROOT / "requirements.txt") or ""
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==([\w.]+)", text, re.M)}


def load_logs() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ds = read_text(LOGS / "01_dataset.log") or ""
    m = re.search(r"(\d+) -> (\d+) filas \(retención ([\d.]+)%\)", ds)
    if m:
        out["raw_rows"], out["clean_rows"], out["retention"] = int(m.group(1)), int(m.group(2)), m.group(3)
    m = re.search(r"\((\d+) estaciones\)", ds)
    out["catalog_stations"] = int(m.group(1)) if m else None
    ft = read_text(LOGS / "02_features.log") or ""
    m = re.search(r"\((\d+), (\d+)\)", ft)
    if m:
        out["matrix_rows"], out["matrix_cols"] = int(m.group(1)), int(m.group(2))
    tr = read_text(LOGS / "04_train.log") or ""
    out["trials"] = re.findall(r"(\w+): (\d+)/(\d+) trials completados", tr)
    m = re.search(r"duracion_segundos: (\d+)", tr)
    out["train_seconds"] = int(m.group(1)) if m else None
    pr = read_text(LOGS / "05_predict.log") or ""
    m = re.search(r"MAE ([\d.]+) \| RMSE ([\d.]+) \| R2 ([\d.]+)", pr)
    out["predict"] = m.groups() if m else None
    m = re.search(r"(\d+) predicciones guardadas", pr)
    out["predict_rows"] = int(m.group(1)) if m else None
    dr = read_text(LOGS / "30_drift.log") or ""
    m = re.search(r"Tiempo total: ([\d.]+) s", dr)
    out["drift_seconds"] = m.group(1) if m else None
    return out


def load_git() -> Dict[str, Any]:
    """Commits por integrante y fase (todas las ramas, `git log --all`)."""
    try:
        raw = subprocess.run(["git", "log", "--all", "--format=%h|%an|%ae|%ad|%P|%s", "--date=short"],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    commits = []
    for line in raw.strip().splitlines():
        h, an, ae, ad, parents, subj = line.split("|", 5)
        who = None
        key = f"{an} {ae}".lower()
        for mem in MEMBERS:
            if any(a in key for a in mem["aliases"]):
                who = mem["key"]
        d = date.fromisoformat(ad)
        phase = next(p for p, a, b in PHASES if a <= d <= b)
        commits.append({"hash": h, "author": an, "member": who, "date": ad, "phase": phase,
                        "merge": len(parents.split()) > 1, "subject": subj})
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for c in commits:
        table[c["member"]][c["phase"]][1 if c["merge"] else 0] += 1
    return {"available": True, "commits": commits, "table": table}


def load_eda(cache: Path = FF / "eda_resumen.json") -> Optional[Dict[str, Any]]:
    """Resumen del EDA; se calcula desde data/ solo si no existe el caché."""
    if cache.exists():
        return read_json(cache)
    trips_path = ROOT / "data" / "interim" / "viajes_cleaned.csv"
    matrix_path = ROOT / "data" / "processed" / "mibici_hourly_demand.csv"
    stations_path = ROOT / "data" / "raw" / "nomenclatura_estaciones.csv"
    if not (trips_path.exists() and matrix_path.exists()):
        return None
    import pandas as pd

    trips = pd.read_csv(trips_path, usecols=["Inicio_del_viaje", "Origen_Id", "Genero", "Edad", "duracion_minutos"],
                        parse_dates=["Inicio_del_viaje"])
    ts = trips["Inicio_del_viaje"]
    by_hour = ts.dt.hour.value_counts().sort_index()
    days = ts.dt.normalize()
    per_day = days.value_counts()
    dow = per_day.index.dayofweek
    weekday_avg = float(per_day[dow < 5].mean())
    weekend_avg = float(per_day[dow >= 5].mean())
    top = trips["Origen_Id"].value_counts().head(3)
    names = {}
    if stations_path.exists():
        st = pd.read_csv(stations_path)
        names = dict(zip(st["id"], st["name"]))
    gender = trips["Genero"].fillna("Desconocido").value_counts(normalize=True)
    matrix = pd.read_csv(matrix_path, usecols=["demand_count"])
    result = {
        "source": "data/interim/viajes_cleaned.csv y data/processed/mibici_hourly_demand.csv",
        "trips": int(len(trips)),
        "peak_hours": [[int(h), int(by_hour[h])] for h in by_hour.sort_values(ascending=False).index[:2]],
        "night_trips_1_to_4": int(by_hour.reindex(range(1, 5), fill_value=0).sum()),
        "weekday_avg_per_day": round(weekday_avg, 1),
        "weekend_avg_per_day": round(weekend_avg, 1),
        "weekend_drop_pct": round(100 * (1 - weekend_avg / weekday_avg), 1),
        "top_stations": [[int(s), names.get(s, str(s)), int(n)] for s, n in top.items()],
        "duration_median_min": round(float(trips["duracion_minutos"].median()), 1),
        "duration_mean_min": round(float(trips["duracion_minutos"].mean()), 1),
        "gender_share_pct": {str(k): round(100 * float(v), 1) for k, v in gender.items()},
        "age_median": float(trips["Edad"].median()),
        "age_q1": float(trips["Edad"].quantile(0.25)),
        "age_q3": float(trips["Edad"].quantile(0.75)),
        "matrix_zero_share_pct": round(100 * float((matrix["demand_count"] == 0).mean()), 1),
        "matrix_mean_demand": round(float(matrix["demand_count"].mean()), 2),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_config() -> Dict[str, Any]:
    import yaml

    try:
        return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


def comp_row(D: Dict[str, Any], label: str) -> Optional[List[str]]:
    """Fila de la tabla de datos de comparacion_fase2.md cuyo concepto empieza con `label`."""
    t = (D.get("comp") or {}).get("data")
    for row in (t or {}).get("rows", []):
        if row[0].startswith(label):
            return row
    return None


def load_all() -> Dict[str, Any]:
    return {
        "meta": read_json(ROOT / "models" / "champion_metadata.json"),
        "config": load_config(),
        "models": load_models(),
        "comp": load_comparacion(),
        "pytest": load_pytest(),
        "api": load_api(),
        "repro": load_repro(),
        "docker": load_docker(),
        "dvc": load_dvc(),
        "req": load_requirements(),
        "logs": load_logs(),
        "drift": read_json(REPORTS / "drift" / "drift_report.json"),
        "git": load_git(),
        "eda": load_eda(),
        "ci_tests": read_text(ROOT / ".github" / "workflows" / "tests.yml"),
        "ci_docker": read_text(ROOT / ".github" / "workflows" / "docker-reproducibilidad.yml"),
    }
