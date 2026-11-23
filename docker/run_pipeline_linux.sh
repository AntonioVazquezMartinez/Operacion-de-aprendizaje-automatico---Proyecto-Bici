#!/usr/bin/env bash
# Corre el pipeline completo dentro del contenedor Linux (docker/Dockerfile.pipeline) sin tocar el host:
#   /hostraw  CSV crudos del host montados de SOLO LECTURA (datos_abiertos_AAAA_MM.csv y la nomenclatura).
#   /work     volumen de salida: data/, mlruns.db, mlartifacts/, models/, reports/ y logs/ del reentrenamiento.
# Genera /work/config_linux.yaml (config.yaml del repositorio con todas las rutas apuntando a /work) y ejecuta
# dataset -> features -> train -> predict -> export con esa configuración; al final compara contra Windows.
set -euo pipefail

WORK=${WORK:-/work}
HOSTRAW=${HOSTRAW:-/hostraw}
CONFIG="$WORK/config_linux.yaml"
LOGS="$WORK/logs"
mkdir -p "$WORK/data/raw" "$LOGS"

python - "$CONFIG" "$WORK" <<'EOF'
import sys
from pathlib import Path

import yaml

config_path, work = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
cfg["paths"] = {
    "raw_dir": f"{work}/data/raw", "interim_dir": f"{work}/data/interim", "processed_dir": f"{work}/data/processed",
    "external_dir": f"{work}/data/external", "models_dir": f"{work}/models",
    "figures_dir": f"{work}/reports/figures", "reports_dir": f"{work}/reports",
}
cfg["mlflow"]["tracking_uri"] = f"sqlite:///{work}/mlruns.db"
cfg["mlflow"]["artifact_dir"] = f"{work}/mlartifacts"
cfg["serving"]["model_dir"] = f"{work}/models/champion"
cfg["serving"]["metadata_file"] = f"{work}/models/champion_metadata.json"
cfg["serving"]["reference_predictions_file"] = f"{work}/models/reference_predictions.csv"
config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"Configuración alterna: {config_path}")
EOF

# Solo los archivos fuente que lee src.dataset (meses de data.months y la nomenclatura de data.stations_url).
python - "$CONFIG" "$HOSTRAW" <<'EOF'
import shutil
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
src, dst = Path(sys.argv[2]), Path(cfg["paths"]["raw_dir"])
names = [f"datos_abiertos_{m.replace('-', '_')}.csv" for m in cfg["data"]["months"]]
names.append(Path(cfg["data"]["stations_url"]).name)
for name in names:
    if not (dst / name).exists():
        shutil.copy2(src / name, dst / name)
print(f"{len(names)} archivos fuente copiados de {src} a {dst}")
EOF

step() {
    local name=$1; shift
    local log="$LOGS/$name.log"
    local start
    start=$(date +%s)
    {
        echo "# comando: $*"
        echo "# python: $(python -V 2>&1) | $(uname -srm)"
        echo "# inicio: $(date -Iseconds)"
    } > "$log"
    local rc=0
    "$@" >> "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "# codigo_salida: $rc" >> "$log"
        echo "[$name] FALLÓ (código $rc); ver $log"
        tail -20 "$log"
        exit "$rc"
    fi
    {
        echo "# fin: $(date -Iseconds)"
        echo "# duracion_segundos: $(( $(date +%s) - start ))"
        echo "# codigo_salida: 0"
    } >> "$log"
    echo "[$name] OK en $(( $(date +%s) - start )) s"
}

step 01_dataset python -m src.dataset --config "$CONFIG"
step 02_features python -m src.features --config "$CONFIG"
step 04_train python -m src.modeling.train --config "$CONFIG"
step 05_predict python -m src.modeling.predict --config "$CONFIG"
step 06_export python -m src.modeling.export --config "$CONFIG"
step 07_compare python docker/compare_pipeline.py --work "$WORK" --output "$WORK/comparison"
cat "$LOGS/07_compare.log"
