#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ARG1="${1:-outputs/latest}"
REFRESH_MS="${2:-200}"
RUN_ARG="${ARG1}"
telemetry=""

if [[ -f "${ARG1}" ]]; then
  out_dir=$(python3 - <<PY
import json
with open(r"${ARG1}", "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("out_dir", "out"))
PY
)
  if [[ "${out_dir}" = /* ]]; then
    out_root="${out_dir}"
  else
    out_root="${REPO_ROOT}/${out_dir}"
  fi
  telemetry="${out_root}/mission_store/telemetry_cycles.csv"
  RUN_ARG="${telemetry}"
fi

# Fail fast with a clear hint instead of showing a Python traceback.
if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
need = ["pyqtgraph", "PyQt5"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  echo "ERROR: Missing dashboard dependencies. Run: python3 -m pip install -r tools/telemetry_dashboard/requirements.txt" >&2
  exit 2
fi

pushd "${REPO_ROOT}" >/dev/null
exec python3 tools/telemetry_dashboard/dashboard.py "${RUN_ARG}" --refresh-ms "${REFRESH_MS}"
