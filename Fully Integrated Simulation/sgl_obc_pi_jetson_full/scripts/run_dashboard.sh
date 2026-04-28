#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CFG_DEFAULT="${REPO_ROOT}/config/local_no_tcp.json"
CFG_PATH="${1:-${CFG_DEFAULT}}"
REFRESH_MS="${2:-200}"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config not found: ${CFG_PATH}" >&2
  exit 1
fi

out_dir=$(python3 - <<PY
import json
with open(r"${CFG_PATH}", "r", encoding="utf-8") as f:
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
events="${out_root}/mission_store/events.csv"

if [[ ! -f "${telemetry}" ]]; then
  echo "ERROR: telemetry not found: ${telemetry}" >&2
  echo "Run scripts/run_local_demo.sh or scripts/run_tcp_pi.sh first." >&2
  exit 1
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
exec python3 tools/telemetry_dashboard/dashboard.py --telemetry "${telemetry}" --events "${events}" --refresh-ms "${REFRESH_MS}"
