#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CFG_DEFAULT="${REPO_ROOT}/config/local_no_tcp.json"
CFG_PATH="${CFG_DEFAULT}"
REFRESH_MS="200"
START_SIM=1
KEEP_EXISTING=0

usage() {
  cat <<USAGE
Usage:
  scripts/run_gui_demo.sh [config_path] [refresh_ms] [--no-start-sim] [--keep-existing]

Default behavior:
  - safe live-mode startup
  - moves existing live out_dir aside to *_OLD_YYYYMMDD_HHMMSS (unless --keep-existing)
  - starts sim in background (unless --no-start-sim)
  - launches dashboard against fresh live telemetry path
USAGE
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start-sim) START_SIM=0; shift ;;
    --keep-existing) KEEP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  CFG_PATH="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  REFRESH_MS="${POSITIONAL[1]}"
fi

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

if [[ -d "${out_root}" && "${KEEP_EXISTING}" != "1" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  moved="${out_root}_OLD_${ts}"
  echo "Moving stale live output aside:"
  echo "  ${out_root}"
  echo "  -> ${moved}"
  mv "${out_root}" "${moved}"
fi

mkdir -p "${out_root}/mission_store"
: > "${out_root}/mission_store/telemetry_cycles.csv"
: > "${out_root}/mission_store/events.csv"
: > "${out_root}/mission_store/products_manifest.csv"
: > "${out_root}/mission_store/downlink_queue.csv"

sim_pid=""
if [[ "${START_SIM}" == "1" ]]; then
  echo "Starting simulation in background: ./scripts/run_local_demo.sh ${CFG_PATH}"
  (cd "${REPO_ROOT}" && ./scripts/run_local_demo.sh "${CFG_PATH}") &
  sim_pid=$!
  echo "Simulation PID: ${sim_pid}"
else
  echo "Simulation not started (--no-start-sim)."
  echo "Run this in another terminal when ready:"
  echo "  cd ${REPO_ROOT} && ./scripts/run_local_demo.sh ${CFG_PATH}"
fi

echo "Launching dashboard live mode..."
(cd "${REPO_ROOT}" && ./scripts/run_dashboard.sh --live --keep-existing "${CFG_PATH}" "${REFRESH_MS}") || true

if [[ -n "${sim_pid}" ]]; then
  if kill -0 "${sim_pid}" >/dev/null 2>&1; then
    echo "Dashboard closed while sim still running (PID ${sim_pid})."
    echo "Let it finish, or stop with: kill ${sim_pid}"
  else
    echo "Simulation already finished. Packaged outputs are under outputs/<timestamp>_..."
  fi
fi
