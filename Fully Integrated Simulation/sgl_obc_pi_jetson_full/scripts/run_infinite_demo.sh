#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
BASE_CFG_DEFAULT="${REPO_ROOT}/config/live_systems_demo.json"

BASE_CFG="${BASE_CFG_DEFAULT}"
REFRESH_MS="300"
SAVE_OUTPUTS=0
KEEP_RUNNING=0

usage() {
  cat <<USAGE
Usage:
  scripts/run_infinite_demo.sh [--save] [--keep-running] [--config <cfg>] [refresh_ms]

Behavior:
  - Default: infinite live GUI demo with ephemeral output under /tmp (cleaned up on exit).
  - --save: infinite live GUI demo with persisted output under out_live_infinite_saved/.
  - --keep-running: leave simulation running after dashboard closes.
USAGE
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save)
      SAVE_OUTPUTS=1
      shift
      ;;
    --keep-running)
      KEEP_RUNNING=1
      shift
      ;;
    --config)
      BASE_CFG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  REFRESH_MS="${POSITIONAL[0]}"
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "ERROR: base config not found: ${BASE_CFG}" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/sgl_pi_flight" ]]; then
  echo "Binary missing: ${BUILD_DIR}/sgl_pi_flight"
  echo "Building..."
  "${SCRIPT_DIR}/build_all.sh"
fi

ts=$(date +%Y%m%d_%H%M%S)
if [[ "${SAVE_OUTPUTS}" == "1" ]]; then
  out_root="${REPO_ROOT}/out_live_infinite_saved"
else
  out_root="/tmp/sgl_live_infinite_${ts}"
fi

tmp_cfg=$(mktemp "/tmp/sgl_infinite_cfg_${ts}_XXXXXX.json")
python3 - "${BASE_CFG}" "${tmp_cfg}" "${out_root}" "${SAVE_OUTPUTS}" <<'PY'
import json, os, sys
base_cfg, out_cfg, out_root, save_flag = sys.argv[1:5]
save_outputs = save_flag == "1"
cfg = json.load(open(base_cfg, "r", encoding="utf-8"))
cfg["sim_cycles"] = -1
cfg["out_dir"] = out_root
cfg["jetson_scratch_dir"] = os.path.join(out_root, "jetson_scratch")
cfg["profile_name"] = "live_infinite_saved" if save_outputs else "live_infinite_ephemeral"
if not save_outputs:
    cfg["outputs_retention_enabled"] = False
json.dump(cfg, open(out_cfg, "w", encoding="utf-8"), indent=2)
PY

mkdir -p "${out_root}/mission_store"

echo "Starting infinite simulation:"
echo "  config: ${tmp_cfg}"
echo "  out_dir: ${out_root}"
if [[ "${SAVE_OUTPUTS}" == "0" ]]; then
  echo "  mode: ephemeral (will be removed on exit unless --keep-running)"
else
  echo "  mode: saved outputs"
fi

sim_pid=""
cleanup() {
  if [[ -n "${sim_pid}" ]]; then
    if kill -0 "${sim_pid}" >/dev/null 2>&1; then
      if [[ "${KEEP_RUNNING}" == "1" ]]; then
        echo "Simulation still running (PID ${sim_pid})"
      else
        echo "Stopping simulation PID ${sim_pid}..."
        kill "${sim_pid}" >/dev/null 2>&1 || true
        wait "${sim_pid}" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "${tmp_cfg}" || true
  if [[ "${SAVE_OUTPUTS}" == "0" && "${KEEP_RUNNING}" == "0" ]]; then
    rm -rf "${out_root}" || true
  fi
}
trap cleanup EXIT

(cd "${REPO_ROOT}" && "${BUILD_DIR}/sgl_pi_flight" --config "${tmp_cfg}") &
sim_pid=$!
echo "Simulation PID: ${sim_pid}"

echo "Launching dashboard..."
(cd "${REPO_ROOT}" && ./scripts/run_dashboard.sh --live --keep-existing "${tmp_cfg}" "${REFRESH_MS}") || true

if [[ "${KEEP_RUNNING}" == "1" ]]; then
  echo "Dashboard closed. Simulation continues in background."
  echo "Stop it with: kill ${sim_pid}"
fi

