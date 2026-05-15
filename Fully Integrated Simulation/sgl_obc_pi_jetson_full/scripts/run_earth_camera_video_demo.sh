#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CFG_CUDA="${REPO_ROOT}/config/earth_camera_video_long_local_cuda.json"
CFG_CPU="${REPO_ROOT}/config/earth_camera_video_long_local_cpu.json"
CFG="${CFG_CUDA}"
REFRESH_MS="300"

if [[ ${1:-} == "--cpu" ]]; then
  CFG="${CFG_CPU}"
  shift
fi

if [[ $# -ge 1 ]]; then
  REFRESH_MS="$1"
fi

if [[ ! -f "${CFG}" ]]; then
  echo "ERROR: config not found: ${CFG}" >&2
  exit 1
fi

OUT_DIR=$(python3 - <<PY
import json
with open(r"${CFG}", "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("out_dir", ""))
PY
)

PROFILE=$(python3 - <<PY
import json
with open(r"${CFG}", "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("profile_name", ""))
PY
)

echo "Starting Earth camera video demo"
echo "  config:      ${CFG}"
echo "  profile:     ${PROFILE}"
echo "  out_dir:     ${OUT_DIR}"
echo "  refresh_ms:  ${REFRESH_MS}"

cd "${REPO_ROOT}"
./scripts/run_gui_demo.sh "${CFG}" "${REFRESH_MS}"
