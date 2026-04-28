#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
CFG_DEFAULT="${REPO_ROOT}/config/local_no_tcp.json"
CFG_PATH="${1:-${CFG_DEFAULT}}"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config not found: ${CFG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/sgl_pi_flight" ]]; then
  echo "Binary missing: ${BUILD_DIR}/sgl_pi_flight"
  echo "Building..."
  "${SCRIPT_DIR}/build_all.sh"
fi

pushd "${REPO_ROOT}" >/dev/null
"${BUILD_DIR}/sgl_pi_flight" --config "${CFG_PATH}"
popd >/dev/null

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

echo "Local demo complete. Outputs:"
echo "  telemetry: ${out_root}/mission_store/telemetry_cycles.csv"
echo "  events:    ${out_root}/mission_store/events.csv"
echo "  manifest:  ${out_root}/mission_store/products_manifest.csv"
echo "  downlink:  ${out_root}/mission_store/downlink_queue.csv"
