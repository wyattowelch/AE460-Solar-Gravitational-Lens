#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

echo "Built integrated OBC/Jetson binaries in: ${BUILD_DIR}"

echo "Building standalone modules"
for mod in \
  "${REPO_ROOT}/../sgl_star_tracker_module/sgl_star_tracker_module" \
  "${REPO_ROOT}/../sgl_eps_module" \
  "${REPO_ROOT}/../sgl_thermal_module" \
  "${REPO_ROOT}/../sgl_comms_module" \
  "${REPO_ROOT}/../sgl_propulsion_module" \
  "${REPO_ROOT}/../sgl_payload_module"; do
  if [[ -f "${mod}/CMakeLists.txt" ]]; then
    mod_name=$(basename "${mod}")
    mod_build="${BUILD_DIR}/${mod_name}_standalone_build"
    cmake -S "${mod}" -B "${mod_build}"
    cmake --build "${mod_build}" -j
    echo "Built module: ${mod_name}"
  else
    echo "Skipping missing module path: ${mod}"
  fi
done

if [[ -x "${BUILD_DIR}/sgl_pi_flight" && -x "${BUILD_DIR}/sgl_jetson_service" ]]; then
  echo "OK: integrated binaries ready"
else
  echo "ERROR: expected binaries missing in ${BUILD_DIR}" >&2
  exit 1
fi
