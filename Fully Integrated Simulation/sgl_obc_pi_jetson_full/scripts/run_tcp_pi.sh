#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
CFG_DEFAULT="${REPO_ROOT}/config/tcp_localhost.json"
CFG_PATH="${1:-${CFG_DEFAULT}}"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config not found: ${CFG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/sgl_pi_flight" ]]; then
  echo "ERROR: binary missing: ${BUILD_DIR}/sgl_pi_flight" >&2
  echo "Run scripts/build_all.sh first." >&2
  exit 1
fi

pushd "${REPO_ROOT}" >/dev/null
exec "${BUILD_DIR}/sgl_pi_flight" --config "${CFG_PATH}"
