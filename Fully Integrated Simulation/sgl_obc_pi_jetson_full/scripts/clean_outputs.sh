#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

for d in out out_local out_tcp out_tcp_validation out_tcp_validation2 out_local_tcp_harden; do
  if [[ -d "${REPO_ROOT}/${d}" ]]; then
    rm -rf "${REPO_ROOT:?}/${d}"
    echo "Removed ${REPO_ROOT}/${d}"
  fi
done

echo "Output cleanup complete."
