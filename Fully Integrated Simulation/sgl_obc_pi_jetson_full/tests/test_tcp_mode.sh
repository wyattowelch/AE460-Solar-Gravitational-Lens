#!/usr/bin/env bash
set -euo pipefail

PI_BIN=${1:?pi binary required}
JETSON_BIN=${2:?jetson binary required}
REPO_ROOT=${3:?repo root required}

WORK_DIR="${PWD}/test_out_tcp_mode"
CFG="${PWD}/test_config_tcp_mode.json"
PORT=$((16500 + ($$ % 1000)))

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

cat > "${CFG}" <<JSON
{
  "power_cap_W": 25.0,
  "nominal_fraction": 0.75,
  "reserve_margin_W": 20.0,
  "lowres_N": 256,
  "highres_N": 512,
  "coarse_groups_x": 4,
  "coarse_groups_y": 4,
  "roi_count": 6,
  "progressive_base_N": 128,
  "progressive_max_N": 512,
  "progressive_scale": 2,
  "progressive_max_stages": 3,
  "progressive_roi_growth": 2,
  "tile_px_x": 64,
  "tile_px_y": 64,
  "ring_radius": 0.38,
  "ring_sigma": 0.04,
  "pi_idle_W": 4.0,
  "pi_active_W": 8.0,
  "jetson_idle_W": 5.0,
  "jetson_coarse_W": 10.0,
  "jetson_refine_W": 15.0,
  "payload_input_mode": "synthetic_truth",
  "payload_fusion_alpha": 0.35,
  "jetson_transport": "tcp",
  "require_adcs_stable_for_jetson": false,
  "jetson_backend": "cpu",
  "jetson_allow_cpu_fallback": true,
  "connect_timeout_ms": 1500,
  "job_ack_timeout_ms": 1500,
  "job_result_timeout_ms": 12000,
  "host": "127.0.0.1",
  "port": ${PORT},
  "sim_cycles": 25,
  "dt_s": 1.0,
  "source_image": "${REPO_ROOT}/bluemarble.ppm",
  "out_dir": "${WORK_DIR}",
  "jetson_scratch_dir": "${WORK_DIR}/jetson_scratch"
}
JSON

JETSON_LOG="${WORK_DIR}/jetson_stdout.log"
PI_LOG="${WORK_DIR}/pi_stdout.log"
JETSON_FILE_LOG="${WORK_DIR}/logs/sgl_jetson_service.log"

"${JETSON_BIN}" --config "${CFG}" >"${JETSON_LOG}" 2>&1 &
JETSON_PID=$!

cleanup() {
  if kill -0 "${JETSON_PID}" >/dev/null 2>&1; then
    kill "${JETSON_PID}" >/dev/null 2>&1 || true
    wait "${JETSON_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep 0.5
if ! kill -0 "${JETSON_PID}" >/dev/null 2>&1; then
  if grep -qi "bind failed" "${JETSON_LOG}" 2>/dev/null || grep -qi "bind failed" "${JETSON_FILE_LOG}" 2>/dev/null; then
    echo "SKIP: socket bind unavailable in this environment"
    exit 0
  fi
  wait "${JETSON_PID}" || true
  echo "Jetson exited before test start"
  exit 1
fi

set +e
timeout 60 "${PI_BIN}" --config "${CFG}" >"${PI_LOG}" 2>&1
PI_RC=$?
set -e

if [[ ${PI_RC} -ne 0 ]]; then
  if grep -qi "connect failed" "${PI_LOG}" 2>/dev/null; then
    echo "SKIP: socket connect unavailable in this environment"
    exit 0
  fi
  echo "Pi failed in TCP smoke test"
  tail -n 80 "${PI_LOG}" || true
  exit "${PI_RC}"
fi

for _ in $(seq 1 20); do
  if ! kill -0 "${JETSON_PID}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done
if kill -0 "${JETSON_PID}" >/dev/null 2>&1; then
  echo "Jetson did not exit after shutdown request, terminating"
  kill "${JETSON_PID}" >/dev/null 2>&1 || true
  wait "${JETSON_PID}" >/dev/null 2>&1 || true
fi

TELEM="${WORK_DIR}/mission_store/telemetry_cycles.csv"
EVENTS="${WORK_DIR}/mission_store/events.csv"
MANIFEST="${WORK_DIR}/mission_store/products_manifest.csv"
DOWNLINK="${WORK_DIR}/mission_store/downlink_queue.csv"

[[ -s "${TELEM}" ]] || { echo "missing telemetry CSV"; exit 1; }
[[ -s "${EVENTS}" ]] || { echo "missing events CSV"; exit 1; }
[[ -s "${MANIFEST}" ]] || { echo "missing manifest CSV"; exit 1; }
[[ -s "${DOWNLINK}" ]] || { echo "missing downlink queue CSV"; exit 1; }

grep -q "jetson_coarse_started" "${EVENTS}" || { echo "missing jetson coarse start event"; exit 1; }
grep -q "jetson_coarse_completed" "${EVENTS}" || { echo "missing jetson coarse complete event"; exit 1; }
grep -q "jetson_refine_completed" "${EVENTS}" || { echo "missing jetson refine complete event"; exit 1; }
grep -q ",\"coarse\"," "${MANIFEST}" || { echo "missing coarse manifest entries"; exit 1; }
grep -q ",\"refined\"," "${MANIFEST}" || { echo "missing refined manifest entries"; exit 1; }
