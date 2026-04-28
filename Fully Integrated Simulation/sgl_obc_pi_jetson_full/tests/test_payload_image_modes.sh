#!/usr/bin/env bash
set -euo pipefail

PI_BIN=${1:?pi binary required}
SRC_DIR=${2:?repo root required}

run_mode() {
  local mode=$1
  local out_dir=$2
  local cfg="${PWD}/test_config_${mode}.json"

  rm -rf "${PWD}/${out_dir}"
  cat >"${cfg}" <<JSON
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
  "payload_input_mode": "${mode}",
  "payload_fusion_alpha": 0.35,
  "jetson_transport": "local",
  "require_adcs_stable_for_jetson": false,
  "jetson_backend": "cpu",
  "jetson_allow_cpu_fallback": true,
  "connect_timeout_ms": 3000,
  "job_ack_timeout_ms": 2000,
  "job_result_timeout_ms": 12000,
  "host": "127.0.0.1",
  "port": 5500,
  "sim_cycles": 24,
  "dt_s": 1.0,
  "source_image": "${SRC_DIR}/bluemarble.ppm",
  "out_dir": "${out_dir}",
  "jetson_scratch_dir": "${out_dir}/jetson_scratch"
}
JSON

  "${PI_BIN}" --config "${cfg}" >/dev/null 2>&1

  local telem="${PWD}/${out_dir}/mission_store/telemetry_cycles.csv"
  local events="${PWD}/${out_dir}/mission_store/events.csv"
  local manifest="${PWD}/${out_dir}/mission_store/products_manifest.csv"
  [[ -s "${telem}" ]] || { echo "missing telemetry for mode ${mode}" >&2; exit 1; }
  [[ -s "${events}" ]] || { echo "missing events for mode ${mode}" >&2; exit 1; }
  [[ -s "${manifest}" ]] || { echo "missing manifest for mode ${mode}" >&2; exit 1; }

  python3 - "$telem" "$events" "$manifest" "$mode" <<'PY'
import csv, sys
telem_path, events_path, manifest_path, mode = sys.argv[1:]
with open(telem_path, newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"no telemetry rows for {mode}")
required_cols = [
    'camera_mode','camera_frame_ready','alignment_valid','alignment_score',
    'blur_score','brightness_mean','contrast_score','raw_capture_path','rectified_image_path'
]
for c in required_cols:
    if c not in rows[0]:
        raise SystemExit(f"missing telemetry column {c} for {mode}")
if not any(r.get('camera_mode','').strip('"') == mode for r in rows):
    raise SystemExit(f"camera_mode never equals {mode}")
if mode in ('image_file','pi_camera_demo'):
    if not any(r.get('camera_frame_ready') == '1' for r in rows):
        raise SystemExit(f"camera_frame_ready never asserted for {mode}")
with open(events_path, newline='') as f:
    ev = list(csv.DictReader(f))
types = {r.get('event_type','').strip('"') for r in ev}
if mode in ('image_file','pi_camera_demo'):
    needed = {'camera_capture_started','payload_capture_accepted'}
    missing = needed - types
    if missing:
        raise SystemExit(f"missing events for {mode}: {sorted(missing)}")
with open(manifest_path, newline='') as f:
    man = list(csv.DictReader(f))
if len(man) <= 1:
    raise SystemExit(f"manifest has no products for {mode}")
PY
}

run_mode image_file test_out_image_file_mode
run_mode pi_camera_demo test_out_pi_camera_mode
