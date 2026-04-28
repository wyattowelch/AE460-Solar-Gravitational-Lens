#!/usr/bin/env bash
set -euo pipefail

PI_BIN=${1:?pi binary required}
SRC_DIR=${2:?repo root required}
OUT_DIR="test_out_manifest_consistency"
CFG="${PWD}/test_config_manifest_consistency.json"

rm -rf "${PWD}/${OUT_DIR}"
cat >"${CFG}" <<JSON
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
  "payload_input_mode": "image_file",
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
  "out_dir": "${OUT_DIR}",
  "jetson_scratch_dir": "${OUT_DIR}/jetson_scratch"
}
JSON

"${PI_BIN}" --config "${CFG}" >/dev/null 2>&1

python3 - "${PWD}" "${OUT_DIR}" <<'PY'
import csv
import os
import sys

repo_root, out_dir = sys.argv[1:]
store_dir = os.path.join(repo_root, out_dir, "mission_store")
manifest = os.path.join(store_dir, "products_manifest.csv")
telemetry = os.path.join(store_dir, "telemetry_cycles.csv")

if not os.path.isfile(manifest):
    raise SystemExit(f"missing manifest: {manifest}")
if not os.path.isfile(telemetry):
    raise SystemExit(f"missing telemetry: {telemetry}")

def resolve_path(raw: str):
    raw = (raw or "").strip().strip('"')
    if not raw:
        return None
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    else:
        candidates.append(os.path.join(repo_root, raw))
        candidates.append(os.path.join(repo_root, out_dir, raw))
        candidates.append(raw)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

with open(manifest, newline="") as f:
    mrows = list(csv.DictReader(f))
if not mrows:
    raise SystemExit("manifest has no rows")

missing_manifest = []
for idx, row in enumerate(mrows, start=2):
    p = row.get("path", "")
    if not p:
        missing_manifest.append((idx, "<empty>"))
        continue
    if resolve_path(p) is None:
        missing_manifest.append((idx, p))

if missing_manifest:
    preview = ", ".join(f"line {ln}: {p}" for ln, p in missing_manifest[:5])
    raise SystemExit(f"manifest path(s) missing ({len(missing_manifest)}): {preview}")

with open(telemetry, newline="") as f:
    trows = list(csv.DictReader(f))
if not trows:
    raise SystemExit("telemetry has no rows")

path_cols = ["raw_capture_path", "rectified_image_path"]
for c in path_cols:
    if c not in trows[0]:
        raise SystemExit(f"missing telemetry column: {c}")

missing_telem = []
for idx, row in enumerate(trows, start=2):
    for c in path_cols:
        v = row.get(c, "")
        if not v:
            continue
        if resolve_path(v) is None:
            missing_telem.append((idx, c, v))

if missing_telem:
    preview = ", ".join(f"line {ln} {col}: {p}" for ln, col, p in missing_telem[:5])
    raise SystemExit(f"telemetry path(s) missing ({len(missing_telem)}): {preview}")

print("manifest/telemetry path consistency checks passed")
PY

