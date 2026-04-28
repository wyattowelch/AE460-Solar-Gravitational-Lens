#!/usr/bin/env bash
set -euo pipefail

PI_BIN=${1:?pi binary required}
SRC_DIR=${2:?repo root required}
OUT_DIR="${PWD}/test_out_progressive_outputs"
CFG="${PWD}/test_config_progressive_outputs.json"

rm -rf "${OUT_DIR}"
cat >"${CFG}" <<JSON
{
  "power_cap_W": 25.0,
  "nominal_fraction": 0.75,
  "reserve_margin_W": 20.0,
  "lowres_N": 256,
  "highres_N": 1024,
  "coarse_groups_x": 4,
  "coarse_groups_y": 4,
  "roi_count": 8,
  "progressive_base_N": 128,
  "progressive_max_N": 1024,
  "progressive_scale": 2,
  "progressive_max_stages": 4,
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
  "sim_cycles": 80,
  "dt_s": 1.0,
  "source_image": "${SRC_DIR}/bluemarble.ppm",
  "out_dir": "test_out_progressive_outputs",
  "jetson_scratch_dir": "test_out_progressive_outputs/jetson_scratch"
}
JSON

"${PI_BIN}" --config "${CFG}" >/dev/null 2>&1

MANIFEST="${OUT_DIR}/mission_store/products_manifest.csv"
[[ -s "${MANIFEST}" ]] || { echo "missing manifest" >&2; exit 1; }

python3 - "${MANIFEST}" <<'PY'
import csv
import os
import sys

manifest = sys.argv[1]
rows = list(csv.DictReader(open(manifest, newline="")))
if not rows:
    raise SystemExit("manifest empty")

required = {128, 256, 512}
by_kind = {"coarse": set(), "refined": set()}
for r in rows:
    try:
        n = int(r.get("out_n", "0"))
    except ValueError:
        continue
    k = r.get("kind", "").strip('"')
    if k in by_kind:
        by_kind[k].add(n)
    p = (r.get("path") or "").strip().strip('"')
    if not p:
        raise SystemExit("manifest path empty")
    if not (os.path.exists(p) or os.path.exists(os.path.join(os.getcwd(), p))):
        raise SystemExit(f"manifest path missing: {p}")

for kind in ("coarse", "refined"):
    missing = sorted(required - by_kind[kind])
    if missing:
        raise SystemExit(f"{kind} missing progressive outputs: {missing}")
PY

