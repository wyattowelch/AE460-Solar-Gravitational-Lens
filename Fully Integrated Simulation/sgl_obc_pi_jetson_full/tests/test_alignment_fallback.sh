#!/usr/bin/env bash
set -euo pipefail

PI_BIN=${1:?pi binary required}
OUT_DIR="${PWD}/test_out_alignment_fallback"
CFG="${PWD}/test_config_alignment_fallback.json"
SRC_IMG="${PWD}/test_alignment_fallback_input.ppm"

rm -rf "${OUT_DIR}"

python3 - "${SRC_IMG}" <<'PY'
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
w, h = 960, 720
cx, cy = 600, 430
rx, ry = 220, 170
with out.open("wb") as f:
    f.write(f"P6\n{w} {h}\n255\n".encode())
    for y in range(h):
        row = bytearray()
        for x in range(w):
            # bright paper-like background with mild gradient
            bg = int(225 - 18 * (x / w) - 14 * (y / h))
            val = bg
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            if dx * dx + dy * dy <= 1.0:
                # off-center colorful target
                r = int(80 + 140 * max(0.0, 1.0 - abs(dx)))
                g = int(70 + 150 * max(0.0, 1.0 - abs(dy)))
                b = int(90 + 130 * max(0.0, 1.0 - abs(dx + dy) * 0.5))
                row.extend(bytes((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))))
            else:
                row.extend(bytes((val, val, val)))
        f.write(row)
PY

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
  "sim_cycles": 30,
  "dt_s": 1.0,
  "source_image": "${SRC_IMG}",
  "out_dir": "test_out_alignment_fallback",
  "jetson_scratch_dir": "test_out_alignment_fallback/jetson_scratch"
}
JSON

"${PI_BIN}" --config "${CFG}" >/dev/null 2>&1

EVENTS="${OUT_DIR}/mission_store/events.csv"
TELEM="${OUT_DIR}/mission_store/telemetry_cycles.csv"
[[ -s "${EVENTS}" ]] || { echo "missing events" >&2; exit 1; }
[[ -s "${TELEM}" ]] || { echo "missing telemetry" >&2; exit 1; }

python3 - "${OUT_DIR}" "${EVENTS}" "${TELEM}" <<'PY'
import csv
import os
import sys

out_dir, events_csv, telem_csv = sys.argv[1:]
events = list(csv.DictReader(open(events_csv, newline="")))
types = {e.get("event_type", "").strip('"') for e in events}
if "payload_alignment_failed" not in types:
    raise SystemExit("expected payload_alignment_failed event for fallback test")

rows = list(csv.DictReader(open(telem_csv, newline="")))
paths = [r.get("rectified_image_path", "").strip('"') for r in rows if r.get("rectified_image_path")]
if not paths:
    raise SystemExit("no rectified_image_path logged")
for p in paths:
    if os.path.exists(p) or os.path.exists(os.path.join(os.getcwd(), p)) or os.path.exists(os.path.join(out_dir, p)):
        print("ok")
        break
else:
    raise SystemExit("rectified image path(s) do not exist")
PY

