#!/usr/bin/env bash
set -euo pipefail
PI_BIN="$1"
SRC_DIR="$2"
OUT_DIR="$SRC_DIR/build_wsl/test_out_low_budget"
CFG="$SRC_DIR/build_wsl/test_config_low_budget.json"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cat > "$CFG" <<JSON
{
  "nominal_fraction": 0.75,
  "reserve_margin_W": 80.0,
  "lowres_N": 256,
  "highres_N": 1024,
  "coarse_groups_x": 4,
  "coarse_groups_y": 4,
  "roi_count": 8,
  "progressive_base_N": 128,
  "progressive_max_N": 256,
  "progressive_scale": 2,
  "progressive_max_stages": 2,
  "progressive_roi_growth": 2,
  "tile_px_x": 64,
  "tile_px_y": 64,
  "ring_radius": 0.38,
  "ring_sigma": 0.04,
  "pi_idle_W": 10.0,
  "pi_active_W": 12.0,
  "jetson_idle_W": 5.0,
  "jetson_coarse_W": 1000.0,
  "jetson_refine_W": 1200.0,
  "payload_input_mode": "synthetic_truth",
  "payload_fusion_alpha": 0.35,
  "jetson_transport": "local",
  "require_adcs_stable_for_jetson": false,
  "jetson_backend": "cpu",
  "jetson_allow_cpu_fallback": true,
  "host": "127.0.0.1",
  "port": 5500,
  "sim_cycles": 20,
  "dt_s": 1.0,
  "source_image": "$SRC_DIR/bluemarble.ppm",
  "out_dir": "$OUT_DIR",
  "jetson_scratch_dir": "$OUT_DIR/jetson_scratch"
}
JSON
"$PI_BIN" --config "$CFG" >/dev/null 2>&1
TELEM="$OUT_DIR/mission_store/telemetry_cycles.csv"
MANIFEST="$OUT_DIR/mission_store/products_manifest.csv"
if [[ ! -f "$TELEM" || ! -f "$MANIFEST" ]]; then
  echo "expected telemetry/manifest missing" >&2
  exit 2
fi
if [[ $(wc -l < "$MANIFEST") -ne 1 ]]; then
  echo "unexpected products generated under low budget" >&2
  exit 3
fi
if ! grep -Eq '"THROTTLED"|"SUSPENDED"' "$TELEM"; then
  echo "scheduler did not enter throttled/suspended state" >&2
  exit 4
fi
