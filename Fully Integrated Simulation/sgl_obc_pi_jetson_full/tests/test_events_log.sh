#!/usr/bin/env bash
set -euo pipefail
PI_BIN="$1"
SRC_DIR="$2"
OUT_DIR="$SRC_DIR/build_wsl/test_out_events"
CFG="$SRC_DIR/build_wsl/test_config_events.json"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cat > "$CFG" <<JSON
{
  "nominal_fraction": 0.75,
  "reserve_margin_W": 20.0,
  "lowres_N": 256,
  "highres_N": 512,
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
  "pi_idle_W": 4.0,
  "pi_active_W": 8.0,
  "jetson_idle_W": 5.0,
  "jetson_coarse_W": 10.0,
  "jetson_refine_W": 15.0,
  "payload_input_mode": "synthetic_truth",
  "payload_fusion_alpha": 0.35,
  "jetson_transport": "local",
  "require_adcs_stable_for_jetson": false,
  "jetson_backend": "cpu",
  "jetson_allow_cpu_fallback": true,
  "host": "127.0.0.1",
  "port": 5500,
  "sim_cycles": 35,
  "dt_s": 1.0,
  "source_image": "$SRC_DIR/bluemarble.ppm",
  "out_dir": "$OUT_DIR",
  "jetson_scratch_dir": "$OUT_DIR/jetson_scratch"
}
JSON
"$PI_BIN" --config "$CFG" >/dev/null 2>&1
EVENTS="$OUT_DIR/mission_store/events.csv"
if [[ ! -f "$EVENTS" ]]; then
  echo "events.csv missing" >&2
  exit 2
fi
python3 - "$EVENTS" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print('no events emitted', file=sys.stderr)
    raise SystemExit(3)
have={r.get('event_type','').strip('"') for r in rows}
required=['payload_dataset_ready','jetson_coarse_started','jetson_coarse_completed']
for k in required:
    if k not in have:
        print(f'missing event type: {k}', file=sys.stderr)
        raise SystemExit(4)
if not ({'adcs_correction_started','tracker_degraded','heater_activated','propulsion_burn_started','downlink_active','compute_budget_low','scheduler_mode_changed'} & have):
    print('missing expected autonomous state-transition event', file=sys.stderr)
    raise SystemExit(5)
PY
