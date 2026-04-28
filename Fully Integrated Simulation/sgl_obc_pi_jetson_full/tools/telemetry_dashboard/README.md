# Telemetry Dashboard (Read-only)

This dashboard only reads/tails Pi telemetry CSV output and does not control the simulation.

## 1. Run Local Simulation

From `sgl_obc_pi_jetson_full`:

```bash
./scripts/run_local_demo.sh config/local_no_tcp.json
```

## 2. Run Dashboard Pointing to Telemetry CSV

```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_local/mission_store/telemetry_cycles.csv --events out_local/mission_store/events.csv --manifest out_local/mission_store/products_manifest.csv --refresh-ms 200
```

## 3. Groups Shown

- `Power/EPS`: source/bus/load/budget/scheduler.
- `ADCS`: ADCS and wheel power, truth/estimated pointing errors, tracker confidence/valid/stars.
- `Thermal`: thermal power/mode/heater/temp.
- `Propulsion`: propulsion power/mode/active/thrust.
- `Payload`: payload power/mode/activity/dataset lifecycle fields.
- `COMMS`: comms power/mode/backlog queue.
- `Jetson/Processing`: Jetson power/mode/job and processing queue/ROI counters.

Mode/string fields are shown in a live status table. Numeric metrics are plotted and toggleable via checkboxes.
Autonomous events are shown in a live event table and as optional vertical markers on plots.

## 4. Image Preview Panel

The dashboard includes a read-only image preview panel that auto-refreshes during runtime.

Selectable preview sources:

- `Raw Capture`: camera/image-file frame used for payload ingest
- `Rectified`: aligned/corrected target image
- `Ring Preview`: payload dataset/ring preview artifact when present
- `Coarse`: latest coarse reconstruction product (from manifest)
- `Refined`: latest refined product (from manifest)

The panel discovers paths from telemetry (`raw_capture_path`, `rectified_image_path`, `dataset_id`) and `products_manifest.csv`.
Missing files are handled gracefully.

## 5. Demo Modes with Physical Input Path

Image file demo:

```bash
./scripts/run_local_demo.sh config/image_file_demo.json
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_image_file/mission_store/telemetry_cycles.csv --events out_image_file/mission_store/events.csv --manifest out_image_file/mission_store/products_manifest.csv --refresh-ms 200
```

Pi camera demo (falls back to source image if camera is unavailable):

```bash
./scripts/run_local_demo.sh config/pi_camera_demo.json
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_camera_demo/mission_store/telemetry_cycles.csv --events out_camera_demo/mission_store/events.csv --manifest out_camera_demo/mission_store/products_manifest.csv --refresh-ms 200
```

## 6. Read-only Scope

This is a demo visualization only. It does not send commands to the simulation.

## Smoke Test (Headless)

No display server required:

```bash
python3 -m py_compile tools/telemetry_dashboard/dashboard.py tools/telemetry_dashboard/core.py
PYTHONPATH=tools/telemetry_dashboard python3 -m unittest tools/telemetry_dashboard/tests/test_core_smoke.py
```

## Notes

- Update cadence defaults to 200 ms (recommended range: 100-250 ms).
- The dashboard auto-detects available CSV columns and tolerates appended/new fields.
- It does not depend on strict CSV column ordering.
