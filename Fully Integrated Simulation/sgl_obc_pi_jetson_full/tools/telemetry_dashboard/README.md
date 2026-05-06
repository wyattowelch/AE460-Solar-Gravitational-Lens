# Telemetry Dashboard (Read-only)

This dashboard is read-only and never sends spacecraft control commands.

## Launch Modes

Packaged run review (recommended):

```bash
./scripts/run_dashboard.sh --review outputs/latest 1000
# or
python3 tools/telemetry_dashboard/dashboard.py --run-dir outputs/latest --refresh-ms 1000
```

Live working-output mode:

```bash
./scripts/run_dashboard.sh --live config/image_file_demo.json 200
# or
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_image_file/mission_store/telemetry_cycles.csv --refresh-ms 200
```

Live mode defaults to buffered playback for smoother visuals:
- `live_playback_buffer_enabled=true`
- `live_playback_cycle_period_ms=180`
- `live_playback_lag_cycles=5`
- `live_playback_catchup_multiplier=2.0`

These are display-only pacing controls (raw CSV values and simulation timing are unchanged).
When launched via `scripts/run_dashboard.sh --live <config> ...`, the dashboard reads these knobs from that config.

Fresh live demo mode (recommended for demos):

```bash
./scripts/run_gui_demo.sh config/live_systems_demo.json 200
```

## Tabs

- `Overview`: run name/config summary, completion status (`complete/partial/failed/running`), cycle, scheduler mode, compute budget, bus/source power, Jetson/payload state, downlink backlog, latest event.
- `Metrics/Subsystem`: grouped multi-panel plots (independent y-axes) with views such as `Demo Default`, `Power Summary`, single-group views, and `All Groups: Stacked`.
- `Events`: filter/search by subsystem token and warning/error severity.
- `Image Pipeline`: raw/rectified/preconditioned/ring previews and progressive outputs (`128 base`, `256/512/1024/2048 upscaled+refined` when present). Missing stages show `MISSING / NOT COMPLETED`.
- `Quality/Profile`: reconstruction quality (NMAE/MSE), per-stage timing, observations, ROI counts.
- `Downlink`: `products_manifest` and `downlink_queue` views plus counts.

## Demo Story Clarification

For `image_file` / `pi_camera_demo` modes, camera/image capture is a demo stand-in. The captured/rectified/preconditioned source image is used as ideal truth to generate synthetic SGL observations. In mission reality, the spacecraft would measure Einstein-ring data directly.

Source file intake supports `PPM`, `PNG`, and `JPG/JPEG` directly. In `pi_camera_demo`, camera capture is written as JPG when possible and decoded directly into the pipeline (no external runtime conversion helper required).

Backend resolution metadata is available in manifest status text:
- `backend_requested`
- `backend_resolved`
- `cuda_build_enabled`
- `cuda_runtime_available`
- `fallback_used`
- `backend_reason`

## Scenario Guidance (startup configs, not dashboard controls)

- `nominal_demo`: `config/local_no_tcp.json`
- `full_2048_reconstruction_demo`: `config/profile_progressive_2048_force_complete_unthrottled.json`
- `low_power_throttle_demo`: lower `power_cap_W` or raise subsystem loads in a copied config
- `propulsion_burn_demo`: increase propulsion activity config parameters in a copied config
- `thermal_heater_demo`: use colder thermal start/setpoint in a copied config
- `comms_backlog_demo`: increase payload cadence and/or reduce comms downlink throughput in a copied config

## Dependencies

```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
```

If dependencies are missing, `scripts/run_dashboard.sh` prints:

`Missing dashboard dependencies. Run: python3 -m pip install -r tools/telemetry_dashboard/requirements.txt`

## WSL Note

Dashboard requires a GUI/display server (X/Wayland forwarding). Headless test mode still works without display.

## Marker Behavior

- Event markers default to off to avoid clutter.
- When enabled, use marker filter modes (`Warnings`, `Scheduler/Jetson`, `Payload/Image`, `ADCS/Thermal/Propulsion`, `All`).
- `Markers On Top Panel Only` is enabled by default for readability.

## Smoke Test (Headless)

```bash
python3 -m py_compile tools/telemetry_dashboard/dashboard.py tools/telemetry_dashboard/core.py
PYTHONPATH=tools/telemetry_dashboard python3 -m unittest tools/telemetry_dashboard/tests/test_core_smoke.py
```


Stop stale dashboard processes:
```bash
./scripts/stop_dashboard.sh
```

Check demo/dashboard processes:
```bash
./scripts/check_running_demo_processes.sh
```
