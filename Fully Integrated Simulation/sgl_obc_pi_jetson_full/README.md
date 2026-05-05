# SGL OBC + Image Processing System (Pi + Jetson)

This codebase implements a local-machine prototype of a distributed spacecraft software architecture for the Solar Gravitational Lens mission.

## Documentation Index

- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md): architecture, authority boundaries, subsystem coupling, and operating modes.
- [MODULE_HANDOFF_GUIDE.md](MODULE_HANDOFF_GUIDE.md): module conventions, standalone workflows, and OBC integration rules for subsystem owners.
- [DEMO_WALKTHROUGH.md](DEMO_WALKTHROUGH.md): step-by-step demo script and cause/effect story.
- [TELEMETRY_REFERENCE.md](TELEMETRY_REFERENCE.md): telemetry/events/manifest/downlink schemas and subsystem field ownership.

## Quick Start

Most users can run this minimal sequence end-to-end:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip python3-venv ffmpeg fswebcam

./scripts/clean_outputs.sh
./scripts/build_all.sh
ctest --test-dir build_wsl --output-on-failure
./scripts/run_local_demo.sh config/image_file_demo.json
./scripts/profile_progressive.sh
```

### A) WSL Ubuntu (first-time setup)

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip python3-venv
```

Optional camera/demo tools:

```bash
sudo apt install -y ffmpeg fswebcam
```

For Raspberry Pi camera bring-up later (on Pi OS / Ubuntu Pi image):

```bash
sudo apt install -y rpicam-apps
```

(`libcamera` tools may be used instead on some OS images.)

### B) Native Linux Ubuntu/Debian (first-time setup)

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip python3-venv
```

Optional camera/demo tools:

```bash
sudo apt install -y ffmpeg fswebcam
```

### Dashboard Python deps (optional unless using GUI)

```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
```

Dashboard is optional and not required for core simulation or tests. Dashboard requires a GUI/display server.

If pip reports an externally-managed environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
```

## Build and Test

```bash
./scripts/clean_outputs.sh
./scripts/build_all.sh
ctest --test-dir build_wsl --output-on-failure
```

## Run Demos

```bash
./scripts/run_local_demo.sh config/local_no_tcp.json
./scripts/run_local_demo.sh config/image_file_demo.json
./scripts/run_local_demo.sh config/pi_camera_demo.json
```

## Run Progressive Profiling

```bash
./scripts/profile_progressive.sh
```

Outputs:
- `out_profile/progressive_profile_summary.csv`
- `out_profile/progressive_stage_timings.csv`
- packaged run bundle: `outputs/<timestamp>_<case_name>/`
- latest packaged bundle symlink: `outputs/latest`

Packaged run layout:
- `run_metadata.json`
- `config/original_config.json`, `config/effective_config.json`
- `csv/telemetry_cycles.csv`, `csv/events.csv`, `csv/downlink_queue.csv`, `csv/products_manifest.csv`, `csv/reconstruction_quality.csv`, `csv/progressive_stage_timings.csv`, plus profile CSVs when present
- `images/datasets/...` (lightweight PNG inspection artifacts)
- `images/products/...` (progressive/base/upscaled/refined PNGs and contact sheet)
- `heavy/raw_ppm`, `heavy/ring_frames`, `heavy/annulus`, `heavy/datasets` (large artifacts)
- `subsystems/<name>/telemetry.csv` and `subsystems/<name>/events.csv` filtered views for `adcs`, `eps`, `thermal`, `propulsion`, `comms`, `payload`, `scheduler`, `jetson`, `obc`

## Run Dashboard

Script helper:

```bash
./scripts/run_dashboard.sh --review outputs/latest 1000
```

Direct command:

```bash
python3 tools/telemetry_dashboard/dashboard.py \
  --run-dir outputs/latest \
  --refresh-ms 1000
```

## Live GUI Demo (Fresh Live Mode)

Default live demo behavior (safe move-aside of stale out_dir + start sim + launch dashboard):

```bash
./scripts/run_gui_demo.sh config/live_systems_demo.json 200
```

Infinite live GUI demo (default ephemeral/no retained outputs):

```bash
./scripts/run_infinite_demo.sh 300
```

Infinite live GUI demo with saved outputs:

```bash
./scripts/run_infinite_demo.sh --save 300
```

Live dashboard only (no sim launch):

```bash
./scripts/run_dashboard.sh --live config/live_systems_demo.json 200
```

Review a completed packaged case explicitly:

```bash
./scripts/run_dashboard.sh --review outputs/latest 1000
```

Adjust simulation runtime by editing `sim_cycles` in your config JSON.
Adjust GUI update speed by changing the trailing refresh argument (milliseconds).
Live dashboard pacing/smoothness (display-side only) is controlled by config keys:
- `live_playback_buffer_enabled` (default `true` in live mode, `false` in review mode)
- `live_playback_cycle_period_ms` (default `180`)
- `live_playback_lag_cycles` (default `5`)
- `live_playback_catchup_multiplier` (default `2.0`)

For infinite mode, use `sim_cycles: -1` in config (or `scripts/run_infinite_demo.sh`, which applies this automatically).

Event markers now default to off to reduce clutter. Enable them from Metrics/Subsystem and choose a marker filter (`Warnings`, `Scheduler/Jetson`, `Payload/Image`, `ADCS/Thermal/Propulsion`, `All`).

## Standalone Tests and Smoke Checks

List tests:

```bash
ctest --test-dir build_wsl -N
```

Run targeted tests:

```bash
ctest --test-dir build_wsl -R test_ring_observation --output-on-failure
ctest --test-dir build_wsl -R test_local_processing --output-on-failure
```

Dashboard smoke checks:

```bash
python3 -m py_compile tools/telemetry_dashboard/core.py tools/telemetry_dashboard/dashboard.py
PYTHONPATH=tools/telemetry_dashboard python3 -m unittest tools/telemetry_dashboard/tests/test_core_smoke.py
```

## Nodes
- **Raspberry Pi OBC (`sgl_pi_flight`)**
  - master command/data handling
  - subsystem simulation interfaces
  - synthetic star-tracker + gyro + fused attitude + controller + reaction wheel ADCS loop
  - persistent storage authority
  - power-aware scheduling and autonomy
  - FDIR hooks
  - Jetson job dispatch
- **Jetson Orin Nano service (`sgl_jetson_service`)**
  - worker-side reconstruction jobs
  - adaptive ROI refinement
  - progressive staged outputs (`128 -> 256 -> 512 -> ...`)
  - image product generation
  - intended CUDA insertion point

## Design rules
- Pi is the system authority.
- Jetson is a controlled accelerator.
- Pi owns persistent mission data.
- Compute is throttled by a live power budget derived from subsystem activity.
- Payload data is generated synthetically from a source image.
- Image processing uses staged reconstruction: a 128 base result, then explicit upscaled/refined pairs at each higher resolution.
- Progressive mode writes `recon_base`, `recon_upscaled`, and `recon_refined` products so demos show what changed at each step.

## Build

Project-standard build command:

```bash
./scripts/build_all.sh
```

`build_all.sh` creates and uses `build_wsl/` as the standard build output directory. Run tests with:

```bash
ctest --test-dir build_wsl --output-on-failure
```

Manual CMake build (advanced/optional):

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

If you build manually, the repository scripts and docs still assume `build_wsl/` unless you adjust commands.

## Config Profiles

- `config/config.json`: existing default behavior (unchanged).
- `config/local_no_tcp.json`: local single-process validation (`jetson_transport=local`, ADCS-stability gate disabled, output at `out_local/`).
- `config/tcp_localhost.json`: two-process localhost TCP validation (`jetson_transport=tcp`, `127.0.0.1:5500`, bounded TCP timeouts, output at `out_tcp/`).
- `config/pi_hardware.json`: Pi/OBC-side deployment profile (Jetson host placeholder `192.168.0.50`, Pi-owned persistent output path `/var/lib/sgl/mission`).
- `config/jetson_hardware.json`: Jetson-side deployment profile (bind `0.0.0.0:5500`, scratch/cache paths under `/var/tmp/sgl/jetson_service`, CUDA backend placeholder enabled).
- `config/image_file_demo.json`: local printed-target/image-file payload ingestion demo (`payload_input_mode=image_file`).
- `config/pi_camera_demo.json`: local Pi-camera demo mode (`payload_input_mode=pi_camera_demo`, with source-image fallback when camera capture is unavailable).
- `config/live_infinite_saved.json`: infinite local run (`sim_cycles=-1`) with persisted outputs under `out_live_infinite_saved/`.

## Scripts

- `scripts/build_all.sh`: builds integrated OBC/Jetson binaries and standalone subsystem modules.
- `scripts/run_local_demo.sh [config_path]`: runs local no-TCP OBC simulation (defaults to `config/local_no_tcp.json`).
- `scripts/run_tcp_jetson.sh [config_path]`: runs Jetson service in TCP mode (defaults to `config/tcp_localhost.json`).
- `scripts/run_tcp_pi.sh [config_path]`: runs Pi flight in TCP mode (defaults to `config/tcp_localhost.json`).
- `scripts/run_dashboard.sh [--review|--live] [outputs/latest|config_path] [refresh_ms]`: launches dashboard in explicit review/live mode (auto-detect if omitted).
- `scripts/run_gui_demo.sh [config_path] [refresh_ms] [--no-start-sim] [--keep-existing]`: fresh live GUI demo launcher (moves stale live out_dir aside by default).
- `scripts/run_infinite_demo.sh [--save] [--keep-running] [--config <cfg>] [refresh_ms]`: infinite live GUI run (`sim_cycles=-1`). Default is ephemeral `/tmp` output cleaned on exit; `--save` persists outputs.
- `scripts/clean_outputs.sh`: removes common output directories (`out*` presets).
- `scripts/package_run_outputs.sh --case-name <name> --source-out-root <out_dir> [--config-path <cfg>]`: packages one run into `outputs/<timestamp>_<case>/` using structured `config/csv/images/heavy/subsystems` layout and writes run metadata.
- `scripts/cleanup_packaged_outputs.sh`: retention/cleanup controller (dry-run by default).

## Output Retention and Disk Safety

Default retention policy (automatic after packaging, unless disabled by config):
- keep last `10` runs as lightweight records
- keep last `3` runs with full heavy artifacts
- runs older than full-retention window are pruned to lightweight data
- runs older than lightweight window are deleted entirely
- any run containing `.keep` or `KEEP_RUN` is preserved

Config controls (per run config JSON):
- `outputs_retention_enabled` (default `true`)
- `outputs_keep_lightweight_runs` (default `10`)
- `outputs_keep_full_runs` (default `3`)
- `outputs_max_total_gb` (default `0`, disabled)
- `outputs_prune_raw_ppm` (default `true`)
- `outputs_prune_ring_frames` (default `true`)
- `outputs_prune_annulus_dumps` (default `true`)
- `outputs_preserve_marked_runs` (default `true`)
- `outputs_retention_include_out_profile` (default `false`)
- `outputs_retention_include_working_outs` (default `false`)
- `min_free_disk_gb_before_run` (default `0`)
- `warn_free_disk_gb` (default `25`)
- `fail_if_disk_below_gb` (default `10` for high-fidelity/profile runs)

Preserve an important packaged run:
```bash
touch outputs/<timestamp_case_name>/.keep
```

Dry-run cleanup (recommended first):
```bash
./scripts/cleanup_packaged_outputs.sh --dry-run --keep-lightweight-runs 10 --keep-full-runs 3 --prune-heavy --prune-raw-ppm --prune-ring-frames --prune-annulus --preserve-marked --include-out-profile
```

Apply cleanup:
```bash
./scripts/cleanup_packaged_outputs.sh --delete --keep-lightweight-runs 10 --keep-full-runs 3 --prune-heavy --prune-raw-ppm --prune-ring-frames --prune-annulus --preserve-marked --include-out-profile
```

## Workflow 1: Local No-TCP Demo

1. Build:
```bash
./scripts/build_all.sh
```
2. Run local simulation:
```bash
./scripts/run_local_demo.sh
```
3. Run dashboard (read-only):
```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
./scripts/run_dashboard.sh outputs/latest 1000
```
4. For a clear demo, enable these plot groups:
- `Power/EPS`: source/load/compute budget/scheduler mode
- `ADCS`: adcs/wheel power, truth vs estimated pointing error, tracker confidence/valid/stars
- `Thermal` + `Propulsion`: heater and burn periods
- `Payload` + `Jetson/Processing`: dataset-ready transitions and coarse/refine processing response
- `COMMS`: queue/backlog behavior

### Printed Target / Camera Demo Input Modes

Payload input modes supported:

- `synthetic_image`: existing synthetic truth-image path (default compatibility path).
- `image_file`: load a real image file (for example a captured/printed target image), run alignment/quality checks, then feed corrected image into payload ring/reconstruction dataset generation.
- `pi_camera_demo`: attempt to capture a frame from local camera (`ffmpeg`/`fswebcam` if available), run alignment/quality checks, then feed corrected image into payload dataset generation.

If camera capture is unavailable in your environment, `pi_camera_demo` falls back to `source_image` and emits warning events so pipeline behavior can still be demonstrated.
If corner markers are weak/missing, payload alignment falls back to content-bounding-box crop (then center crop as last resort), writes `alignment_overlay.ppm`, and continues without crashing.
For ring-like captures, the payload path can select ring-observation extraction and writes `ring_unwrapped_preview.ppm` and `ring_detect_overlay.ppm` diagnostics.

Recommended printed target setup:

- use a high-contrast print of Blue Marble/Mars/Jupiter on matte paper.
- place dark square fiducials near the four corners of the printed target.
- keep target mostly front-facing and evenly lit.
- avoid motion blur and severe over/under exposure.

Run with image file mode:

```bash
./scripts/run_local_demo.sh config/image_file_demo.json
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_image_file/mission_store/telemetry_cycles.csv --events out_image_file/mission_store/events.csv --manifest out_image_file/mission_store/products_manifest.csv --refresh-ms 1000
```

Run with camera demo mode:

```bash
./scripts/run_local_demo.sh config/pi_camera_demo.json
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_camera_demo/mission_store/telemetry_cycles.csv --events out_camera_demo/mission_store/events.csv --manifest out_camera_demo/mission_store/products_manifest.csv --refresh-ms 1000
```

This is a physical demo acquisition path only. It is not the final SGL optical forward model.

Dashboard image preview panel (read-only) can show latest:

- raw capture
- rectified image
- ring preview/dataset input artifacts
- base/upscaled product
- refined product

## Workflow 2: TCP Localhost Demo (Two Processes)

Use this for full communication-path validation and target deployment behavior:

Terminal 1:
```bash
./scripts/run_tcp_jetson.sh
```
Terminal 2:
```bash
./scripts/run_tcp_pi.sh
```

Then launch dashboard against TCP output:

```bash
./scripts/run_dashboard.sh --live config/tcp_localhost.json 200
```

## Workflow 3: Future Pi + Jetson Hardware Deployment

- `build_wsl/` is the repository-standard build output directory. On Pi/Jetson, run `./scripts/build_all.sh` on each node and use that node's local `build_wsl/` binaries.

- Pi node:
```bash
./build_wsl/sgl_pi_flight --config config/pi_hardware.json
```
- Jetson node:
```bash
./build_wsl/sgl_jetson_service --config config/jetson_hardware.json
```

Before deploying:
- update `host` in `config/pi_hardware.json` to your Jetson reachable IP.
- align `port` on both profiles.
- ensure writable paths exist (`/var/lib/sgl/mission`, `/var/tmp/sgl/jetson_service`), or adjust paths.

## Workflow 4: Dashboard-Only Demo

After any sim run that generated telemetry:

```bash
./scripts/run_dashboard.sh outputs/latest 1000
```

Use `config/tcp_localhost.json` or hardware profile paths as needed.

## Tests

```bash
ctest --test-dir build_wsl --output-on-failure
```

`test_tcp_mode` is included and validates two-process TCP Pi↔Jetson behavior. In restricted environments where localhost sockets are blocked, it reports `SKIP` and exits successfully.

Common output roots used by current configs/scripts:
- `out_local/`
- `out_image_file/`
- `out_camera_demo/`
- `out_tcp/`
- `out_profile/`
- `outputs/` (packaged bundles with `outputs/latest` symlink)

## Persistent Pi-Owned Data
- `out/mission_store/products_manifest.csv`: authoritative product manifest written by Pi.
- `out/mission_store/downlink_queue.csv`: Pi-owned downlink queue entries with priority and bit cost.
- `out/mission_store/telemetry_cycles.csv`: cycle-by-cycle autonomous telemetry snapshot.
  - Includes dynamic bus power terms (source/load/reserve/compute budget), subsystem power draws, Jetson mode/job/power, tracker validity/confidence/stars, truth+estimated pointing error, queue depth, and stage/ROI counters.

## Read-only Telemetry Dashboard

The dashboard is under `tools/telemetry_dashboard/` and only reads telemetry (no sim control).

```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
./scripts/run_dashboard.sh outputs/latest 1000
```

## Progressive Profiling Sweep

Run progressive reconstruction timing sweeps:

```bash
./scripts/profile_progressive.sh
```

Profiles:
- `config/profile_progressive_fast.json`: stages `128|256`, observations `8|32`
- `config/profile_progressive_balanced.json`: stages `128|256|512`, observations `8|32|96`
- `config/profile_progressive_full.json`: stages `128|256|512|1024`, observations `8|32|96|192`
- `config/profile_progressive_stress.json`: stages `128|256|512|1024`, observations `16|64|192|384`

Outputs:
- `out_profile/progressive_profile_summary.csv`
- `out_profile/progressive_stage_timings.csv`
- per-profile outputs under `out_profile/<profile_name>/`
- packaged per-run bundles under `outputs/<timestamp>_<case_name>/`
- `outputs/latest` points to the most recent packaged run

CSV interpretation:
- summary CSV: total runtime, ring generation timing, reconstruction timing, ROI-selection timing, manifest/product counts.
- stage CSV: per-stage observation counts, ROI count, base/upscale/refine runtime, total stage runtime, product paths.

Recommended selection:
- `fast_demo`: highest reliability / lowest runtime.
- `balanced_demo`: default live demo setting.
- `full_demo`: longer showcase.
- `stress`: benchmarking only.

`balanced_demo` is the recommended live demo profile.
`full_demo` and `stress` treat late-stage `1024` refine as optional in profile reporting by default; if optional outputs are missing, that indicates scheduler/power/runtime limits, not necessarily a build/runtime crash.

For future Pi+Jetson deployment, keep these profiles and switch to TCP transport variants as needed.

## Notes
- `source_image` should be a binary PPM (P6) image.
- The current image-processing path is CPU-only but designed so CUDA kernels can replace coarse/refine functions later.
- `sgl_obc_pi_jetson_full` builds the star tracker module directly from `../sgl_star_tracker_module/sgl_star_tracker_module`.
- Pi power budgeting now includes dynamic ADCS draw from star-tracker validity, pointing correction torque, and wheel saturation behavior.
- On non-NVIDIA hardware (for example AMD RX 6800), CUDA is not available. Use the current CPU path locally; add HIP/OpenCL backend later if desired.
- Progressive controls are in `config/config.json`:
  - `progressive_base_N`, `progressive_max_N`, `progressive_scale`
  - `progressive_max_stages`, `progressive_roi_growth`
- Payload observation controls in `config/config.json`:
  - `payload_input_mode`: `synthetic_image`, `image_file`, or `pi_camera_demo`
  - `payload_fusion_alpha`: fusion weight for each new capture (higher = faster adaptation)
- Jetson backend controls in `config/config.json`:
  - `jetson_backend`: `cpu` or `cuda`
  - `jetson_allow_cpu_fallback`: if `true`, non-CUDA builds fall back to CPU instead of failing
- TCP robustness controls in `config/config.json`:
  - `connect_timeout_ms`: Pi-to-Jetson connect timeout
  - `job_ack_timeout_ms`: timeout waiting for Jetson job acceptance
  - `job_result_timeout_ms`: timeout waiting for Jetson job completion
- If you want to use a Pale Blue Dot image, set `source_image` in `config/config.json` to that PPM path.
- Demo camera/file modes write payload acquisition debug artifacts and telemetry:
  - raw capture path
  - rectified image path
  - alignment validity/score
  - blur/brightness/contrast metrics
- Current image loader accepts PPM (P6). If your source is JPG/PNG from ESA/USGS, convert to PPM first.
  - Example converted assets in your workspace root: `../bluemarble.ppm`, `../mars.ppm`, `../saturn.ppm`
  - For synthetic ring generation from a truth image, keep `payload_input_mode="synthetic_image"`.
  - For printed/camera demo ingestion, use `payload_input_mode="image_file"` or `payload_input_mode="pi_camera_demo"`.

## Standalone Subsystem Modules

The workspace now includes reusable standalone subsystem modules with a common pattern (`include/`, `src/`, `examples/standalone_demo.cpp`, `tests/`, `CMakeLists.txt`, `README.md`):

- `../sgl_eps_module`
- `../sgl_thermal_module`
- `../sgl_comms_module`
- `../sgl_propulsion_module`
- `../sgl_payload_module`

Each module can be built/tested independently and handed off to subsystem owners, while `sgl_obc_pi_jetson_full` remains the mission-integrated authority.

### EPS Integration Note

`sgl_pi_flight` now uses `../sgl_eps_module` as the source of:
- source power generation
- noncompute/reserve-aware compute budget
- total bus load and bus margin/low-power assessment

Mission authority remains in OBC: scheduler mode selection, Jetson job gating, and FDIR transitions are still handled in `pi_flight/main.cpp`.

### Thermal Integration Note

`sgl_pi_flight` now uses `../sgl_thermal_module` for thermal/heater dynamics and thermal telemetry fields.
OBC still owns mission authority decisions (scheduler/FDIR); thermal is a reusable state/power provider.

Telemetry now includes thermal detail fields:
- `thermal_mode`
- `heater_active`
- `thermal_temp_c`

### Propulsion Integration Note

`sgl_pi_flight` now uses `../sgl_propulsion_module` for propulsion activity/power/thrust state.
OBC remains mission authority for scheduling and FDIR decisions.

Telemetry now includes propulsion detail fields:
- `propulsion_mode`
- `propulsion_active`
- `propulsion_thrust_n`

### Payload Integration Note

`sgl_pi_flight` now uses `../sgl_payload_module` as the reusable source of payload acquisition state/power and dataset-ready signaling.
OBC remains mission authority and still decides if/when Jetson jobs run from available datasets.

Telemetry now includes payload detail fields:
- `payload_mode`
- `payload_active`
- `dataset_ready`
- `dataset_id`
- `dataset_count`
- `acquisition_stage`

### ADCS / Star Tracker Boundary Note

`sgl_pi_flight` now consumes the reusable `AdcsSystem` facade from `../sgl_star_tracker_module/sgl_star_tracker_module` for closed-loop ADCS simulation.
The module owns sensor/filter/controller/wheel/truth-loop behavior; OBC owns mission-context authority:
- ADCS stability gating for Jetson jobs
- scheduler/FDIR decisions
- mission telemetry aggregation and storage

## TCP Troubleshooting

- Port already in use:
  - pick a different `port` value in config.
  - stop any process already bound to that port.
- Connection refused:
  - start `sgl_jetson_service` first, then `sgl_pi_flight`.
  - confirm both processes use the same `host` and `port`.
- WSL or sandbox socket restrictions:
  - if localhost bind/connect is blocked, use local transport:
  - `"jetson_transport": "local"`
  - this preserves scheduler/FDIR and product generation behavior while bypassing TCP.

## General Troubleshooting

- Script permission denied:
  - `chmod +x scripts/*.sh tests/*.sh`

- Dashboard import failure (for example `ModuleNotFoundError: pyqtgraph`):
  - install dashboard requirements:
  - `python3 -m pip install -r tools/telemetry_dashboard/requirements.txt`
  - `scripts/run_dashboard.sh` now checks dependencies first and prints this command instead of a traceback.

- Pip externally-managed environment:
  - use a virtual environment:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python3 -m pip install -r tools/telemetry_dashboard/requirements.txt`

- Camera not found:
  - install optional capture tools: `sudo apt install -y ffmpeg fswebcam`
  - Pi bring-up later: install `rpicam-apps` or libcamera tools on target OS.
  - `pi_camera_demo` falls back to `source_image` and logs warning events.

- No GUI in WSL:
  - dashboard requires a display server/X forwarding.
  - use non-GUI checks when headless:
  - `python3 -m py_compile tools/telemetry_dashboard/core.py tools/telemetry_dashboard/dashboard.py`
  - `PYTHONPATH=tools/telemetry_dashboard python3 -m unittest tools/telemetry_dashboard/tests/test_core_smoke.py`

- Optional 1024 refined output missing in `full_demo`/`stress` profiling:
  - this usually indicates scheduler/power/runtime limits, not a build crash.
  - review `out_profile/progressive_profile_summary.csv` fields:
  - `missing_optional_outputs`, `scheduler_throttle_count`, `scheduler_suspend_count`

- Packaged outputs are large:
  - inspect the latest run under `outputs/latest`
  - preview cleanup with `./scripts/cleanup_packaged_outputs.sh --dry-run --keep-lightweight-runs 10 --keep-full-runs 3 --prune-heavy --prune-raw-ppm --prune-ring-frames --prune-annulus --include-out-profile`
  - apply cleanup only when intended with `./scripts/cleanup_packaged_outputs.sh --delete --keep-lightweight-runs 10 --keep-full-runs 3 --prune-heavy --prune-raw-ppm --prune-ring-frames --prune-annulus --include-out-profile`

- `ctest` build directory missing:
  - run `./scripts/build_all.sh` first, then rerun `ctest --test-dir build_wsl --output-on-failure`.

- Test portability:
  - runtime tests do not require `ripgrep` (`rg`) anymore.
  - standard POSIX tools are used in test scripts for product/telemetry checks.
