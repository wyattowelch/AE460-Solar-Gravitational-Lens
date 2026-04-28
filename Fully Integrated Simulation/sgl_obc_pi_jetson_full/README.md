# SGL OBC + Image Processing System (Pi + Jetson)

This codebase implements a local-machine prototype of a distributed spacecraft software architecture for the Solar Gravitational Lens mission.

## Documentation Index

- [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md): architecture, authority boundaries, subsystem coupling, and operating modes.
- [MODULE_HANDOFF_GUIDE.md](MODULE_HANDOFF_GUIDE.md): module conventions, standalone workflows, and OBC integration rules for subsystem owners.
- [DEMO_WALKTHROUGH.md](DEMO_WALKTHROUGH.md): step-by-step demo script and cause/effect story.
- [TELEMETRY_REFERENCE.md](TELEMETRY_REFERENCE.md): telemetry/events/manifest/downlink schemas and subsystem field ownership.

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
  - coarse reconstruction
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
- Image processing uses staged reconstruction: coarse map first, then adaptive ROI refinement.
- Progressive mode runs multiple stages; each stage refines only high-information ROIs and writes stage products.

## Build
```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

## Config Profiles

- `config/config.json`: existing default behavior (unchanged).
- `config/local_no_tcp.json`: local single-process validation (`jetson_transport=local`, ADCS-stability gate disabled, output at `out_local/`).
- `config/tcp_localhost.json`: two-process localhost TCP validation (`jetson_transport=tcp`, `127.0.0.1:5500`, bounded TCP timeouts, output at `out_tcp/`).
- `config/pi_hardware.json`: Pi/OBC-side deployment profile (Jetson host placeholder `192.168.0.50`, Pi-owned persistent output path `/var/lib/sgl/mission`).
- `config/jetson_hardware.json`: Jetson-side deployment profile (bind `0.0.0.0:5500`, scratch/cache paths under `/var/tmp/sgl/jetson_service`, CUDA backend placeholder enabled).
- `config/image_file_demo.json`: local printed-target/image-file payload ingestion demo (`payload_input_mode=image_file`).
- `config/pi_camera_demo.json`: local Pi-camera demo mode (`payload_input_mode=pi_camera_demo`, with source-image fallback when camera capture is unavailable).

## Scripts

- `scripts/build_all.sh`: builds integrated OBC/Jetson binaries and standalone subsystem modules.
- `scripts/run_local_demo.sh [config_path]`: runs local no-TCP OBC simulation (defaults to `config/local_no_tcp.json`).
- `scripts/run_tcp_jetson.sh [config_path]`: runs Jetson service in TCP mode (defaults to `config/tcp_localhost.json`).
- `scripts/run_tcp_pi.sh [config_path]`: runs Pi flight in TCP mode (defaults to `config/tcp_localhost.json`).
- `scripts/run_dashboard.sh [config_path] [refresh_ms]`: launches read-only dashboard for telemetry/events from the selected config output root.
- `scripts/clean_outputs.sh`: removes common output directories (`out*` presets).

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
./scripts/run_dashboard.sh config/local_no_tcp.json 200
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
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_image_file/mission_store/telemetry_cycles.csv --events out_image_file/mission_store/events.csv --manifest out_image_file/mission_store/products_manifest.csv --refresh-ms 200
```

Run with camera demo mode:

```bash
./scripts/run_local_demo.sh config/pi_camera_demo.json
python3 tools/telemetry_dashboard/dashboard.py --telemetry out_camera_demo/mission_store/telemetry_cycles.csv --events out_camera_demo/mission_store/events.csv --manifest out_camera_demo/mission_store/products_manifest.csv --refresh-ms 200
```

This is a physical demo acquisition path only. It is not the final SGL optical forward model.

Dashboard image preview panel (read-only) can show latest:

- raw capture
- rectified image
- ring preview/dataset input artifacts
- coarse product
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
./scripts/run_dashboard.sh config/tcp_localhost.json 200
```

## Workflow 3: Future Pi + Jetson Hardware Deployment

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
./scripts/run_dashboard.sh config/local_no_tcp.json 200
```

Use `config/tcp_localhost.json` or hardware profile paths as needed.

## Tests

```bash
ctest --test-dir build_wsl --output-on-failure
```

`test_tcp_mode` is included and validates two-process TCP Pi↔Jetson behavior. In restricted environments where localhost sockets are blocked, it reports `SKIP` and exits successfully.

Outputs appear in `out/` relative to the process working directory.

## Persistent Pi-Owned Data
- `out/mission_store/products_manifest.csv`: authoritative product manifest written by Pi.
- `out/mission_store/downlink_queue.csv`: Pi-owned downlink queue entries with priority and bit cost.
- `out/mission_store/telemetry_cycles.csv`: cycle-by-cycle autonomous telemetry snapshot.
  - Includes dynamic bus power terms (source/load/reserve/compute budget), subsystem power draws, Jetson mode/job/power, tracker validity/confidence/stars, truth+estimated pointing error, queue depth, and stage/ROI counters.

## Read-only Telemetry Dashboard

The dashboard is under `tools/telemetry_dashboard/` and only reads telemetry (no sim control).

```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
./scripts/run_dashboard.sh config/local_no_tcp.json 200
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

CSV interpretation:
- summary CSV: total runtime, ring generation timing, reconstruction timing, ROI-selection timing, manifest/product counts.
- stage CSV: per-stage observation counts, ROI count, coarse/refine runtime, total stage runtime, product paths.

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

- Dashboard import failure (for example `ModuleNotFoundError: pyqtgraph`):
  - install dashboard requirements:
  - `python3 -m pip install -r tools/telemetry_dashboard/requirements.txt`
  - `scripts/run_dashboard.sh` now checks dependencies first and prints this command instead of a traceback.

- Test portability:
  - runtime tests do not require `ripgrep` (`rg`) anymore.
  - standard POSIX tools are used in test scripts for product/telemetry checks.
