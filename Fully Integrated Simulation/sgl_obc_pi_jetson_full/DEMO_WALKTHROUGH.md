# Demo Walkthrough

## Objective

Show full-system autonomous cause/effect behavior with OBC authority, dynamic subsystem coupling, Jetson job orchestration, and read-only telemetry visualization.

## 1. Build

```bash
cd /home/bunta/sgl/sgl_obc_pi_jetson_full
./scripts/build_all.sh
```

## 2. Run Local Demo (No TCP)

```bash
./scripts/run_local_demo.sh
```

This uses `config/local_no_tcp.json` by default and writes outputs under:

- `out_local/mission_store/telemetry_cycles.csv`
- `out_local/mission_store/events.csv`
- `out_local/mission_store/products_manifest.csv`
- `out_local/mission_store/downlink_queue.csv`

## 3. Launch Dashboard (Read-Only)

```bash
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
./scripts/run_dashboard.sh config/local_no_tcp.json 200
```

## 4. Enable These Dashboard Groups

- Power/EPS
- ADCS
- Thermal
- Propulsion
- Payload
- COMMS
- Jetson/Processing

Keep event markers enabled for explanation.
In the image preview panel, use the dropdown to switch between:

- Raw Capture
- Rectified
- Ring Preview
- Coarse
- Refined

Optional physical-input demo runs:

```bash
./scripts/run_local_demo.sh config/image_file_demo.json
# or
./scripts/run_local_demo.sh config/pi_camera_demo.json
```

## 5. Cause/Effect Narrative to Present

Use telemetry curves together with event markers/events table.

1. ADCS correction:
- `adcs_correction_started` event appears.
- `wheel_power_w` and `adcs_power_w` rise.
- `compute_budget_w` drops.

2. Thermal heater activity:
- `heater_activated` event appears.
- `thermal_power_w` rises.
- `noncompute_w` rises and `compute_budget_w` decreases.

3. Propulsion burn:
- `propulsion_burn_started` event appears.
- `propulsion_power_w` rises.
- scheduler can move to throttled/suspended under lower compute margin.

4. Payload to processing chain:
- `payload_dataset_ready` event appears.
- `jetson_coarse_started/completed` and `jetson_refine_started/completed` events follow.
- `jetson_job_type`, `jetson_mode`, `processing_queue`, and manifest entries update.

4a. Camera/printed-target ingest transparency:
- `camera_capture_started/completed/failed` events show acquisition path.
- `payload_alignment_succeeded/failed` shows marker-vs-fallback rectification.
- `payload_capture_accepted/rejected` with quality-score values shows gate decisions.

5. Autonomous decision transparency:
- `scheduler_mode_changed`, `compute_budget_low/recovered`, and transport/failure events explain why behavior changed.

## 6. Artifacts to Show Reviewers

- `products_manifest.csv` demonstrates coarse/refined product lifecycle.
- `downlink_queue.csv` demonstrates OBC-owned queueing and priority.
- `events.csv` provides timeline of autonomous decisions.
- `telemetry_cycles.csv` provides quantitative subsystem coupling.

## 7. Optional TCP Localhost Demo

Terminal A:

```bash
./scripts/run_tcp_jetson.sh
```

Terminal B:

```bash
./scripts/run_tcp_pi.sh
```

Then:

```bash
./scripts/run_dashboard.sh config/tcp_localhost.json 200
```

If localhost sockets are restricted in your environment, use local mode for demo continuity.
