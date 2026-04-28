# System Overview

## Mission Architecture

This repository implements an integrated Solar Gravitational Lens spacecraft software simulation with a split compute architecture:

- Pi/OBC (`sgl_pi_flight`): mission authority
- Jetson (`sgl_jetson_service`): image-processing worker

Authority boundary:

- OBC owns mission state, scheduler/FDIR decisions, persistent storage, and product/downlink manifests.
- Jetson executes coarse/refine processing jobs requested by OBC and returns results.
- Jetson does not make mission-level decisions.

## Subsystem Modules

Integrated OBC behavior is composed from reusable subsystem modules in sibling repositories/directories:

- `../sgl_star_tracker_module/sgl_star_tracker_module`
- `../sgl_eps_module`
- `../sgl_thermal_module`
- `../sgl_comms_module`
- `../sgl_propulsion_module`
- `../sgl_payload_module`

Each module is intended to be independently buildable/testable and reusable by subsystem owners, while the OBC remains the integration and mission-authority layer.

## Operating Modes

Configuration profiles are in `config/`:

- `config/local_no_tcp.json`
  - single-process local validation
  - `jetson_transport: local`
  - `require_adcs_stable_for_jetson: false`
- `config/image_file_demo.json`
  - local printed-target/image ingestion demo
  - `payload_input_mode: image_file`
- `config/pi_camera_demo.json`
  - local Pi-camera demo acquisition (with file fallback when no camera is available)
  - `payload_input_mode: pi_camera_demo`
- `config/tcp_localhost.json`
  - two-process localhost Pi↔Jetson over TCP
- `config/pi_hardware.json`
  - Pi/OBC-side deployment profile with persistent mission storage paths
- `config/jetson_hardware.json`
  - Jetson-side deployment profile with bind host/port and scratch/cache paths

## Power-Aware Scheduling

OBC computes live noncompute load and available compute budget each cycle using EPS model outputs.

- Noncompute load includes ADCS, COMMS, thermal, propulsion, and payload dynamic power.
- Compute budget changes with subsystem activity.
- Scheduler mode (nominal/throttled/suspended) gates Jetson coarse/refine jobs.

Observed coupling examples:

- ADCS correction activity raises wheel/ADCS power, reducing compute margin.
- Heater or propulsion events raise noncompute load and can throttle/suspend Jetson.
- Dataset-ready events from payload drive processing demand.

## Event-Driven Subsystem Coupling

Subsystem transitions are logged to `events.csv` (append-only), including:

- ADCS correction start/stop
- tracker degraded/recovered
- heater on/off
- propulsion burn start/stop
- payload dataset ready
- Jetson coarse/refine start/complete/fail
- scheduler mode changes
- compute budget low/recovered
- downlink active/inactive
- Jetson unavailable transport events

This event log is used to explain cause/effect during demos and validation.

Payload demo acquisition adds camera/alignment/quality events and telemetry so reviewers can see why captures were accepted, rejected, or fallback-rectified.

## Read-Only Dashboard

`tools/telemetry_dashboard/` provides live visualization of telemetry and events.

- Input: `telemetry_cycles.csv` and optional `events.csv`
- Grouped subsystem metrics with checkbox toggles
- Event/status panel and optional plot event markers
- Read-only by design (no sim control path)

See `DEMO_WALKTHROUGH.md` for a demo sequence.
