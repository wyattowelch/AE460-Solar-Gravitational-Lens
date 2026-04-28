# Module Handoff Guide

## Purpose

This guide is for subsystem owners who need to develop and validate their subsystem standalone, while still plugging into the integrated OBC mission simulation.

Core rule:

- Modules provide reusable subsystem state, telemetry, and dynamic power behavior.
- OBC (`sgl_pi_flight`) owns mission authority (scheduler, FDIR, Jetson gating, persistent mission data, telemetry aggregation).

## Module Convention

Target convention for subsystem modules:

- `include/` public headers/API
- `src/` implementation
- `examples/standalone_demo.cpp` standalone demonstration
- `tests/` module-level tests
- `CMakeLists.txt` standalone build entry
- `README.md` usage and API notes

Current status:

- `sgl_eps_module`, `sgl_thermal_module`, `sgl_comms_module`, `sgl_propulsion_module`, `sgl_payload_module` follow this pattern.
- `sgl_star_tracker_module/sgl_star_tracker_module` is a pre-existing standalone module with `docs/` and the same build/test/demo intent.

## Build/Run Standalone Modules

Example pattern (replace path per module):

```bash
cd /home/bunta/sgl/sgl_eps_module
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/eps_standalone_demo
```

Integrated helper script from this repo:

```bash
cd /home/bunta/sgl/sgl_obc_pi_jetson_full
./scripts/build_all.sh
```

This builds:

- integrated OBC/Jetson binaries in `build_wsl/`
- standalone module builds under `build_wsl/*_standalone_build`

## How Modules Plug Into OBC

Integration occurs through top-level CMake and OBC subsystem wrappers:

- `CMakeLists.txt` adds each module with `add_subdirectory(...)`
- `sgl_pi_lib` links module targets and includes their headers
- `pi_flight/subsystems.cpp` and `pi_flight/subsystems.hpp` wrap module APIs for OBC usage
- `pi_flight/main.cpp` orchestrates mission logic using those wrapper outputs

Integration expectations:

- Module outputs: state/mode, telemetry, power draw, subsystem-local flags.
- OBC consumes those outputs to:
  - compute EPS noncompute load and budget
  - decide scheduler mode
  - gate Jetson work
  - log mission telemetry/events

## Interface Expectations for Subsystem Owners

When extending a module, keep the interface reusable and mission-agnostic:

- deterministic step/update APIs where practical
- explicit telemetry fields
- explicit dynamic power draw output
- no mission-level authority logic in module internals

Avoid inside modules:

- scheduler mode decisions
- FDIR ownership
- Jetson dispatch policy
- persistent mission-store writes

Those remain in OBC.

## Typical Handoff Workflow

1. Owner validates subsystem standalone (demo + tests).
2. Owner shares API/telemetry/power changes.
3. Integrator updates OBC wrapper if needed.
4. Run integrated validation:
   - `./scripts/run_local_demo.sh`
   - `ctest --test-dir build_wsl --output-on-failure`

