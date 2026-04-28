# SGL COMMS Module

Reusable COMMS subsystem model with dynamic power draw and telemetry.

## Layout
- `include/`
- `src/`
- `examples/standalone_demo.cpp`
- `tests/`

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Run demo
```bash
./build/comms_standalone_demo
```

## Test
```bash
ctest --test-dir build --output-on-failure
```
