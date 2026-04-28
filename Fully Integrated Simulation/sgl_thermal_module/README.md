# SGL THERMAL Module

Reusable `thermal` subsystem model with standalone build/demo/test support.

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

## Run Demo
```bash
./build/thermal_standalone_demo
```

## Test
```bash
ctest --test-dir build --output-on-failure
```

## API
Public headers are under `include/sgl/` and expose a model class with typed input/telemetry and dynamic power output.

Thermal telemetry includes:
- mode/state string
- heater active flag
- heater power and total thermal power
- current temperature
- low/high temperature warning flags
