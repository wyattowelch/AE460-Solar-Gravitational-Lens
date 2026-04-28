# SGL EPS Module

Reusable `eps` subsystem model with standalone build/demo/test support.
It provides source power, load budgeting, total bus load, and low-power/bus-margin telemetry.

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
./build/eps_standalone_demo
```

## Test
```bash
ctest --test-dir build --output-on-failure
```

## API
Public headers are under `include/sgl/` and expose a model class with typed input/telemetry and dynamic power output.

`EpsModel` supports:
- `step(...)`: advances source-power dynamics and returns budget/bus telemetry
- `evaluate(...)`: evaluates budget/bus state without advancing time
