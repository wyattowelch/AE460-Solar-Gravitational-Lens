# SGL PROPULSION Module

Reusable `propulsion` subsystem model with standalone build/demo/test support.

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
./build/propulsion_standalone_demo
```

## Test
```bash
ctest --test-dir build --output-on-failure
```

## API
Public headers are under `include/sgl/` and expose a model class with typed input/telemetry and dynamic power output.

Propulsion telemetry includes:
- mode/state string
- active/inactive flag
- burn-event flag
- current power draw
- representative thrust (`thrust_n`)
- remaining propellant estimate (`remaining_propellant_kg`)
