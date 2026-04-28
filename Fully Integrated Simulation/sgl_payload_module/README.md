# SGL PAYLOAD Module

Reusable `payload` subsystem model with standalone build/demo/test support.

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
./build/payload_standalone_demo
```

## Test
```bash
ctest --test-dir build --output-on-failure
```

## API
Public headers are under `include/sgl/` and expose a model class with typed input/telemetry and dynamic power output.

Payload telemetry includes:
- mode/state string
- active/inactive flag
- dataset-ready flag
- dataset id and dataset counter
- acquisition stage
- synthetic signal score (placeholder metadata)
- current payload power draw
