# Jetson CUDA + TCP Setup

## Build Modes

CPU/default build:

```bash
./scripts/build_all.sh
```

CUDA-enabled build (Jetson):

```bash
SGL_ENABLE_CUDA=1 ./scripts/build_all.sh
```

If CUDA compiler/runtime is unavailable, build continues in CPU mode unless strict CUDA is requested at runtime.

## Backend Resolution Rules

- `jetson_backend=cpu`: always CPU.
- `jetson_backend=auto`: CUDA if available, otherwise CPU.
- `jetson_backend=cuda`:
  - use CUDA if available
  - if unavailable and `jetson_allow_cpu_fallback=true`, fall back to CPU
  - if unavailable and `jetson_allow_cpu_fallback=false`, fail job clearly

Status strings in `products_manifest.csv` include:

- `backend_requested`
- `backend_resolved`
- `cuda_build_enabled`
- `cuda_runtime_available`
- `fallback_used`
- `backend_reason`

## Probe

```bash
./scripts/probe_jetson_backend.sh
```

This runs local backend probe cases and prints a behavior table plus logs under `out_backend_probe/`.

## TCP Localhost

CPU:

```bash
./scripts/run_tcp_jetson.sh config/tcp_pi_to_jetson_cpu.json
./scripts/run_tcp_pi.sh config/tcp_pi_to_jetson_cpu.json
```

CUDA-with-fallback:

```bash
./scripts/run_tcp_jetson.sh config/tcp_pi_to_jetson_cuda.json
./scripts/run_tcp_pi.sh config/tcp_pi_to_jetson_cuda.json
```

## Pi + Jetson Hardware

1. Put both nodes on the same network.
2. On Jetson, get IP:

```bash
hostname -I
```

3. On Pi, test connectivity:

```bash
ping <jetson_ip>
nc -vz <jetson_ip> 5500
```

4. Update host/port in config.
5. Start Jetson service first, then Pi flight.
