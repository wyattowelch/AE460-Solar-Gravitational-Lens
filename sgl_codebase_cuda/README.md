# SGL Data Handling Codebase (Simulation + Flight-Style Hybrid Scaffold)

This repo contains **two** build targets:

- `sgl_sim` : CPU/OpenMP proof-of-concept pipeline (tile -> ring -> reconstruction) producing **PPM** outputs.
- `sgl_jetson` : Flight-style **power-aware scheduler** (CPU fallback) that models how a Jetson-class processor would
  run low-res quick-look reconstructions first, then contrast-aware refinement, under an Arduino-enforced power policy.

## Why PPM instead of PNG?
To keep the C++ build dependency-free and portable, outputs are written as **PPM/PGM**.
If you need PNGs:
- `python3 scripts/convert_ppm_to_png.py out/ring_highres.ppm out/ring_highres.png`
- `python3 scripts/convert_to_ppm.py bluemarble.jpg bluemarble.ppm`

## Build (WSL/Linux/macOS)
```bash
mkdir -p build
cd build
cmake ..
make -j
```

## Run the simulation
1) Convert your source image to PPM:
```bash
python3 scripts/convert_to_ppm.py bluemarble.jpg bluemarble.ppm
```
2) Run:
```bash
./build/sgl_sim --config config/config.json
```
Outputs go to `out/`:
- `src_with_tiles.ppm`
- `ring_lowres.ppm`
- `ring_highres.ppm`
- `reconstructed_from_tiles.ppm`

## Run the Jetson scheduler demo
```bash
./build/sgl_jetson --config config/config.json
```
It writes logs to `out/logs/sgl_jetson.log`.

### Power telemetry stub
If you create `out/power_telemetry.json` with:
```json
{"available_W": 22.0, "draw_W": 5.0}
```
the scheduler will adapt as power changes.
In flight, this telemetry would come from the Arduino over UART/I2C/CAN.

## What’s still mission-specific / not finalized
- Real sensor power telemetry (EPS interface + calibration)
- Real camera interfaces and data formats
- Radiation-tolerant storage and file system strategy
- Full PSF-accurate deconvolution at full resolution (CUDA FFT + regularized inversion)


## CUDA (Jetson) acceleration
If you build on a Jetson with `nvcc` available:
```bash
cmake -DENABLE_CUDA=ON ..
make -j
```
This adds CUDA kernels in `jetson/cuda/` for PSF evaluation and the Wiener-deconvolution interface.
If cuFFT is available, you can enable it by defining `SGL_USE_CUFFT` and linking against `cufft`.

