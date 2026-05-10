# AE460 Solar Gravitational Lens

This repository contains standalone subsystem simulations and the integrated OBC/payload live simulation for the Solar Gravitational Lens mission architecture.

## Repository layout

```text
ADCS/
Electrical/
Flight Dynamics/
Fully Integrated Simulation/
Mechanical/
OBC/
Payload/
Propulsion/
Test Folder/
Thermal/
TT&C/
```

`Electrical/` is present in the repository, but no usable setup/run instructions are documented here yet.

---

## 1. Standalone subsystem simulations

These are smaller, separate simulations. They are not part of the integrated OBC/payload codebase.

### Propulsion

Files:

```text
Propulsion/FinalMassCalcs.m
Propulsion/PostSolarIonProp.m
Propulsion/PreSolarIon.m
```

Run instructions:

| File | How to run |
| --- | --- |
| `FinalMassCalcs.m` | Open in MATLAB, edit values in the `Setup` section as needed, then run. Results appear in the Command Window and generated plots. |
| `PostSolarIonProp.m` | Open in MATLAB, edit values in the `Ion Prop Calcs` section as needed, then run. Review generated plots. |
| `PreSolarIon.m` | Open in MATLAB, edit values in the setup sections as needed, then run. Results appear in the Command Window. |

### Flight Dynamics

File:

```text
Flight Dynamics/SGL_Mission_FlightDynamics.m
```

`SGL_Mission_FlightDynamics.m` is a simple mission architecture calculator that returns theoretical estimates. The script is commented and should explain its internal calculations.

Important note: update the perihelion distance in the script before relying on the output:

```matlab
r_perihelion = 0.14 * AU;
```

The existing value may be:

```matlab
r_perihelion = 0.046 * AU;
```

The flight dynamics outputs should be verified using empirical solar system data and a higher-fidelity tool such as AGI Systems Tool Kit/STK.

### Thermal

File:

```text
Thermal/final_thermal_simulation.m
```

The thermal simulation is a MATLAB two-node lumped-parameter transient thermal model.

Run instructions:

1. Open MATLAB and navigate to the directory containing `final_thermal_simulation.m`.
2. Run the script.
3. The script automatically runs both hot and cold bounding cases.
4. Review the generated outputs:
   - Command Window summary of maximum and minimum temperatures
   - Temperature vs. time plot for hot and cold cases
   - Temperature vs. heliocentric distance plot

### TT&C

File:

```text
TT&C/optical_link_budget_spring_2026_final_report.m
```

Run instructions:

1. Open MATLAB and open `optical_link_budget_spring_2026_final_report.m`.
2. Run the script.
3. Review results in the Command Window.

No additional MATLAB toolboxes are required.

### ADCS

File:

```text
ADCS/Destiny_SGL_ADCS.m
```

The ADCS simulation implements a MATLAB-style attitude determination and control simulation based on the PDR architecture:

- 4 star trackers using the Sodern SED16 model
- 4 reaction wheels in a pyramid configuration using the Honeywell HR04 model
- OBC closes the loop using tracker/gyro inputs
- Mission mode assumes ultra-fine pointing

Run instructions:

1. Open MATLAB and open `Destiny_SGL_ADCS.m`.
2. Run the script.
3. Review results in the Command Window.

No additional MATLAB toolboxes are required.

### Mechanical

No setup or run instructions are currently documented.

---

## 2. OBC and Payload

The integrated OBC/payload simulation lives under:

```bash
cd "/AE460-Solar-Gravitational-Lens/Fully Integrated Simulation/sgl_obc_pi_jetson_full"
```

If the repository is cloned somewhere else, replace `/AE460-Solar-Gravitational-Lens` with the actual repository root path.

### 2.1 System dependencies

Install Linux system dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip python3-venv ffmpeg fswebcam
```

### 2.2 Python environment and dashboard dependencies

Recommended setup:

```bash
cd "/AE460-Solar-Gravitational-Lens/Fully Integrated Simulation/sgl_obc_pi_jetson_full"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -c "from PyQt5 import QtCore; print(QtCore.QT_VERSION_STR)"
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
```

Using `--system-site-packages` allows the virtual environment to see system-installed packages such as PyQt5.

### 2.3 Build and test

CPU build:

```bash
./scripts/build_all.sh
```

CUDA build, only on a CUDA-capable host:

```bash
SGL_ENABLE_CUDA=1 ./scripts/build_all.sh
```

Run tests:

```bash
ctest --test-dir build_wsl --output-on-failure
```

### 2.4 Basic demo runs

Run without the GUI:

```bash
./scripts/run_local_demo.sh config/image_file_demo.json
```

Run a live GUI dashboard:

```bash
./scripts/run_gui_demo.sh config/live_systems_demo.json 300
```

The `300` value is the GUI refresh period in milliseconds. The GUI run saves the telemetry outputs shown in the dashboard and the images processed during the run.

Review the latest completed run:

```bash
./scripts/run_dashboard.sh --review outputs/latest 1000
```

### 2.5 Dashboard smoke checks

Check the Python dashboard files without launching the GUI:

```bash
python3 -m py_compile tools/telemetry_dashboard/core.py tools/telemetry_dashboard/dashboard.py
PYTHONPATH=tools/telemetry_dashboard python3 -m unittest tools/telemetry_dashboard/tests/test_core_smoke.py
```

If a script fails with a permissions error, make the shell scripts executable and reinstall dashboard dependencies:

```bash
chmod +x scripts/*.sh tests/*.sh
python3 -m pip install -r tools/telemetry_dashboard/requirements.txt
```

### 2.6 GUI playback tuning

For the live dashboard command:

```bash
./scripts/run_gui_demo.sh config/live_systems_demo.json 300
```

- Increase the refresh period above `300` ms if the GUI lags.
- Increase `live_playback_lag_cycles` in the config to increase the playback buffer.
- Example: `300` ms refresh with `live_playback_lag_cycles = 30` gives about a 9 second buffer.
- Increase `catchup_multiplier` above `1.0` if the GUI should speed up playback to stay near the configured lag target.
- Keep `catchup_multiplier` close to `1.0` if smooth playback matters more than staying close to real time.
- For faster display, a refresh period near `16.67` ms approximates 60 FPS. Use a larger `live_playback_lag_cycles` value, such as `1000+`, to reduce stuttering.

---

## 3. OBC and Payload full run matrix

This matrix covers Earth and Mars runs across:

- Local and TCP transport
- CPU and CUDA backends
- Camera and no-camera modes

Start from the integrated simulation directory with the Python environment active:

```bash
cd "/AE460-Solar-Gravitational-Lens/Fully Integrated Simulation/sgl_obc_pi_jetson_full"
source .venv/bin/activate
```

### 3.1 Build for the desired backend

CPU:

```bash
./scripts/build_all.sh
```

CUDA:

```bash
SGL_ENABLE_CUDA=1 ./scripts/build_all.sh
```

### 3.2 Generate matrix configs once

```bash
mkdir -p config/run_matrix
python3 - <<'PY'
import json, pathlib
root = pathlib.Path("config/run_matrix")
root.mkdir(parents=True, exist_ok=True)
base = json.load(open("config/earth_jpg_2048_force_complete.json", "r", encoding="utf-8"))
planets = {"earth": "../bluemarble.jpg", "mars": "../mars.jpg"}
transports = {"local": "local", "tcp": "tcp"}
backends = {"cpu": "cpu", "cuda": "cuda"}
modes = {"nocam": "image_file", "camera": "pi_camera_demo"}
for p, src in planets.items():
    for tname, tval in transports.items():
        for bname, bval in backends.items():
            for mname, mval in modes.items():
                c = dict(base)
                c["source_image"] = src
                c["payload_input_mode"] = mval
                c["jetson_transport"] = tval
                c["jetson_backend"] = bval
                c["jetson_allow_cpu_fallback"] = True
                c["require_adcs_stable_for_jetson"] = (tval == "tcp")
                c["host"] = "127.0.0.1"
                c["port"] = 5500
                c["profile_name"] = f"{p}_{tname}_{bname}_{mname}_2048"
                c["out_dir"] = f"out_{p}_{tname}_{bname}_{mname}_2048"
                c["jetson_scratch_dir"] = f"{c['out_dir']}/jetson_scratch"
                out = root / f"{p}_{tname}_{bname}_{mname}.json"
                json.dump(c, open(out, "w", encoding="utf-8"), indent=2)
                print(out)
PY
```

### 3.3 Non-TCP local GUI runs

Earth:

```bash
./scripts/run_gui_demo.sh config/run_matrix/earth_local_cpu_nocam.json 300
./scripts/run_gui_demo.sh config/run_matrix/earth_local_cuda_nocam.json 300
./scripts/run_gui_demo.sh config/run_matrix/earth_local_cpu_camera.json 300
./scripts/run_gui_demo.sh config/run_matrix/earth_local_cuda_camera.json 300
```

Mars:

```bash
./scripts/run_gui_demo.sh config/run_matrix/mars_local_cpu_nocam.json 300
./scripts/run_gui_demo.sh config/run_matrix/mars_local_cuda_nocam.json 300
./scripts/run_gui_demo.sh config/run_matrix/mars_local_cpu_camera.json 300
./scripts/run_gui_demo.sh config/run_matrix/mars_local_cuda_camera.json 300
```

### 3.4 TCP runs

Each TCP case uses three terminals.

Terminal 1, dashboard:

```bash
./scripts/run_dashboard.sh --live <config> 300
```

Terminal 2, Jetson service:

```bash
./scripts/run_tcp_jetson.sh <config>
```

Terminal 3, Pi flight:

```bash
./scripts/run_tcp_pi.sh <config>
```

Use one of these configs for `<config>`:

| Case | Config |
| --- | --- |
| Earth TCP CPU no camera | `config/run_matrix/earth_tcp_cpu_nocam.json` |
| Earth TCP CUDA no camera | `config/run_matrix/earth_tcp_cuda_nocam.json` |
| Earth TCP CPU camera | `config/run_matrix/earth_tcp_cpu_camera.json` |
| Earth TCP CUDA camera | `config/run_matrix/earth_tcp_cuda_camera.json` |
| Mars TCP CPU no camera | `config/run_matrix/mars_tcp_cpu_nocam.json` |
| Mars TCP CUDA no camera | `config/run_matrix/mars_tcp_cuda_nocam.json` |
| Mars TCP CPU camera | `config/run_matrix/mars_tcp_cpu_camera.json` |
| Mars TCP CUDA camera | `config/run_matrix/mars_tcp_cuda_camera.json` |

Example TCP case:

```bash
# Terminal 1
./scripts/run_dashboard.sh --live config/run_matrix/earth_tcp_cpu_nocam.json 300

# Terminal 2
./scripts/run_tcp_jetson.sh config/run_matrix/earth_tcp_cpu_nocam.json

# Terminal 3
./scripts/run_tcp_pi.sh config/run_matrix/earth_tcp_cpu_nocam.json
```

### 3.5 Output checks

Check the latest run pointer:

```bash
ls -ld outputs/latest
```

Check a specific local output directory:

```bash
ls out_earth_local_cpu_nocam_2048/mission_store
ls out_earth_local_cpu_nocam_2048/products
```

Open a completed run in review mode:

```bash
./scripts/run_dashboard.sh --review outputs/latest 1000
```

### 3.6 Matrix notes

- On non-CUDA systems, CUDA-requested configs can fall back to CPU when `jetson_allow_cpu_fallback=true`.
- For strict CUDA-only behavior, set `jetson_allow_cpu_fallback=false` in the config.
- If a camera is unavailable, `pi_camera_demo` may fall back to the configured source image depending on the current config and implementation.
