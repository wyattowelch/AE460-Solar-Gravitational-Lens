#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
PI_BIN="${BUILD_DIR}/sgl_pi_flight"
BASE_CFG="${REPO_ROOT}/config/local_no_tcp.json"

if [[ ! -x "${PI_BIN}" ]]; then
  echo "ERROR: missing ${PI_BIN}. Run ./scripts/build_all.sh first." >&2
  exit 1
fi
if [[ ! -f "${BASE_CFG}" ]]; then
  echo "ERROR: missing ${BASE_CFG}" >&2
  exit 1
fi

cuda_build_enabled=0
if grep -q '^SGL_ENABLE_CUDA:BOOL=ON' "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null; then
  cuda_build_enabled=1
fi

probe_root="${REPO_ROOT}/out_backend_probe"
rm -rf "${probe_root}"
mkdir -p "${probe_root}"

make_cfg() {
  local out_cfg="$1"
  local out_dir="$2"
  local backend="$3"
  local allow_fb="$4"
  python3 - "${BASE_CFG}" "${out_cfg}" "${out_dir}" "${backend}" "${allow_fb}" <<'PY'
import json, sys
base, outp, out_dir, backend, allow_fb = sys.argv[1:]
cfg = json.load(open(base, "r", encoding="utf-8"))
cfg.update({
    "jetson_transport": "local",
    "require_adcs_stable_for_jetson": False,
    "jetson_backend": backend,
    "jetson_allow_cpu_fallback": allow_fb.lower() in ("1", "true", "yes"),
    "payload_input_mode": "image_file",
    "source_image": "../bluemarble.jpg",
    "power_cap_W": 120.0,
    "nominal_fraction": 0.90,
    "reserve_margin_W": 5.0,
    "sim_cycles": 80,
    "lowres_N": 128,
    "highres_N": 128,
    "progressive_base_N": 128,
    "progressive_max_N": 128,
    "progressive_scale": 2,
    "progressive_max_stages": 1,
    "profiling_mode": False,
    "profiling_force_full_compute": False,
    "required_max_resolution": 128,
    "optional_max_resolution": 128,
    "outputs_retention_enabled": False,
    "out_dir": out_dir,
    "jetson_scratch_dir": f"{out_dir}/jetson_scratch",
    "profile_name": f"backend_probe_{backend}_{'fb1' if (allow_fb.lower() in ('1','true','yes')) else 'fb0'}",
})
with open(outp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY
}

extract_status() {
  local run_dir="$1"
  python3 - "${run_dir}" <<'PY'
import csv, os, sys
run_dir = sys.argv[1]
manifest = os.path.join(run_dir, "mission_store", "products_manifest.csv")
events = os.path.join(run_dir, "mission_store", "events.csv")
status = ""
event_val = ""
if os.path.exists(manifest):
    with open(manifest, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = row.get("kind", "")
            if k.startswith("recon_"):
                status = row.get("status", "") or status
                if status:
                    break
if (not status) and os.path.exists(events):
    with open(events, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            et = row.get("event_type", "")
            if et in ("jetson_coarse_failed", "jetson_refine_failed", "jetson_unavailable"):
                event_val = row.get("value", "")
if status:
    print(status)
elif event_val:
    print(event_val)
else:
    print("")
PY
}

parse_field() {
  local status="$1"
  local key="$2"
  python3 - "${status}" "${key}" <<'PY'
import sys
s, k = sys.argv[1], sys.argv[2]
out = ""
for part in s.split(";"):
    p = part.strip()
    if p.startswith(k + "="):
        out = p.split("=", 1)[1].strip()
        break
print(out)
PY
}

run_case() {
  local name="$1"
  local backend="$2"
  local allow_fb="$3"
  local cfg="${probe_root}/${name}.json"
  local out_dir="${probe_root}/${name}"
  local log="${probe_root}/${name}.log"
  make_cfg "${cfg}" "${out_dir}" "${backend}" "${allow_fb}"
  local rc=0
  if ! "${PI_BIN}" --config "${cfg}" >"${log}" 2>&1; then
    rc=$?
  fi
  local status
  status=$(extract_status "${out_dir}")
  local req resolved fallback runtime reason
  req=$(parse_field "${status}" "backend_requested")
  resolved=$(parse_field "${status}" "backend_resolved")
  fallback=$(parse_field "${status}" "fallback_used")
  runtime=$(parse_field "${status}" "cuda_runtime_available")
  reason=$(parse_field "${status}" "backend_reason")
  echo "${name}|${backend}|${allow_fb}|${rc}|${req}|${resolved}|${fallback}|${runtime}|${reason}|${status}|${cfg}|${out_dir}|${log}"
}

cpu_row=$(run_case "cpu_local" "cpu" "true")
auto_row=$(run_case "auto_local" "auto" "true")
cuda_fb_true_row=$(run_case "cuda_local_fallback_true" "cuda" "true")
cuda_fb_false_row=$(run_case "cuda_local_fallback_false" "cuda" "false")

printf 'build_cuda_enabled=%d\n' "${cuda_build_enabled}"
echo "case,requested,allow_cpu_fallback,exit_code,backend_requested,backend_resolved,fallback_used,cuda_runtime_available,backend_reason"
for row in "${cpu_row}" "${auto_row}" "${cuda_fb_true_row}" "${cuda_fb_false_row}"; do
  IFS='|' read -r name requested allow rc req resolved fallback runtime reason status cfg out_dir log <<<"${row}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "${requested}" "${allow}" "${rc}" "${req}" "${resolved}" "${fallback}" "${runtime}" "${reason}"
done

python3 - "${cpu_row}" "${auto_row}" "${cuda_fb_true_row}" "${cuda_fb_false_row}" <<'PY'
import sys
rows = [r.split("|") for r in sys.argv[1:]]
ok = True
def need(cond, msg):
    global ok
    if not cond:
        print("FAIL:", msg)
        ok = False

by_name = {r[0]: r for r in rows}

cpu = by_name["cpu_local"]
need(cpu[5] == "cpu", "cpu_local should resolve to cpu")
need(cpu[6] in ("0", ""), "cpu_local should not use fallback")

auto = by_name["auto_local"]
need(auto[5] in ("cpu", "cuda"), "auto_local should resolve cpu or cuda")

fb1 = by_name["cuda_local_fallback_true"]
need(fb1[5] in ("cpu", "cuda"), "cuda_local_fallback_true should resolve cpu or cuda")

fb0 = by_name["cuda_local_fallback_false"]
if fb0[5] == "cuda":
    need(fb0[3] == "0", "cuda_local_fallback_false resolved cuda but run failed")
else:
    # On non-CUDA systems this should fail to produce successful backend resolution.
    need(fb0[5] in ("", "cpu"), "cuda_local_fallback_false unexpected resolution")

if not ok:
    raise SystemExit(1)
print("Backend probe checks passed.")
PY

echo "Probe artifacts:"
echo "  ${probe_root}"
echo "  $(find "${probe_root}" -maxdepth 1 -type f -name '*.log' | wc -l) logs"
