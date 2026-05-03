#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
CFG_DEFAULT="${REPO_ROOT}/config/local_no_tcp.json"
CFG_PATH="${1:-${CFG_DEFAULT}}"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config not found: ${CFG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/sgl_pi_flight" ]]; then
  echo "Binary missing: ${BUILD_DIR}/sgl_pi_flight"
  echo "Building..."
  "${SCRIPT_DIR}/build_all.sh"
fi

cfg_summary=$(
python3 - "${CFG_PATH}" "${REPO_ROOT}/config/output_retention_defaults.json" <<'PY'
import json, os, sys
cfg=json.load(open(sys.argv[1], "r", encoding="utf-8"))
defaults={}
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    try:
        defaults=json.load(open(sys.argv[2], "r", encoding="utf-8"))
    except Exception:
        defaults={}
for k,v in defaults.items():
    cfg.setdefault(k, v)

def b(name, default):
    v=cfg.get(name, default)
    return "1" if bool(v) else "0"
def i(name, default):
    try: return str(int(cfg.get(name, default)))
    except Exception: return str(default)
def f(name, default):
    try: return f"{float(cfg.get(name, default)):.3f}"
    except Exception: return f"{float(default):.3f}"

out_dir=cfg.get("out_dir", "out")
profile_name=cfg.get("profile_name") or os.path.splitext(os.path.basename(sys.argv[1]))[0]
progressive_max=int(cfg.get("progressive_max_N", 1024))
ring_sensor=int(cfg.get("ring_sensor_N", 0))
profiling_mode=bool(cfg.get("profiling_mode", False))
high_fidelity=profiling_mode or progressive_max >= 2048 or ring_sensor >= 4096

base=max(16, int(cfg.get("progressive_base_N", 128)))
mx=max(base, int(cfg.get("progressive_max_N", 1024)))
scale=max(2, int(cfg.get("progressive_scale", 2)))
stages=[]
n=base
for _ in range(max(1, int(cfg.get("progressive_max_stages", 4)))):
    if n > mx:
        break
    stages.append(n)
    if n == mx:
        break
    n=min(mx, n*scale)

obs_total=max(1, int(cfg.get("observation_count_stage0", 1)))
obs_total += max(0, int(cfg.get("observation_count_stage1", 1)))
obs_total += max(0, int(cfg.get("observation_count_stage2", 1)))
obs_total += max(0, int(cfg.get("observation_count_stage3", 1)))

store_every=max(1, int(cfg.get("store_ring_preview_every", 16)))
ring_n=max(0, int(cfg.get("ring_sensor_N", 0)))
ring_bytes_per_frame=ring_n * ring_n * 3
preview_count=max(1, (obs_total + store_every - 1)//store_every)
full_frames=bool(cfg.get("store_full_ring_frames", False) or cfg.get("store_all_full_ring_frames_debug", False))

stage_bytes=0
if stages:
    stage_bytes += stages[0] * stages[0] * 3
    for s in stages[1:]:
        stage_bytes += s * s * 3 * 2
diag_n=max(1, int(cfg.get("source_canvas_N", 1024)))
diag_bytes=diag_n * diag_n * 3 * 6
annulus_bytes=max(1, int(cfg.get("ring_angular_samples", 8192))) * max(1, int(cfg.get("ring_radial_samples", 96))) * 4 * obs_total
ring_preview_bytes=preview_count * ring_bytes_per_frame
full_ring_bytes=(obs_total * ring_bytes_per_frame) if full_frames else 0
estimate_bytes=stage_bytes + diag_bytes + annulus_bytes + ring_preview_bytes + full_ring_bytes
estimate_gb=estimate_bytes / (1024**3)

fields=[
    out_dir,
    profile_name,
    b("outputs_retention_enabled", True),
    i("outputs_keep_lightweight_runs", 10),
    i("outputs_keep_full_runs", 3),
    f("outputs_max_total_gb", 0.0),
    b("outputs_prune_raw_ppm", True),
    b("outputs_prune_ring_frames", True),
    b("outputs_prune_annulus_dumps", True),
    b("outputs_preserve_marked_runs", True),
    f("min_free_disk_gb_before_run", 0.0),
    f("warn_free_disk_gb", 25.0),
    f("fail_if_disk_below_gb", 10.0),
    "1" if high_fidelity else "0",
    f"{estimate_gb:.3f}",
    b("outputs_retention_include_out_profile", False),
    b("outputs_retention_include_working_outs", False),
]
print("\t".join(fields))
PY
)

IFS=$'\t' read -r out_dir case_name retention_enabled keep_lightweight_runs keep_full_runs max_total_gb prune_raw_ppm prune_ring_frames prune_annulus preserve_marked min_free_gb warn_free_gb fail_free_gb high_fidelity estimate_gb include_out_profile include_working_outs <<<"${cfg_summary}"

if [[ "${out_dir}" = /* ]]; then
  out_root="${out_dir}"
else
  out_root="${REPO_ROOT}/${out_dir}"
fi

free_bytes_before=$(df -PB1 "${REPO_ROOT}" | awk 'NR==2 {print $4}')
free_gb_before=$(python3 - "${free_bytes_before:-0}" <<'PY'
import sys
v=float(sys.argv[1] or 0)
print(f"{v/(1024**3):.3f}")
PY
)

echo "Disk check:"
echo "  free before run: ${free_gb_before} GB"
echo "  estimated output footprint: ~${estimate_gb} GB"
if [[ "${high_fidelity}" == "1" ]]; then
  echo "  run class: high-fidelity/profile"
fi

python3 - "${free_gb_before}" "${min_free_gb}" "${warn_free_gb}" "${fail_free_gb}" "${high_fidelity}" <<'PY'
import sys
free=float(sys.argv[1]); min_free=float(sys.argv[2]); warn=float(sys.argv[3]); fail=float(sys.argv[4]); high=(sys.argv[5] == "1")
if warn > 0 and free < warn:
    print(f"WARNING: free disk below warning threshold ({free:.3f} GB < {warn:.3f} GB).", file=sys.stderr)
if min_free > 0 and free < min_free:
    print(f"ERROR: free disk below min_free_disk_gb_before_run ({free:.3f} GB < {min_free:.3f} GB).", file=sys.stderr)
    raise SystemExit(1)
if high and fail > 0 and free < fail:
    print(f"ERROR: high-fidelity/profile run blocked by fail_if_disk_below_gb ({free:.3f} GB < {fail:.3f} GB).", file=sys.stderr)
    raise SystemExit(1)
PY

case "${out_root}" in
  "${REPO_ROOT}/out"|"${REPO_ROOT}/out_"*|"${REPO_ROOT}/out-"*)
    rm -rf "${out_root}"
    ;;
esac

pushd "${REPO_ROOT}" >/dev/null
"${BUILD_DIR}/sgl_pi_flight" --config "${CFG_PATH}"
popd >/dev/null

free_bytes_after=$(df -PB1 "${REPO_ROOT}" | awk 'NR==2 {print $4}')
free_gb_after=$(python3 - "${free_bytes_after:-0}" <<'PY'
import sys
v=float(sys.argv[1] or 0)
print(f"{v/(1024**3):.3f}")
PY
)

RUN_DISK_FREE_BEFORE_GB="${free_gb_before}" \
RUN_DISK_FREE_AFTER_GB="${free_gb_after}" \
RUN_OUTPUT_ESTIMATE_GB="${estimate_gb}" \
"${SCRIPT_DIR}/package_run_outputs.sh" \
  --case-name "${case_name}" \
  --source-out-root "${out_root}" \
  --config-path "${CFG_PATH}"

echo "Local demo complete. Outputs:"
echo "  telemetry: ${out_root}/mission_store/telemetry_cycles.csv"
echo "  events:    ${out_root}/mission_store/events.csv"
echo "  manifest:  ${out_root}/mission_store/products_manifest.csv"
echo "  downlink:  ${out_root}/mission_store/downlink_queue.csv"
echo "  free disk after run: ${free_gb_after} GB"
