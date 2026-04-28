#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
PI_BIN="${BUILD_DIR}/sgl_pi_flight"
OUT_ROOT="${REPO_ROOT}/out_profile"
SUMMARY_CSV="${OUT_ROOT}/progressive_profile_summary.csv"
STAGE_AGG_CSV="${OUT_ROOT}/progressive_stage_timings.csv"

PROFILES=(
  "profile_progressive_fast.json"
  "profile_progressive_balanced.json"
  "profile_progressive_full.json"
  "profile_progressive_stress.json"
)

if [[ ! -x "${PI_BIN}" ]]; then
  echo "ERROR: binary missing: ${PI_BIN}" >&2
  echo "Run scripts/build_all.sh first." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
for d in fast_demo balanced_demo full_demo stress; do
  rm -rf "${OUT_ROOT:?}/${d}"
done

echo "profile_name,stages,observations_by_stage,max_resolution,total_runtime_ms,ring_generation_ms,reconstruction_ms,roi_selection_ms,products_written,manifest_rows,completed_required_stages,completed_optional_stages,missing_required_outputs,missing_optional_outputs,scheduler_throttle_count,scheduler_suspend_count,output_dir,exit_status" >"${SUMMARY_CSV}"
echo "profile_name,stage_index,out_n,observations_used,new_observations_added,roi_count,coarse_runtime_ms,refine_runtime_ms,roi_selection_ms,total_stage_runtime_ms,coarse_path,refined_path" >"${STAGE_AGG_CSV}"

run_one() {
  local cfg_name=$1
  local cfg_path="${REPO_ROOT}/config/${cfg_name}"
  local run_log="${OUT_ROOT}/$(basename "${cfg_name}" .json)_stdout.log"
  local start_ns end_ns runtime_ms

  read -r profile_name out_dir stages obs_stage max_res required_max optional_max <<EOF
$(python3 - "${cfg_path}" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
base=max(16,int(cfg.get("progressive_base_N",128)))
mx=max(base,int(cfg.get("progressive_max_N",1024)))
scale=max(2,int(cfg.get("progressive_scale",2)))
max_stages=max(1,int(cfg.get("progressive_max_stages",4)))
vals=[]
n=base
for _ in range(max_stages):
    if n>mx: break
    vals.append(n)
    if n==mx: break
    n=min(mx,n*scale)
obs=[int(cfg.get("observation_count_stage0",1)),int(cfg.get("observation_count_stage1",1)),int(cfg.get("observation_count_stage2",1)),int(cfg.get("observation_count_stage3",1))]
required_max=int(cfg.get("required_max_resolution", vals[-1] if vals else 0))
optional_max=int(cfg.get("optional_max_resolution", required_max))
print(cfg.get("profile_name","default"),cfg.get("out_dir","out"),"|".join(map(str,vals)),"|".join(str(obs[min(i,3)]) for i in range(len(vals))),vals[-1] if vals else 0,required_max,optional_max)
PY
)
EOF

  start_ns=$(date +%s%N)
  set +e
  "${PI_BIN}" --config "${cfg_path}" >"${run_log}" 2>&1
  local rc=$?
  set -e
  end_ns=$(date +%s%N)
  runtime_ms=$(( (end_ns - start_ns) / 1000000 ))

  local out_root
  if [[ "${out_dir}" = /* ]]; then
    out_root="${out_dir}"
  else
    out_root="${REPO_ROOT}/${out_dir}"
  fi
  local mission="${out_root}/mission_store"
  local manifest="${mission}/products_manifest.csv"
  local events="${mission}/events.csv"
  local stage_csv="${mission}/progressive_stage_timings.csv"

  local ring_ms recon_ms roi_ms products_written manifest_rows
  local completed_required completed_optional missing_required missing_optional throttle_count suspend_count
  ring_ms=0
  recon_ms=0
  roi_ms=0
  products_written=0
  manifest_rows=0
  completed_required=""
  completed_optional=""
  missing_required=""
  missing_optional=""
  throttle_count=0
  suspend_count=0

  set +e
  python3 - "${manifest}" "${events}" "${stage_csv}" "${stages}" "${out_root}" "${STAGE_AGG_CSV}" "${profile_name}" "${obs_stage}" "${required_max}" "${optional_max}" "${mission}/telemetry_cycles.csv" >"${OUT_ROOT}/.${profile_name}_metrics.tmp" <<'PY'
import csv, os, sys
manifest, events, stage_csv, stages_s, out_root, stage_agg_csv, profile_name, obs_stage, required_max_s, optional_max_s, telemetry_csv = sys.argv[1:]
stages=[int(x) for x in stages_s.split("|") if x]
required_max=int(required_max_s)
optional_max=int(optional_max_s)
required_stages=[s for s in stages if s <= required_max]
optional_stages=[s for s in stages if required_max < s <= optional_max]
def fail(msg):
    print(f"ERROR:{msg}")
    raise SystemExit(1)
if not os.path.exists(manifest):
    fail(f"missing manifest: {manifest}")
rows=list(csv.DictReader(open(manifest,newline="")))
if not rows:
    fail("manifest empty")
manifest_rows=len(rows)
products_written=manifest_rows
found_coarse={}
found_refined={}
for r in rows:
    try:
        n=int(r.get("out_n","0"))
    except Exception:
        continue
    kind=r.get("kind","").strip('"')
    p=(r.get("path") or "").strip().strip('"')
    if not p:
        fail("empty path in manifest")
    cands=[p, os.path.join(os.getcwd(), p), os.path.join(out_root, p)]
    p_exist=None
    for c in cands:
        if os.path.exists(c):
            p_exist=c
            break
    if p_exist is None:
        fail(f"dangling manifest path: {p}")
    if kind == "coarse":
        found_coarse[n]=p_exist
    if kind == "refined":
        found_refined[n]=p_exist
    # optional resolution check for PPM
    try:
        with open(p_exist, "rb") as f:
            magic=f.readline().strip()
            if magic==b"P6":
                dims=f.readline().strip().split()
                if len(dims)>=2:
                    w=int(dims[0]); h=int(dims[1])
                    if n>0 and (w!=n or h!=n):
                        fail(f"image size mismatch out_n={n} got {w}x{h} file={p_exist}")
    except Exception:
        pass
missing_required=[]
missing_optional=[]
completed_required=[]
completed_optional=[]
for n in sorted(required_stages):
    stage_missing=[]
    if n not in found_coarse:
        stage_missing.append("coarse")
    if n not in found_refined:
        stage_missing.append("refined")
    if stage_missing:
        missing_required.append(f"{n}:{'+'.join(stage_missing)}")
    else:
        completed_required.append(str(n))
for n in sorted(optional_stages):
    stage_missing=[]
    if n not in found_coarse:
        stage_missing.append("coarse")
    if n not in found_refined:
        stage_missing.append("refined")
    if stage_missing:
        missing_optional.append(f"{n}:{'+'.join(stage_missing)}")
    else:
        completed_optional.append(str(n))

if missing_required:
    fail("missing required outputs: " + "|".join(missing_required))

ring_ms=0.0
if os.path.exists(events):
    for e in csv.DictReader(open(events,newline="")):
        if (e.get("event_type","").strip('"')=="ring_generation_timing"):
            try: ring_ms += float((e.get("value") or "0").strip('"'))
            except Exception: pass

recon_ms=0.0
roi_ms=0.0
if os.path.exists(stage_csv):
    srows=list(csv.DictReader(open(stage_csv,newline="")))
    for s in srows:
        try: recon_ms += float(s.get("coarse_runtime_ms","0")) + float(s.get("refine_runtime_ms","0"))
        except Exception: pass
        try: roi_ms += float(s.get("roi_selection_ms","0"))
        except Exception: pass
    # append stage timings into aggregated top-level CSV
    with open(stage_agg_csv, "a", newline="") as out:
        w=csv.writer(out)
        for s in srows:
            w.writerow([
                profile_name,
                s.get("stage_index",""),
                s.get("out_n",""),
                s.get("observations_used",""),
                s.get("new_observations_added",""),
                s.get("roi_count",""),
                s.get("coarse_runtime_ms",""),
                s.get("refine_runtime_ms",""),
                s.get("roi_selection_ms",""),
                s.get("total_stage_runtime_ms",""),
                s.get("coarse_path",""),
                s.get("refined_path",""),
            ])
throttle_count=0
suspend_count=0
if os.path.exists(telemetry_csv):
    for t in csv.DictReader(open(telemetry_csv,newline="")):
        mode=(t.get("scheduler_mode") or "").strip().strip('"')
        if mode=="1":
            throttle_count += 1
        elif mode=="2":
            suspend_count += 1
print(f"{ring_ms},{recon_ms},{roi_ms},{products_written},{manifest_rows},{'|'.join(completed_required)},{'|'.join(completed_optional)},{'|'.join(missing_required)},{'|'.join(missing_optional)},{throttle_count},{suspend_count}")
PY
  local py_rc=$?
  set -e

  if [[ ${py_rc} -ne 0 ]] || grep -q '^ERROR:' "${OUT_ROOT}/.${profile_name}_metrics.tmp"; then
    local msg
    msg=$(cat "${OUT_ROOT}/.${profile_name}_metrics.tmp")
    echo "Profile ${profile_name} validation failed: ${msg}" >&2
    rc=1
  else
    IFS=',' read -r ring_ms recon_ms roi_ms products_written manifest_rows completed_required completed_optional missing_required missing_optional throttle_count suspend_count <"${OUT_ROOT}/.${profile_name}_metrics.tmp"
  fi

  echo "${profile_name},\"${stages}\",\"${obs_stage}\",${max_res},${runtime_ms},${ring_ms},${recon_ms},${roi_ms},${products_written},${manifest_rows},\"${completed_required}\",\"${completed_optional}\",\"${missing_required}\",\"${missing_optional}\",${throttle_count},${suspend_count},\"${out_root}\",${rc}" >>"${SUMMARY_CSV}"
  rm -f "${OUT_ROOT}/.${profile_name}_metrics.tmp"
  return ${rc}
}

overall_rc=0
for cfg in "${PROFILES[@]}"; do
  echo "Profiling ${cfg}"
  if ! run_one "${cfg}"; then
    overall_rc=1
  fi
done

echo "Progressive profiling complete:"
echo "  summary: ${SUMMARY_CSV}"
echo "  stages:  ${STAGE_AGG_CSV}"
exit ${overall_rc}
