#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build_wsl"
PI_BIN="${BUILD_DIR}/sgl_pi_flight"
OUT_ROOT="${REPO_ROOT}/out_profile"
SUMMARY_CSV="${OUT_ROOT}/progressive_profile_summary.csv"
STAGE_AGG_CSV="${OUT_ROOT}/progressive_stage_timings.csv"
RESOLUTION_SUMMARY_CSV="${OUT_ROOT}/progressive_resolution_summary.csv"
UNIFIED_CSV="${OUT_ROOT}/progressive_profile_unified.csv"

DEFAULT_PROFILES=(
  "profile_progressive_fast.json"
  "profile_progressive_balanced.json"
  "profile_progressive_full.json"
  "profile_progressive_stress.json"
)
if [[ $# -gt 0 ]]; then
  PROFILES=("$@")
else
  PROFILES=("${DEFAULT_PROFILES[@]}")
fi

if [[ ! -x "${PI_BIN}" ]]; then
  echo "ERROR: binary missing: ${PI_BIN}" >&2
  echo "Run scripts/build_all.sh first." >&2
  exit 1
fi

export OMP_DYNAMIC=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

mkdir -p "${OUT_ROOT}"
for cfg in "${PROFILES[@]}"; do
  cfg_path="${REPO_ROOT}/config/${cfg}"
  if [[ -f "${cfg_path}" ]]; then
    out_dir=$(python3 - "${cfg_path}" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg.get("out_dir","").strip())
PY
)
    if [[ -n "${out_dir}" ]]; then
      if [[ "${out_dir}" = /* ]]; then
        rm -rf "${out_dir}"
      else
        rm -rf "${REPO_ROOT:?}/${out_dir}"
      fi
    fi
  fi
done

echo "profile_name,stages,observations_by_stage,max_resolution,total_runtime_ms,ring_generation_ms,reconstruction_ms,roi_selection_ms,products_written,manifest_rows,completed_required_stages,completed_optional_stages,missing_required_outputs,missing_optional_outputs,scheduler_throttle_count,scheduler_suspend_count,baseline_system_cpu_busy_percent,baseline_system_mem_used_mb,avg_process_cpu_percent,peak_process_cpu_percent_inst,peak_process_rss_mb,peak_system_cpu_busy_percent,peak_system_mem_used_mb,output_dir,exit_status" >"${SUMMARY_CSV}"
echo "profile_name,stage_index,out_n,observations_used,new_observations_added,roi_count,base_runtime_ms,upscale_runtime_ms,refine_runtime_ms,roi_selection_ms,total_stage_runtime_ms,base_path,upscaled_path,refined_path" >"${STAGE_AGG_CSV}"
echo "profile_name,stage_index,out_n,observations_used,new_observations_added,roi_count,base_runtime_ms,upscale_runtime_ms,refine_runtime_ms,roi_selection_ms,total_stage_runtime_ms,base_exists,upscaled_exists,refined_exists,base_path,upscaled_path,refined_path" >"${RESOLUTION_SUMMARY_CSV}"
echo "group_name,row_type,profile_name,stage_index,out_n,metric_name,metric_value,unit,observations_used,new_observations_added,roi_count,base_runtime_ms,upscale_runtime_ms,refine_runtime_ms,roi_selection_ms,total_stage_runtime_ms,base_exists,upscaled_exists,refined_exists,stage_status,missing_outputs,output_dir,exit_status" >"${UNIFIED_CSV}"

run_one() {
  local cfg_name=$1
  local cfg_path="${REPO_ROOT}/config/${cfg_name}"
  local run_log="${OUT_ROOT}/$(basename "${cfg_name}" .json)_stdout.log"
  local time_log="${OUT_ROOT}/$(basename "${cfg_name}" .json)_time.log"
  local monitor_log="${OUT_ROOT}/$(basename "${cfg_name}" .json)_monitor.csv"
  local stage_tmp="${OUT_ROOT}/.$(basename "${cfg_name}" .json)_stages.tmp"
  local metrics_tmp="${OUT_ROOT}/.$(basename "${cfg_name}" .json)_metrics.tmp"
  local start_ns end_ns runtime_ms

  local baseline_cpu_busy baseline_mem_used_mb
  read -r baseline_cpu_busy baseline_mem_used_mb <<EOF2
$(python3 - <<'PY'
import time
def read_cpu():
    with open('/proc/stat','r',encoding='utf-8') as f:
        p=f.readline().split()[1:]
    v=[int(x) for x in p]
    idle=v[3]+(v[4] if len(v)>4 else 0)
    return idle,sum(v)
def mem_used_mb():
    vals={}
    with open('/proc/meminfo','r',encoding='utf-8') as f:
        for line in f:
            k,v=line.split(':',1)
            vals[k]=int(v.strip().split()[0])
    return max(0, vals.get('MemTotal',0)-vals.get('MemAvailable',0))/1024.0
i0,t0=read_cpu()
time.sleep(0.8)
i1,t1=read_cpu()
dt=max(1,t1-t0)
di=max(0,i1-i0)
busy=max(0.0,min(100.0,100.0*(dt-di)/dt))
print(f"{busy:.3f} {mem_used_mb():.3f}")
PY
)
EOF2

  local profile_name out_dir stages obs_stage max_res required_max optional_max
  read -r profile_name out_dir stages obs_stage max_res required_max optional_max <<EOF2
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
EOF2

  start_ns=$(date +%s%N)
  set +e
  /usr/bin/time -v -o "${time_log}" "${PI_BIN}" --config "${cfg_path}" >"${run_log}" 2>&1 &
  local sim_pid=$!

python3 - "${sim_pid}" "${monitor_log}" <<'PY' &
import os,sys,time
pid=int(sys.argv[1]); out=sys.argv[2]
ncpu=os.cpu_count() or 1

def read_proc_jiffies(p):
    with open(f"/proc/{p}/stat","r",encoding="utf-8") as f:
        data=f.read().split()
    # Use utime+stime only to avoid double-counting child times across subtree sums.
    return int(data[13]) + int(data[14])

def read_total_idle():
    with open('/proc/stat','r',encoding='utf-8') as f:
        vals=[int(x) for x in f.readline().split()[1:]]
    idle=vals[3]+(vals[4] if len(vals)>4 else 0)
    return idle,sum(vals)

def read_mem_used_mb():
    vals={}
    with open('/proc/meminfo','r',encoding='utf-8') as f:
        for line in f:
            k,v=line.split(':',1)
            vals[k]=int(v.strip().split()[0])
    return max(0, vals.get('MemTotal',0)-vals.get('MemAvailable',0))/1024.0

def read_rss_mb(p):
    with open(f"/proc/{p}/status","r",encoding='utf-8') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])/1024.0
    return 0.0

def ppid_of(p):
    try:
        with open(f"/proc/{p}/status","r",encoding="utf-8") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except Exception:
        return -1
    return -1

def descendants(root):
    roots={root}
    pids=[]
    for name in os.listdir("/proc"):
        if name.isdigit():
            pids.append(int(name))
    changed=True
    # Small fixed-point expansion over current /proc snapshot.
    while changed:
        changed=False
        for p in pids:
            if p in roots:
                continue
            pp=ppid_of(p)
            if pp in roots:
                roots.add(p)
                changed=True
    return roots

def subtree_stats(root):
    s=descendants(root)
    total_j=0
    total_rss=0.0
    for p in s:
        try:
            total_j += read_proc_jiffies(p)
            total_rss += read_rss_mb(p)
        except Exception:
            pass
    return total_j,total_rss

peak_proc_cpu=0.0
peak_proc_rss=0.0
peak_sys_cpu=0.0
peak_sys_mem=0.0

try:
    prev_p,_=subtree_stats(pid)
    prev_i,prev_t=read_total_idle()
except Exception:
    with open(out,'w',encoding='utf-8') as f:
        f.write('0,0,0,0\n')
    raise SystemExit(0)

while True:
    if not os.path.exists(f"/proc/{pid}"):
        break
    time.sleep(0.2)
    try:
        cur_p,cur_rss=subtree_stats(pid)
        cur_i,cur_t=read_total_idle()
        cur_mem=read_mem_used_mb()
    except Exception:
        break

    dp=max(0,cur_p-prev_p)
    dt=max(1,cur_t-prev_t)
    di=max(0,cur_i-prev_i)

    proc_cpu=min(100.0*ncpu, max(0.0, 100.0*ncpu*dp/dt))
    sys_cpu=max(0.0, min(100.0, 100.0*(dt-di)/dt))

    if proc_cpu>peak_proc_cpu: peak_proc_cpu=proc_cpu
    if cur_rss>peak_proc_rss: peak_proc_rss=cur_rss
    if sys_cpu>peak_sys_cpu: peak_sys_cpu=sys_cpu
    if cur_mem>peak_sys_mem: peak_sys_mem=cur_mem

    prev_p,prev_i,prev_t=cur_p,cur_i,cur_t

with open(out,'w',encoding='utf-8') as f:
    f.write(f"{peak_proc_cpu:.3f},{peak_proc_rss:.3f},{peak_sys_cpu:.3f},{peak_sys_mem:.3f}\n")
PY
  local mon_pid=$!

  wait ${sim_pid}
  local rc=$?
  wait ${mon_pid} >/dev/null 2>&1 || true
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
  local telemetry_csv="${mission}/telemetry_cycles.csv"

  local ring_ms recon_ms roi_ms products_written manifest_rows completed_required completed_optional missing_required missing_optional throttle_count suspend_count
  ring_ms=0; recon_ms=0; roi_ms=0; products_written=0; manifest_rows=0
  completed_required=""; completed_optional=""; missing_required=""; missing_optional=""
  throttle_count=0; suspend_count=0

  set +e
  python3 - "${manifest}" "${events}" "${stage_csv}" "${stages}" "${out_root}" "${profile_name}" "${obs_stage}" "${required_max}" "${optional_max}" "${telemetry_csv}" "${stage_tmp}" >"${metrics_tmp}" <<'PY'
import csv, os, sys
manifest, events, stage_csv, stages_s, out_root, profile_name, obs_stage, required_max_s, optional_max_s, telemetry_csv, stage_tmp = sys.argv[1:]
stages=[int(x) for x in stages_s.split('|') if x]
required_max=int(required_max_s)
optional_max=int(optional_max_s)
required_stages=[s for s in stages if s <= required_max]
optional_stages=[s for s in stages if required_max < s <= optional_max]

def die(msg):
    print(f"ERROR:{msg}")
    raise SystemExit(1)

def path_exists(p):
    if not p:
        return False, ""
    cands=[p, os.path.join(os.getcwd(),p), os.path.join(out_root,p)]
    for c in cands:
        if os.path.exists(c):
            return True,c
    return False,""

if not os.path.exists(manifest):
    die(f"missing manifest: {manifest}")
rows=list(csv.DictReader(open(manifest,newline='')))
if not rows:
    die('manifest empty')

manifest_rows=len(rows)
products_written=manifest_rows
found_base={}
found_upscaled={}
found_refined={}
for r in rows:
    kind=(r.get('kind') or '').strip().strip('"')
    p=(r.get('path') or '').strip().strip('"')
    if not p:
        die('empty path in manifest')
    ok,p_real=path_exists(p)
    if not ok:
        die(f'dangling manifest path: {p}')
    try:
        n=int((r.get('out_n') or '0').strip().strip('"'))
    except Exception:
        n=0
    if kind in ('recon_base','base') or (kind=='coarse' and n == (stages[0] if stages else n)):
        found_base[n]=p_real
    if kind in ('recon_upscaled','upscaled'):
        found_upscaled[n]=p_real
    if kind in ('recon_refined','refined'):
        found_refined[n]=p_real

base_n=stages[0] if stages else 0

def expected_outputs(n):
    if n == base_n:
        return [('base', found_base)]
    return [('upscaled', found_upscaled), ('refined', found_refined)]

missing_required=[]; completed_required=[]
missing_optional=[]; completed_optional=[]
for n in sorted(required_stages):
    miss=[]
    for label, found in expected_outputs(n):
        if n not in found: miss.append(label)
    if miss: missing_required.append(f"{n}:{'+'.join(miss)}")
    else: completed_required.append(str(n))
for n in sorted(optional_stages):
    miss=[]
    for label, found in expected_outputs(n):
        if n not in found: miss.append(label)
    if miss: missing_optional.append(f"{n}:{'+'.join(miss)}")
    else: completed_optional.append(str(n))
if missing_required:
    die('missing required outputs: ' + '|'.join(missing_required))

ring_ms=0.0
if os.path.exists(events):
    for e in csv.DictReader(open(events,newline='')):
        if (e.get('event_type') or '').strip().strip('"')=='ring_generation_timing':
            try: ring_ms += float((e.get('value') or '0').strip().strip('"'))
            except Exception: pass

stage_rows=[]
recon_ms=0.0; roi_ms=0.0
stage_map={}
if os.path.exists(stage_csv):
    for s in csv.DictReader(open(stage_csv,newline='')):
        try: n=int((s.get('out_n') or '0').strip().strip('"'))
        except Exception: continue
        stage_map[n]=s
        try:
            recon_ms += (
                float(s.get('base_runtime_ms') or s.get('coarse_runtime_ms') or '0') +
                float(s.get('upscale_runtime_ms') or '0') +
                float(s.get('refine_runtime_ms') or '0')
            )
        except Exception: pass
        try: roi_ms += float(s.get('roi_selection_ms','0'))
        except Exception: pass

for n in stages:
    s=stage_map.get(n,{})
    bp=(s.get('base_path') or '').strip().strip('"')
    up=(s.get('upscaled_path') or '').strip().strip('"')
    rp=(s.get('refined_path') or '').strip().strip('"')
    be,_=path_exists(bp)
    ue,_=path_exists(up)
    re,_=path_exists(rp)
    miss=[]
    if n == base_n:
        if not be: miss.append('base')
    else:
        if not ue: miss.append('upscaled')
        if not re: miss.append('refined')
    status='complete' if not miss else 'incomplete'
    stage_rows.append([
        s.get('stage_index',''), str(n), s.get('observations_used',''), s.get('new_observations_added',''), s.get('roi_count',''),
        s.get('base_runtime_ms') or s.get('coarse_runtime_ms',''), s.get('upscale_runtime_ms',''), s.get('refine_runtime_ms',''), s.get('roi_selection_ms',''), s.get('total_stage_runtime_ms',''),
        bp, up, rp, '1' if be else '0', '1' if ue else '0', '1' if re else '0', status, '|'.join(miss)
    ])

with open(stage_tmp,'w',newline='') as f:
    w=csv.writer(f)
    for r in stage_rows:
        w.writerow(r)

throttle_count=0; suspend_count=0
if os.path.exists(telemetry_csv):
    for t in csv.DictReader(open(telemetry_csv,newline='')):
        mode=(t.get('scheduler_mode') or '').strip().strip('"')
        if mode=='1': throttle_count += 1
        elif mode=='2': suspend_count += 1

print(','.join([
    str(ring_ms), str(recon_ms), str(roi_ms), str(products_written), str(manifest_rows),
    '|'.join(completed_required), '|'.join(completed_optional), '|'.join(missing_required), '|'.join(missing_optional),
    str(throttle_count), str(suspend_count)
]))
PY
  local py_rc=$?
  set -e

  if [[ ${py_rc} -ne 0 ]] || grep -q '^ERROR:' "${metrics_tmp}"; then
    echo "Profile ${profile_name} validation failed: $(cat "${metrics_tmp}")" >&2
    rc=1
  else
    IFS=',' read -r ring_ms recon_ms roi_ms products_written manifest_rows completed_required completed_optional missing_required missing_optional throttle_count suspend_count <"${metrics_tmp}"
  fi

  local avg_proc_cpu=0 peak_proc_cpu_inst=0 peak_proc_rss_mb=0 peak_sys_cpu=0 peak_sys_mem_mb=0
  if [[ -f "${time_log}" ]]; then
    avg_proc_cpu=$(awk -F: '/Percent of CPU this job got/{gsub(/^[ \t]+/,"",$2); gsub(/%/,"",$2); print $2+0}' "${time_log}" 2>/dev/null || echo 0)
  fi
  if [[ -f "${monitor_log}" ]]; then
    IFS=',' read -r peak_proc_cpu_inst peak_proc_rss_mb peak_sys_cpu peak_sys_mem_mb <"${monitor_log}" || true
  fi

  if [[ -f "${stage_tmp}" ]]; then
    while IFS=',' read -r stage_index out_n observations_used new_observations_added roi_count base_runtime_ms upscale_runtime_ms refine_runtime_ms roi_selection_ms_stage total_stage_runtime_ms base_path upscaled_path refined_path base_exists upscaled_exists refined_exists stage_status missing_outputs; do
      stage_status=${stage_status//$'\r'/}
      missing_outputs=${missing_outputs//$'\r'/}
      echo "${profile_name},${stage_index},${out_n},${observations_used},${new_observations_added},${roi_count},${base_runtime_ms},${upscale_runtime_ms},${refine_runtime_ms},${roi_selection_ms_stage},${total_stage_runtime_ms},\"${base_path}\",\"${upscaled_path}\",\"${refined_path}\"" >>"${STAGE_AGG_CSV}"
      echo "${profile_name},${stage_index},${out_n},${observations_used},${new_observations_added},${roi_count},${base_runtime_ms},${upscale_runtime_ms},${refine_runtime_ms},${roi_selection_ms_stage},${total_stage_runtime_ms},${base_exists},${upscaled_exists},${refined_exists},\"${base_path}\",\"${upscaled_path}\",\"${refined_path}\"" >>"${RESOLUTION_SUMMARY_CSV}"
      echo "stage_${out_n},stage,${profile_name},${stage_index},${out_n},stage_runtime_ms,${total_stage_runtime_ms},ms,${observations_used},${new_observations_added},${roi_count},${base_runtime_ms},${upscale_runtime_ms},${refine_runtime_ms},${roi_selection_ms_stage},${total_stage_runtime_ms},${base_exists},${upscaled_exists},${refined_exists},${stage_status},\"${missing_outputs}\",\"${out_root}\",${rc}" >>"${UNIFIED_CSV}"
    done <"${stage_tmp}"
  fi

  append_total_metric() {
    local metric_name=$1
    local metric_value=$2
    local unit=$3
    echo "total,total_metric,${profile_name},,,${metric_name},${metric_value},${unit},,,,,,,,,,,,complete,,\"${out_root}\",${rc}" >>"${UNIFIED_CSV}"
  }

  append_total_metric "total_runtime_ms" "${runtime_ms}" "ms"
  append_total_metric "ring_generation_ms" "${ring_ms}" "ms"
  append_total_metric "reconstruction_ms" "${recon_ms}" "ms"
  append_total_metric "roi_selection_ms" "${roi_ms}" "ms"
  append_total_metric "products_written" "${products_written}" "count"
  append_total_metric "manifest_rows" "${manifest_rows}" "count"
  append_total_metric "scheduler_throttle_count" "${throttle_count}" "count"
  append_total_metric "scheduler_suspend_count" "${suspend_count}" "count"
  append_total_metric "baseline_system_cpu_busy_percent" "${baseline_cpu_busy}" "percent"
  append_total_metric "baseline_system_mem_used_mb" "${baseline_mem_used_mb}" "mb"
  append_total_metric "avg_process_cpu_percent" "${avg_proc_cpu}" "percent"
  append_total_metric "peak_process_cpu_percent_inst" "${peak_proc_cpu_inst}" "percent"
  append_total_metric "peak_process_rss_mb" "${peak_proc_rss_mb}" "mb"
  append_total_metric "peak_system_cpu_busy_percent" "${peak_sys_cpu}" "percent"
  append_total_metric "peak_system_mem_used_mb" "${peak_sys_mem_mb}" "mb"

  echo "${profile_name},\"${stages}\",\"${obs_stage}\",${max_res},${runtime_ms},${ring_ms},${recon_ms},${roi_ms},${products_written},${manifest_rows},\"${completed_required}\",\"${completed_optional}\",\"${missing_required}\",\"${missing_optional}\",${throttle_count},${suspend_count},${baseline_cpu_busy},${baseline_mem_used_mb},${avg_proc_cpu},${peak_proc_cpu_inst},${peak_proc_rss_mb},${peak_sys_cpu},${peak_sys_mem_mb},\"${out_root}\",${rc}" >>"${SUMMARY_CSV}"

  "${SCRIPT_DIR}/package_run_outputs.sh" \
    --case-name "${profile_name}" \
    --source-out-root "${out_root}" \
    --config-path "${cfg_path}" \
    --profile-dir "${OUT_ROOT}" \
    --profile-prefix "$(basename "${cfg_name}" .json)"

  rm -f "${stage_tmp}" "${metrics_tmp}"
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
echo "  unified: ${UNIFIED_CSV}"
exit ${overall_rc}
