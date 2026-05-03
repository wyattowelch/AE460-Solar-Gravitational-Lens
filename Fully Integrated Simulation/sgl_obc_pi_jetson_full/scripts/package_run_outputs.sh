#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

usage() {
  cat <<USAGE
Usage:
  scripts/package_run_outputs.sh --case-name <name> --source-out-root <path> [--config-path <path>] [--profile-dir <path>] [--profile-prefix <prefix>]

Creates a portable run bundle under:
  ${REPO_ROOT}/outputs/<timestamp>_<case-name>/
USAGE
}

case_name=""
source_out_root=""
config_path=""
profile_dir=""
profile_prefix=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-name)
      case_name=${2:-}
      shift 2
      ;;
    --source-out-root)
      source_out_root=${2:-}
      shift 2
      ;;
    --config-path)
      config_path=${2:-}
      shift 2
      ;;
    --profile-dir)
      profile_dir=${2:-}
      shift 2
      ;;
    --profile-prefix)
      profile_prefix=${2:-}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${case_name}" || -z "${source_out_root}" ]]; then
  echo "ERROR: --case-name and --source-out-root are required" >&2
  usage >&2
  exit 2
fi

if [[ "${source_out_root}" != /* ]]; then
  source_out_root="${REPO_ROOT}/${source_out_root}"
fi
if [[ ! -d "${source_out_root}" ]]; then
  echo "ERROR: source output directory not found: ${source_out_root}" >&2
  exit 1
fi

if [[ -n "${profile_dir}" && "${profile_dir}" != /* ]]; then
  profile_dir="${REPO_ROOT}/${profile_dir}"
fi
if [[ -n "${config_path}" && "${config_path}" != /* ]]; then
  config_path="${REPO_ROOT}/${config_path}"
fi

timestamp=$(date +%Y%m%d_%H%M%S)
safe_case=$(printf '%s' "${case_name}" | tr -cs 'A-Za-z0-9._-' '_')
bundle_root="${REPO_ROOT}/outputs/${timestamp}_${safe_case}"

config_dir="${bundle_root}/config"
csv_dir="${bundle_root}/csv"
images_dir="${bundle_root}/images"
images_datasets_dir="${images_dir}/datasets"
images_products_dir="${images_dir}/products"
heavy_dir="${bundle_root}/heavy"
heavy_raw_ppm_dir="${heavy_dir}/raw_ppm"
heavy_ring_frames_dir="${heavy_dir}/ring_frames"
heavy_annulus_dir="${heavy_dir}/annulus"
heavy_datasets_dir="${heavy_dir}/datasets"
subsystems_dir="${bundle_root}/subsystems"

mkdir -p \
  "${config_dir}" \
  "${csv_dir}" \
  "${images_datasets_dir}" \
  "${images_products_dir}" \
  "${heavy_raw_ppm_dir}" \
  "${heavy_ring_frames_dir}" \
  "${heavy_annulus_dir}" \
  "${heavy_datasets_dir}" \
  "${subsystems_dir}"

mission_dir="${source_out_root}/mission_store"
if [[ -d "${mission_dir}" ]]; then
  find "${mission_dir}" -maxdepth 1 -type f -name '*.csv' -exec cp -f {} "${csv_dir}/" \;
fi

if [[ -d "${source_out_root}/datasets" ]]; then
  while IFS= read -r -d '' f; do
    rel=${f#"${source_out_root}/"}
    mkdir -p "${csv_dir}/$(dirname "${rel}")"
    cp -f "${f}" "${csv_dir}/${rel}"
  done < <(find "${source_out_root}/datasets" -type f -name '*.csv' -print0)
fi

if [[ -d "${source_out_root}" ]]; then
  find "${source_out_root}" -maxdepth 1 -type f -name '*.csv' -exec cp -f {} "${csv_dir}/" \; 2>/dev/null || true
fi

if [[ -d "${profile_dir}" ]]; then
  if [[ -n "${profile_prefix}" ]]; then
    cp -f "${profile_dir}/${profile_prefix}"*.csv "${csv_dir}/" 2>/dev/null || true
    cp -f "${profile_dir}/${profile_prefix}"*.log "${bundle_root}/" 2>/dev/null || true
  fi
  cp -f "${profile_dir}"/progressive_profile_*.csv "${csv_dir}/" 2>/dev/null || true
  cp -f "${profile_dir}"/progressive_stage_timings.csv "${csv_dir}/" 2>/dev/null || true
  cp -f "${profile_dir}"/progressive_resolution_summary.csv "${csv_dir}/" 2>/dev/null || true
  cp -f "${profile_dir}"/progressive_profile_unified.csv "${csv_dir}/" 2>/dev/null || true
fi

have_convert=0
if command -v convert >/dev/null 2>&1; then
  have_convert=1
fi

copy_png_or_ppm() {
  local src_ppm=$1
  local png_dest=$2
  if [[ ${have_convert} -eq 1 ]]; then
    mkdir -p "$(dirname "${png_dest}")"
    convert "${src_ppm}" "${png_dest}" || true
  fi
}

while IFS= read -r -d '' ppm_path; do
  rel_path=${ppm_path#"${source_out_root}/"}
  base_name=$(basename "${ppm_path}" .ppm)

  mkdir -p "${heavy_raw_ppm_dir}/$(dirname "${rel_path}")"
  cp -f "${ppm_path}" "${heavy_raw_ppm_dir}/${rel_path}"

  if [[ "${rel_path}" == products/* ]]; then
    copy_png_or_ppm "${ppm_path}" "${images_products_dir}/${base_name}.png"
  elif [[ "${rel_path}" == datasets/* ]]; then
    dataset_rel=${rel_path#datasets/}
    copy_png_or_ppm "${ppm_path}" "${images_datasets_dir}/${dataset_rel%.ppm}.png"
  else
    copy_png_or_ppm "${ppm_path}" "${images_dir}/${rel_path%.ppm}.png"
  fi

  bn=$(basename "${ppm_path}")
  if [[ "${bn}" == ring_frame_*.ppm || "${bn}" == full_ring_*.ppm || "${bn}" == ring_preview*.ppm ]]; then
    cp -f "${ppm_path}" "${heavy_ring_frames_dir}/${bn}" || true
  fi
done < <(find "${source_out_root}" -type f -name '*.ppm' -print0)

while IFS= read -r -d '' bin_path; do
  rel_path=${bin_path#"${source_out_root}/"}
  if [[ "${rel_path}" == *annulus* || "${rel_path}" == *.bin ]]; then
    mkdir -p "${heavy_annulus_dir}/$(dirname "${rel_path}")"
    cp -f "${bin_path}" "${heavy_annulus_dir}/${rel_path}"
  fi
done < <(find "${source_out_root}" -type f -name '*.bin' -print0)

if [[ -d "${source_out_root}/datasets" ]]; then
  cp -a "${source_out_root}/datasets" "${heavy_datasets_dir}/" 2>/dev/null || true
fi

if [[ ${have_convert} -eq 0 ]]; then
  echo "WARNING: ImageMagick 'convert' not found. PNG conversion skipped; only heavy/raw_ppm copied." >&2
fi

if [[ -n "${config_path}" && -f "${config_path}" ]]; then
  cp -f "${config_path}" "${config_dir}/original_config.json"
fi

python3 - "${bundle_root}" "${source_out_root}" "${case_name}" "${config_path}" "${REPO_ROOT}" "${RUN_DISK_FREE_BEFORE_GB:-}" "${RUN_DISK_FREE_AFTER_GB:-}" "${RUN_OUTPUT_ESTIMATE_GB:-}" <<'PY'
import csv
import json
import os
import pathlib
import shutil
import sys
from datetime import datetime, timezone

bundle_root = pathlib.Path(sys.argv[1])
source_out_root = pathlib.Path(sys.argv[2])
case_name = sys.argv[3]
config_path = pathlib.Path(sys.argv[4]) if sys.argv[4] else None
repo_root = pathlib.Path(sys.argv[5])
free_before = sys.argv[6]
free_after = sys.argv[7]
estimate = sys.argv[8]

csv_dir = bundle_root / "csv"
subsystems_dir = bundle_root / "subsystems"
config_dir = bundle_root / "config"

telemetry = csv_dir / "telemetry_cycles.csv"
events = csv_dir / "events.csv"
manifest_csv = csv_dir / "products_manifest.csv"

if config_path and config_path.exists():
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
else:
    cfg = {}

effective = dict(cfg)
effective["_resolved"] = {
    "source_out_root": str(source_out_root.resolve()),
    "bundle_root": str(bundle_root.resolve()),
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
(config_dir / "effective_config.json").write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8")

subsystem_fields = {
    "adcs": ["cycle","adcs_mode","adcs_power_w","wheel_power_w","truth_pointing_err_deg","est_pointing_err_deg","tracker_conf","tracker_valid","tracked_stars"],
    "eps": ["cycle","source_w","reserve_w","noncompute_w","compute_budget_w","total_bus_load_w","scheduler_mode"],
    "thermal": ["cycle","thermal_mode","thermal_power_w","heater_active","thermal_temp_c"],
    "propulsion": ["cycle","propulsion_mode","propulsion_active","propulsion_power_w","propulsion_thrust_n"],
    "comms": ["cycle","comms_mode","comms_power_w","comms_backlog_bits"],
    "payload": ["cycle","payload_mode","payload_active","payload_power_w","dataset_ready","dataset_id","dataset_count","acquisition_stage","camera_mode","camera_frame_ready","alignment_valid","alignment_score","raw_capture_path","rectified_image_path","preconditioned_source_path"],
    "scheduler": ["cycle","scheduler_mode","compute_budget_w","jetson_allow_w","noncompute_w","source_w","total_bus_load_w"],
    "jetson": ["cycle","jetson_mode","jetson_job_type","jetson_power_w","processing_queue","roi_count","active_stage","active_stage_n"],
    "obc": ["cycle","pi_power_w","dt_s"],
}

if telemetry.exists():
    with telemetry.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = f.seek(0) or []
    if rows:
        telem_fields = set(rows[0].keys())
        for name, wanted in subsystem_fields.items():
            out_dir = subsystems_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            use = [c for c in wanted if c in telem_fields]
            if not use:
                continue
            with (out_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as w:
                writer = csv.DictWriter(w, fieldnames=use)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in use})

if events.exists():
    keywords = {
        "adcs": ["adcs", "tracker", "pointing", "wheel"],
        "eps": ["budget", "power", "source", "bus"],
        "thermal": ["thermal", "heater"],
        "propulsion": ["propulsion", "burn", "thrust"],
        "comms": ["comms", "downlink", "queue"],
        "payload": ["payload", "dataset", "camera", "alignment", "ring"],
        "scheduler": ["scheduler", "budget", "throttle", "suspend"],
        "jetson": ["jetson", "coarse", "refine", "reconstruction"],
        "obc": ["fdir", "safe", "warning", "obc"],
    }
    with events.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for name, kws in keywords.items():
        out_dir = subsystems_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = []
        for r in rows:
            et = (r.get("event_type") or "").lower()
            msg = (r.get("message") or "").lower()
            if any(k in et or k in msg for k in kws):
                keep.append(r)
        if keep:
            with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as w:
                writer = csv.DictWriter(w, fieldnames=list(keep[0].keys()))
                writer.writeheader()
                writer.writerows(keep)

def compute_requested_stages(cfg_obj):
    base = int(cfg_obj.get("progressive_base_N", 128))
    max_n = int(cfg_obj.get("progressive_max_N", 1024))
    scale = int(cfg_obj.get("progressive_scale", 2))
    max_stages = int(cfg_obj.get("progressive_max_stages", 8))
    stages = []
    n = base
    for _ in range(max_stages):
        if n > max_n:
            break
        stages.append(n)
        if scale <= 1:
            break
        n *= scale
    return stages

requested_stages = compute_requested_stages(cfg)
base_stage_n = requested_stages[0] if requested_stages else 128

def stage_expected_kinds(stage_n):
    return ["base"] if stage_n == base_stage_n else ["upscaled", "refined"]

required_max = int(cfg.get("required_max_resolution", requested_stages[-1] if requested_stages else 0))
optional_max = int(cfg.get("optional_max_resolution", required_max))
if optional_max < required_max:
    optional_max = required_max

present = {}
dataset_id = "dataset_0"
manifest_rows = []
if manifest_csv.exists():
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    for r in manifest_rows:
        ds = (r.get("dataset_id") or "").strip()
        if ds:
            dataset_id = ds
        try:
            out_n = int(r.get("out_n", "0") or "0")
        except ValueError:
            continue
        kind_raw = (r.get("kind") or "").strip()
        k = ""
        if kind_raw == "recon_base":
            k = "base"
        elif kind_raw == "recon_upscaled":
            k = "upscaled"
        elif kind_raw == "recon_refined":
            k = "refined"
        if not k:
            continue
        present.setdefault(out_n, set()).add(k)

completed_stages = []
missing_required = []
missing_optional = []
missing_required_rows = []
for out_n in requested_stages:
    expected = stage_expected_kinds(out_n)
    got = present.get(out_n, set())
    miss = [k for k in expected if k not in got]
    if not miss:
        completed_stages.append(out_n)
        continue
    entry = f"{out_n}:{'+'.join(miss)}"
    if out_n <= required_max:
        missing_required.append(entry)
        for k in miss:
            missing_required_rows.append({
                "cycle": "",
                "dataset_id": dataset_id,
                "stage": "",
                "kind": f"missing_required_{k}",
                "out_n": str(out_n),
                "path": "",
                "bytes": "0",
                "roi_count": "0",
                "roi_score_mean": "0",
                "status": "required output missing at packaging",
            })
    elif out_n <= optional_max:
        missing_optional.append(entry)

# Mark missing required outputs in packaged manifest for explicit diagnostics.
if missing_required_rows:
    fieldnames = ["cycle","dataset_id","stage","kind","out_n","path","bytes","roi_count","roi_score_mean","status"]
    with manifest_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for row in missing_required_rows:
            w.writerow(row)

if not manifest_rows:
    run_completion_status = "failed"
    completion_reason = "products_manifest.csv missing or empty"
elif missing_required:
    run_completion_status = "partial"
    completion_reason = "missing required outputs"
elif missing_optional:
    run_completion_status = "partial"
    completion_reason = "missing optional outputs"
else:
    run_completion_status = "complete"
    completion_reason = "all requested outputs completed"

size_total = shutil.disk_usage(bundle_root).total
bundle_bytes = 0
for p in bundle_root.rglob("*"):
    if p.is_file():
        try:
            bundle_bytes += p.stat().st_size
        except OSError:
            pass

def dir_bytes(p):
    if not p.exists():
        return 0
    n = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                n += f.stat().st_size
            except OSError:
                pass
    return n

meta = {
    "case_name": case_name,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_out_root": str(source_out_root.resolve()),
    "bundle_root": str(bundle_root.resolve()),
    "disk_free_before_gb": float(free_before) if free_before else None,
    "disk_free_after_gb": float(free_after) if free_after else None,
    "estimated_output_gb": float(estimate) if estimate else None,
    "sizes_bytes": {
        "bundle_total": bundle_bytes,
        "csv": dir_bytes(bundle_root / "csv"),
        "images": dir_bytes(bundle_root / "images"),
        "heavy": dir_bytes(bundle_root / "heavy"),
        "heavy_raw_ppm": dir_bytes(bundle_root / "heavy" / "raw_ppm"),
        "heavy_ring_frames": dir_bytes(bundle_root / "heavy" / "ring_frames"),
        "heavy_annulus": dir_bytes(bundle_root / "heavy" / "annulus"),
    },
    "config_path": str(config_path.resolve()) if config_path and config_path.exists() else None,
    "requested_stages": requested_stages,
    "completed_stages": completed_stages,
    "missing_required_outputs": missing_required,
    "missing_optional_outputs": missing_optional,
    "run_completion_status": run_completion_status,
    "completion_reason": completion_reason,
}
(bundle_root / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
PY

cat > "${bundle_root}/README.txt" <<INFO
Case: ${case_name}
Source out root: ${source_out_root}
Created: $(date '+%F %T')

Contents:
- config/: original + effective config snapshot
- csv/: telemetry/events/manifest/downlink/profile CSVs
- images/: lightweight PNG products for demo inspection
- heavy/: large artifacts (raw PPM, ring frames, annulus, dataset dumps)
- subsystems/: filtered subsystem telemetry/event views
- run_metadata.json: run and disk/size metadata
INFO

ln -sfn "${bundle_root}" "${REPO_ROOT}/outputs/latest"

bundle_total=$(du -sh "${bundle_root}" 2>/dev/null | awk '{print $1}')
images_total=$(du -sh "${images_dir}" 2>/dev/null | awk '{print $1}')
heavy_total=$(du -sh "${heavy_dir}" 2>/dev/null | awk '{print $1}')
csv_total=$(du -sh "${csv_dir}" 2>/dev/null | awk '{print $1}')

echo "Packaged outputs: ${bundle_root}"
echo "Latest link:      ${REPO_ROOT}/outputs/latest"
echo "Bundle size:"
echo "  total:   ${bundle_total}"
echo "  images:  ${images_total}"
echo "  heavy:   ${heavy_total}"
echo "  csv:     ${csv_total}"

if [[ -n "${config_path}" && -f "${config_path}" ]]; then
  retention_args=$(python3 - "${config_path}" "${REPO_ROOT}/config/output_retention_defaults.json" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1], 'r', encoding='utf-8'))
defaults={}
if len(sys.argv) > 2:
    try:
        defaults=json.load(open(sys.argv[2], 'r', encoding='utf-8'))
    except Exception:
        defaults={}
for k,v in defaults.items():
    cfg.setdefault(k, v)
if not bool(cfg.get('outputs_retention_enabled', True)):
    print('DISABLED')
    raise SystemExit(0)
args=[]
args.extend(['--keep-lightweight-runs', str(int(cfg.get('outputs_keep_lightweight_runs', 10)))])
args.extend(['--keep-full-runs', str(int(cfg.get('outputs_keep_full_runs', 3)))])
max_total=float(cfg.get('outputs_max_total_gb', 0.0))
if max_total > 0:
    args.extend(['--max-total-gb', str(max_total)])
args.append('--prune-heavy')
if bool(cfg.get('outputs_prune_raw_ppm', True)):
    args.append('--prune-raw-ppm')
if bool(cfg.get('outputs_prune_ring_frames', True)):
    args.append('--prune-ring-frames')
if bool(cfg.get('outputs_prune_annulus_dumps', True)):
    args.append('--prune-annulus')
if bool(cfg.get('outputs_preserve_marked_runs', True)):
    args.append('--preserve-marked')
if bool(cfg.get('outputs_retention_include_out_profile', False)):
    args.append('--include-out-profile')
if bool(cfg.get('outputs_retention_include_working_outs', False)):
    args.append('--include-working-outs')
print(' '.join(args))
PY
)

  if [[ "${retention_args}" == "DISABLED" ]]; then
    echo "Automatic retention: disabled by config"
  else
    if [[ "${SGL_SKIP_AUTO_RETENTION:-0}" == "1" || "${SGL_RETENTION_DRY_RUN_ONLY:-0}" == "1" ]]; then
      echo "Automatic retention: dry-run only (SGL_SKIP_AUTO_RETENTION or SGL_RETENTION_DRY_RUN_ONLY set)"
      # shellcheck disable=SC2086
      "${SCRIPT_DIR}/cleanup_packaged_outputs.sh" --dry-run ${retention_args}
    else
      echo "Automatic retention: applying policy"
      # shellcheck disable=SC2086
      "${SCRIPT_DIR}/cleanup_packaged_outputs.sh" --delete ${retention_args}
    fi
  fi
fi
