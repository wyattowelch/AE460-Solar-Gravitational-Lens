#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUTPUTS_DIR="${REPO_ROOT}/outputs"
OUT_PROFILE_DIR="${REPO_ROOT}/out_profile"
ORIGINAL_ARGS=("$@")

DRY_RUN=1
DO_DELETE=0
KEEP_LIGHTWEIGHT_RUNS=10
KEEP_FULL_RUNS=3
MAX_TOTAL_GB=0
PRUNE_HEAVY=0
PRUNE_RAW_PPM=0
PRUNE_RING_FRAMES=0
PRUNE_ANNULUS=0
PRESERVE_MARKED=0
INCLUDE_OUT_PROFILE=0
INCLUDE_WORKING_OUTS=0

usage() {
  cat <<USAGE
Usage:
  scripts/cleanup_packaged_outputs.sh [options]

Options:
  --dry-run                       Preview only (default).
  --delete                        Apply cleanup.
  --keep-lightweight-runs N       Keep newest N runs as lightweight records (default: 10).
  --keep-full-runs N              Keep newest N runs with full/heavy data (default: 3).
  --max-total-gb N                Optional cap for outputs/ total size.
  --prune-heavy                   Prune heavy artifacts on runs outside full-retention window.
  --prune-raw-ppm                 Prune raw PPM data when pruning heavy artifacts.
  --prune-ring-frames             Prune ring frame artifacts when pruning heavy artifacts.
  --prune-annulus                 Prune annulus binary artifacts when pruning heavy artifacts.
  --preserve-marked               Preserve runs containing .keep or KEEP_RUN.
  --include-out-profile           Include out_profile/ in size report and optional heavy-prune pass.
  --include-working-outs          Include out_* working dirs in size report and optional heavy-prune pass.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      DO_DELETE=0
      shift
      ;;
    --delete)
      DO_DELETE=1
      DRY_RUN=0
      shift
      ;;
    --keep-lightweight-runs)
      KEEP_LIGHTWEIGHT_RUNS=${2:-}
      shift 2
      ;;
    --keep-full-runs)
      KEEP_FULL_RUNS=${2:-}
      shift 2
      ;;
    --max-total-gb)
      MAX_TOTAL_GB=${2:-}
      shift 2
      ;;
    --prune-heavy)
      PRUNE_HEAVY=1
      shift
      ;;
    --prune-raw-ppm)
      PRUNE_RAW_PPM=1
      shift
      ;;
    --prune-ring-frames)
      PRUNE_RING_FRAMES=1
      shift
      ;;
    --prune-annulus)
      PRUNE_ANNULUS=1
      shift
      ;;
    --preserve-marked)
      PRESERVE_MARKED=1
      shift
      ;;
    --include-out-profile)
      INCLUDE_OUT_PROFILE=1
      shift
      ;;
    --include-working-outs)
      INCLUDE_WORKING_OUTS=1
      shift
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

is_nonneg_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_nonneg_int "${KEEP_LIGHTWEIGHT_RUNS}"; then
  echo "ERROR: --keep-lightweight-runs must be an integer" >&2
  exit 2
fi
if ! is_nonneg_int "${KEEP_FULL_RUNS}"; then
  echo "ERROR: --keep-full-runs must be an integer" >&2
  exit 2
fi

if (( KEEP_FULL_RUNS > KEEP_LIGHTWEIGHT_RUNS )); then
  KEEP_FULL_RUNS=${KEEP_LIGHTWEIGHT_RUNS}
fi

if [[ ! -d "${OUTPUTS_DIR}" ]]; then
  echo "No packaged outputs directory found: ${OUTPUTS_DIR}"
  exit 0
fi

latest_target=""
if [[ -L "${OUTPUTS_DIR}/latest" ]]; then
  latest_target=$(readlink -f "${OUTPUTS_DIR}/latest" || true)
fi

human_bytes() {
  python3 - "$1" <<'PY'
import sys
n=float(sys.argv[1] or 0)
for u in ['B','KB','MB','GB','TB']:
    if n < 1024 or u == 'TB':
        print(f"{n:.1f}{u}")
        break
    n /= 1024
PY
}

bytes_of_path() {
  local p=$1
  if [[ ! -e "${p}" ]]; then
    echo 0
    return
  fi
  du -sb "${p}" 2>/dev/null | awk '{print $1+0}'
}

has_preserve_marker() {
  local run=$1
  [[ -e "${run}/.keep" || -e "${run}/KEEP_RUN" ]]
}

run_completion_status() {
  local run=$1
  local meta="${run}/run_metadata.json"
  if [[ ! -f "${meta}" ]]; then
    echo "unknown"
    return
  fi
  python3 - "${meta}" <<'PY'
import json, sys
try:
    obj=json.load(open(sys.argv[1], 'r', encoding='utf-8'))
    print(str(obj.get("run_completion_status", "unknown")).strip().lower() or "unknown")
except Exception:
    print("unknown")
PY
}

collect_keep_image_candidates() {
  local run=$1
  local out=$2
  : > "${out}"

  find "${run}/images/datasets" -type f \( -name 'preconditioned_source.png' -o -name 'source_overlay.png' \) -print 2>/dev/null >> "${out}" || true

  if [[ -f "${run}/images/products/reconstruction_contact_sheet.png" ]]; then
    printf '%s\n' "${run}/images/products/reconstruction_contact_sheet.png" >> "${out}"
  fi

  local latest_refined
  latest_refined=$(find "${run}/images/products" -maxdepth 1 -type f -name '*_refined_*.png' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "${latest_refined}" ]]; then
    printf '%s\n' "${latest_refined}" >> "${out}"
  fi

  sort -u "${out}" -o "${out}"
}

prune_run_heavy_bytes() {
  local run=$1
  local keep_list=$2
  local sum=0

  for d in "${run}/heavy" "${run}/raw_ppm"; do
    if [[ -d "${d}" ]]; then
      b=$(bytes_of_path "${d}")
      sum=$((sum + b))
    fi
  done

  if (( PRUNE_RING_FRAMES )); then
    while IFS= read -r -d '' f; do
      b=$(bytes_of_path "${f}")
      sum=$((sum + b))
    done < <(find "${run}" -type f \( -name 'ring_frame_*.ppm' -o -name 'full_ring_*.ppm' -o -name 'ring_preview_*.ppm' \) -print0 2>/dev/null)
  fi

  if (( PRUNE_ANNULUS )); then
    while IFS= read -r -d '' f; do
      b=$(bytes_of_path "${f}")
      sum=$((sum + b))
    done < <(find "${run}" -type f \( -name '*annulus*' -o -name '*.bin' \) -print0 2>/dev/null)
  fi

  if [[ -d "${run}/images" ]]; then
    declare -A keep=()
    while IFS= read -r p; do
      [[ -n "${p}" ]] || continue
      keep["$(readlink -f "${p}")"]=1
    done < "${keep_list}"

    while IFS= read -r -d '' f; do
      rf=$(readlink -f "${f}")
      if [[ -n "${keep[${rf}]:-}" ]]; then
        continue
      fi
      b=$(bytes_of_path "${f}")
      sum=$((sum + b))
    done < <(find "${run}/images" -type f -print0 2>/dev/null)
  fi

  if (( PRUNE_RAW_PPM )); then
    while IFS= read -r -d '' ppm; do
      png="${ppm%.ppm}.png"
      if [[ -f "${png}" ]]; then
        b=$(bytes_of_path "${ppm}")
        sum=$((sum + b))
      fi
    done < <(find "${run}" -type f -name '*.ppm' -print0 2>/dev/null)
  fi

  echo "${sum}"
}

prune_run_heavy_apply() {
  local run=$1
  local keep_list=$2
  local reason=$3

  local removed_bytes=0

  delete_path() {
    local p=$1
    if [[ -e "${p}" ]]; then
      b=$(bytes_of_path "${p}")
      removed_bytes=$((removed_bytes + b))
      rm -rf -- "${p}"
    fi
  }

  delete_path "${run}/heavy"
  delete_path "${run}/raw_ppm"

  if (( PRUNE_RING_FRAMES )); then
    while IFS= read -r -d '' f; do
      b=$(bytes_of_path "${f}")
      removed_bytes=$((removed_bytes + b))
      rm -f -- "${f}"
    done < <(find "${run}" -type f \( -name 'ring_frame_*.ppm' -o -name 'full_ring_*.ppm' -o -name 'ring_preview_*.ppm' \) -print0 2>/dev/null)
  fi

  if (( PRUNE_ANNULUS )); then
    while IFS= read -r -d '' f; do
      b=$(bytes_of_path "${f}")
      removed_bytes=$((removed_bytes + b))
      rm -f -- "${f}"
    done < <(find "${run}" -type f \( -name '*annulus*' -o -name '*.bin' \) -print0 2>/dev/null)
  fi

  if [[ -d "${run}/images" ]]; then
    declare -A keep=()
    while IFS= read -r p; do
      [[ -n "${p}" ]] || continue
      keep["$(readlink -f "${p}")"]=1
    done < "${keep_list}"

    while IFS= read -r -d '' f; do
      rf=$(readlink -f "${f}")
      if [[ -n "${keep[${rf}]:-}" ]]; then
        continue
      fi
      b=$(bytes_of_path "${f}")
      removed_bytes=$((removed_bytes + b))
      rm -f -- "${f}"
    done < <(find "${run}/images" -type f -print0 2>/dev/null)
    find "${run}/images" -depth -type d -empty -delete 2>/dev/null || true
  fi

  if (( PRUNE_RAW_PPM )); then
    while IFS= read -r -d '' ppm; do
      png="${ppm%.ppm}.png"
      if [[ -f "${png}" ]]; then
        b=$(bytes_of_path "${ppm}")
        removed_bytes=$((removed_bytes + b))
        rm -f -- "${ppm}"
      fi
    done < <(find "${run}" -type f -name '*.ppm' -print0 2>/dev/null)
  fi

  mkdir -p "${run}/heavy"
  cat > "${run}/heavy/PRUNED.txt" <<TXT
pruned_at: $(date '+%F %T')
reason: ${reason}
removed_bytes: ${removed_bytes}
keep_lightweight_runs: ${KEEP_LIGHTWEIGHT_RUNS}
keep_full_runs: ${KEEP_FULL_RUNS}
TXT
}

collect_out_profile_candidates() {
  local profile_dir=$1
  local list_file=$2
  : > "${list_file}"
  local names=(datasets raw_ppm ring_frames annulus heavy jetson_scratch)
  for n in "${names[@]}"; do
    local p="${profile_dir}/${n}"
    if [[ -e "${p}" ]]; then
      printf '%s\n' "${p}" >> "${list_file}"
    fi
  done
  sort -u "${list_file}" -o "${list_file}"
}

out_profile_candidate_bytes() {
  local list_file=$1
  local sum=0
  while IFS= read -r p; do
    [[ -n "${p}" ]] || continue
    b=$(bytes_of_path "${p}")
    sum=$((sum + b))
  done < "${list_file}"
  echo "${sum}"
}

apply_out_profile_prune() {
  local profile_dir=$1
  local list_file=$2
  local removed_bytes=0
  while IFS= read -r p; do
    [[ -n "${p}" && -e "${p}" ]] || continue
    b=$(bytes_of_path "${p}")
    removed_bytes=$((removed_bytes + b))
    rm -rf -- "${p}"
  done < "${list_file}"

  local cmd_text
  cmd_text="./scripts/cleanup_packaged_outputs.sh"
  for a in "${ORIGINAL_ARGS[@]}"; do
    cmd_text="${cmd_text} ${a}"
  done
  cat > "${profile_dir}/PRUNED.txt" <<TXT
pruned_at: $(date '+%F %T')
reason: out_profile heavy artifact cleanup
removed_bytes: ${removed_bytes}
command: ${cmd_text}
preserved_lightweight: *.csv, mission_store/*.csv, profile summaries, stage timings, manifests, configs/metadata, products/
TXT
}

declare -a report_roots=("${OUTPUTS_DIR}")
if (( INCLUDE_OUT_PROFILE )); then
  report_roots+=("${REPO_ROOT}/out_profile")
fi
if (( INCLUDE_WORKING_OUTS )); then
  while IFS= read -r d; do
    report_roots+=("${d}")
  done < <(find "${REPO_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'out*' ! -path "${OUTPUTS_DIR}" ! -path "${REPO_ROOT}/out_profile" | sort)
fi

echo "Output roots summary:"
for r in "${report_roots[@]}"; do
  if [[ -d "${r}" ]]; then
    echo "  $(du -sh "${r}" 2>/dev/null | awk '{print $1}')  ${r}"
  fi
done
echo

echo "Largest packaged run directories:"
find "${OUTPUTS_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name latest -print0 | while IFS= read -r -d '' d; do
  size=$(du -sh "${d}" 2>/dev/null | awk '{print $1}')
  printf '%s\t%s\n' "${size}" "${d}"
done | sort -h | tail -n 15 || true
echo

mapfile -t run_dirs < <(find "${OUTPUTS_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name latest -printf '%T@|%p\n' | sort -nr)

if ((${#run_dirs[@]} == 0)); then
  echo "No packaged run directories under ${OUTPUTS_DIR}."
  exit 0
fi

declare -a delete_runs=()
declare -a prune_runs=()
declare -a full_runs=()
declare -a light_runs=()
declare -a preserved_runs=()

auto_rank=0
full_rank=0
for line in "${run_dirs[@]}"; do
  run=${line#*|}
  run_real=$(readlink -f "${run}")
  status=$(run_completion_status "${run}")

  if [[ -n "${latest_target}" && "${run_real}" == "${latest_target}" ]]; then
    if [[ "${status}" == "complete" ]]; then
      full_runs+=("${run}")
      full_rank=$((full_rank + 1))
    else
      light_runs+=("${run}")
      prune_runs+=("${run}")
    fi
    continue
  fi

  if (( PRESERVE_MARKED )) && has_preserve_marker "${run}"; then
    preserved_runs+=("${run}")
    continue
  fi

  auto_rank=$((auto_rank + 1))
  if [[ "${status}" == "complete" && ${full_rank} -lt ${KEEP_FULL_RUNS} ]]; then
    full_runs+=("${run}")
    full_rank=$((full_rank + 1))
  elif (( auto_rank <= KEEP_LIGHTWEIGHT_RUNS )); then
    light_runs+=("${run}")
    prune_runs+=("${run}")
  else
    delete_runs+=("${run}")
  fi
done

would_delete_bytes=0
for d in "${delete_runs[@]}"; do
  b=$(bytes_of_path "${d}")
  would_delete_bytes=$((would_delete_bytes + b))
done

would_prune_bytes=0
declare -a prune_keep_lists=()
if (( PRUNE_HEAVY )); then
  for run in "${prune_runs[@]}"; do
    tmp_keep=$(mktemp)
    prune_keep_lists+=("${tmp_keep}")
    collect_keep_image_candidates "${run}" "${tmp_keep}"
    b=$(prune_run_heavy_bytes "${run}" "${tmp_keep}")
    would_prune_bytes=$((would_prune_bytes + b))
  done
fi

out_profile_prune_bytes=0
declare -a out_profile_prune_profiles=()
declare -a out_profile_prune_lists=()
declare -a out_profile_preserved=()
if (( INCLUDE_OUT_PROFILE )) && (( PRUNE_HEAVY )) && [[ -d "${OUT_PROFILE_DIR}" ]]; then
  while IFS= read -r profile_dir; do
    [[ -d "${profile_dir}" ]] || continue
    if (( PRESERVE_MARKED )) && has_preserve_marker "${profile_dir}"; then
      out_profile_preserved+=("${profile_dir}")
      continue
    fi
    tmp_list=$(mktemp)
    collect_out_profile_candidates "${profile_dir}" "${tmp_list}"
    b=$(out_profile_candidate_bytes "${tmp_list}")
    if (( b > 0 )); then
      out_profile_prune_profiles+=("${profile_dir}")
      out_profile_prune_lists+=("${tmp_list}")
      out_profile_prune_bytes=$((out_profile_prune_bytes + b))
    else
      rm -f "${tmp_list}"
    fi
  done < <(find "${OUT_PROFILE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

echo "Retention policy summary:"
echo "  keep lightweight runs: ${KEEP_LIGHTWEIGHT_RUNS}"
echo "  keep full runs:        ${KEEP_FULL_RUNS}"
echo "  prune heavy:           ${PRUNE_HEAVY}"
echo "  preserve markers:      ${PRESERVE_MARKED}"
echo

echo "Classification:"
echo "  full runs kept:        ${#full_runs[@]}"
echo "  lightweight runs kept: ${#light_runs[@]}"
echo "  runs to prune heavy:   ${#prune_runs[@]}"
echo "  runs to delete:        ${#delete_runs[@]}"
echo "  preserved runs:        ${#preserved_runs[@]}"
if (( INCLUDE_OUT_PROFILE )); then
  echo "  out_profile prune dirs:${#out_profile_prune_profiles[@]}"
  echo "  out_profile preserved: ${#out_profile_preserved[@]}"
fi
echo

echo "Estimated cleanup impact:"
echo "  delete entire runs:  $(human_bytes "${would_delete_bytes}")"
echo "  prune heavy data:    $(human_bytes "${would_prune_bytes}")"
if (( INCLUDE_OUT_PROFILE )); then
  echo "  prune out_profile:   $(human_bytes "${out_profile_prune_bytes}")"
fi
echo "  total freed estimate: $(human_bytes "$((would_delete_bytes + would_prune_bytes + out_profile_prune_bytes))")"
echo

if ((${#preserved_runs[@]})); then
  echo "Preserved runs (.keep/KEEP_RUN):"
  printf '  %s\n' "${preserved_runs[@]}"
  echo
fi

if ((${#delete_runs[@]})); then
  echo "Runs to delete entirely (> keep-lightweight window):"
  printf '  %s\n' "${delete_runs[@]}"
  echo
fi

if ((${#prune_runs[@]})); then
  echo "Runs to prune to lightweight-only:"
  printf '  %s\n' "${prune_runs[@]}"
  echo
fi

if (( INCLUDE_OUT_PROFILE )) && ((${#out_profile_preserved[@]})); then
  echo "Preserved out_profile directories (.keep/KEEP_RUN):"
  printf '  %s\n' "${out_profile_preserved[@]}"
  echo
fi

if (( INCLUDE_OUT_PROFILE )) && ((${#out_profile_prune_profiles[@]})); then
  echo "out_profile heavy prune candidates:"
  idx=0
  for pdir in "${out_profile_prune_profiles[@]}"; do
    plist=${out_profile_prune_lists[$idx]}
    idx=$((idx + 1))
    pbytes=$(out_profile_candidate_bytes "${plist}")
    echo "  ${pdir} (would free $(human_bytes "${pbytes}"))"
    while IFS= read -r path; do
      [[ -n "${path}" ]] || continue
      echo "    - ${path}"
    done < "${plist}"
  done
  echo "  Total out_profile savings estimate: $(human_bytes "${out_profile_prune_bytes}")"
  echo "  Preserved lightweight files: out_profile/*.csv, out_profile/*/*.csv, profile summaries/stage timings/manifests, and existing products/config/metadata."
  echo
fi

if (( PRUNE_HEAVY )) && ((${#prune_runs[@]})); then
  echo "Prune plan (retained lightweight files + pruned heavy roots):"
  idx=0
  for run in "${prune_runs[@]}"; do
    keep_list=${prune_keep_lists[$idx]}
    idx=$((idx + 1))
    echo "  Run: ${run}"
    echo "    Prune candidates:"
    echo "      ${run}/heavy/"
    echo "      ${run}/raw_ppm/"
    if (( PRUNE_RAW_PPM )); then
      echo "      duplicate *.ppm where PNG exists"
    fi
    if (( PRUNE_RING_FRAMES )); then
      echo "      ring_frame_*.ppm / full_ring_*.ppm / ring_preview_*.ppm"
    fi
    if (( PRUNE_ANNULUS )); then
      echo "      annulus *.bin / *annulus* files"
    fi
    echo "    Retained lightweight files:"
    if [[ -s "${keep_list}" ]]; then
      while IFS= read -r kf; do
        echo "      ${kf}"
      done < "${keep_list}"
    else
      echo "      (none discovered; CSV/config/metadata still retained)"
    fi
  done
  echo
fi

current_outputs_bytes=$(bytes_of_path "${OUTPUTS_DIR}")
if [[ "${MAX_TOTAL_GB}" != "0" && "${MAX_TOTAL_GB}" != "0.0" ]]; then
  max_bytes=$(python3 - "${MAX_TOTAL_GB}" <<'PY'
import sys
print(int(float(sys.argv[1]) * (1024**3)))
PY
)
  estimated_after=$((current_outputs_bytes - would_delete_bytes - would_prune_bytes - out_profile_prune_bytes))
  if (( estimated_after > max_bytes )); then
    echo "WARNING: estimated outputs size after policy ($(human_bytes "${estimated_after}")) exceeds --max-total-gb (${MAX_TOTAL_GB} GB)."
    echo "         Consider lower keep counts or manual preserved-run review."
    echo
  fi
fi

if (( DRY_RUN || ! DO_DELETE )); then
  echo "Dry run only. No files were deleted."
  for t in "${prune_keep_lists[@]}"; do rm -f "${t}"; done
  for t in "${out_profile_prune_lists[@]}"; do rm -f "${t}"; done
  exit 0
fi

for run in "${delete_runs[@]}"; do
  [[ -n "${run}" && -d "${run}" && "${run}" == "${OUTPUTS_DIR}"/* ]] || continue
  rm -rf -- "${run}"
done

if (( PRUNE_HEAVY )); then
  idx=0
  for run in "${prune_runs[@]}"; do
    keep_list=${prune_keep_lists[$idx]}
    idx=$((idx + 1))
    [[ -d "${run}" ]] || continue
    prune_run_heavy_apply "${run}" "${keep_list}" "outside-full-retention-window"
  done
fi

for t in "${prune_keep_lists[@]}"; do rm -f "${t}"; done
if (( INCLUDE_OUT_PROFILE )) && (( PRUNE_HEAVY )); then
  idx=0
  for pdir in "${out_profile_prune_profiles[@]}"; do
    plist=${out_profile_prune_lists[$idx]}
    idx=$((idx + 1))
    apply_out_profile_prune "${pdir}" "${plist}"
  done
fi
for t in "${out_profile_prune_lists[@]}"; do rm -f "${t}"; done

if (( INCLUDE_WORKING_OUTS )) && (( PRUNE_HEAVY )); then
  while IFS= read -r work_dir; do
    [[ -d "${work_dir}" ]] || continue
    if (( PRUNE_ANNULUS )); then
      find "${work_dir}" -type f \( -name '*annulus*' -o -name '*.bin' \) -delete 2>/dev/null || true
    fi
    if (( PRUNE_RAW_PPM )); then
      find "${work_dir}" -type f -name '*.ppm' -delete 2>/dev/null || true
    fi
    if (( PRUNE_RING_FRAMES )); then
      find "${work_dir}" -type f \( -name 'ring_frame_*.ppm' -o -name 'full_ring_*.ppm' -o -name 'ring_preview_*.ppm' \) -delete 2>/dev/null || true
    fi
  done < <(find "${REPO_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'out*' ! -path "${OUTPUTS_DIR}" ! -path "${REPO_ROOT}/out_profile" | sort)
fi

echo "Cleanup complete."
echo "New outputs size: $(du -sh "${OUTPUTS_DIR}" 2>/dev/null | awk '{print $1}')"
