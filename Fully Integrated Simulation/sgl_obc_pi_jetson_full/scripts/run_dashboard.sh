#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE="auto"   # auto|review|live
KEEP_EXISTING=0
ARG1="outputs/latest"
REFRESH_MS=""
REFRESH_GIVEN=0
CFG_PATH_FOR_DASH=""

usage() {
  cat <<USAGE
Usage:
  scripts/run_dashboard.sh [--review] [--keep-existing] [config_or_run_path] [refresh_ms]

Behavior:
  - Config path input => live mode (fresh out_dir by default, moved aside safely unless --keep-existing).
  - outputs/latest or outputs/<case> => review mode.
  - --review forces review mode.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --review)
      MODE="review"
      shift
      ;;
    --live)
      MODE="live"
      shift
      ;;
    --keep-existing)
      KEEP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "${ARG1}" == "outputs/latest" ]]; then
        ARG1="$1"
      elif [[ -z "${REFRESH_MS}" ]]; then
        REFRESH_MS="$1"
        REFRESH_GIVEN=1
      else
        echo "ERROR: unexpected arg: $1" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

RUN_ARG="${ARG1}"

if [[ "${MODE}" == "auto" ]]; then
  if [[ "${ARG1}" == outputs/* || "${ARG1}" == */outputs/* ]]; then
    MODE="review"
  elif [[ -d "${ARG1}" ]]; then
    MODE="review"
  elif [[ -f "${ARG1}" ]]; then
    MODE="live"
  else
    # fallback to review path semantics
    MODE="review"
  fi
fi

if [[ "${MODE}" == "live" ]]; then
  CFG_PATH="${ARG1}"
  if [[ ! -f "${CFG_PATH}" ]]; then
    echo "ERROR: config not found: ${CFG_PATH}" >&2
    exit 1
  fi
  out_dir=$(python3 - <<PY
import json
with open(r"${CFG_PATH}", "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("out_dir", "out"))
PY
)
  if [[ "${out_dir}" = /* ]]; then
    out_root="${out_dir}"
  else
    out_root="${REPO_ROOT}/${out_dir}"
  fi
  if [[ -d "${out_root}" && "${KEEP_EXISTING}" != "1" ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    moved="${out_root}_OLD_${ts}"
    echo "Live mode: moving existing out_dir aside:"
    echo "  ${out_root}"
    echo "  -> ${moved}"
    mv "${out_root}" "${moved}"
  fi
  mkdir -p "${out_root}/mission_store"
  # seed empty files so dashboard can start before sim writes first rows
  : > "${out_root}/mission_store/telemetry_cycles.csv"
  : > "${out_root}/mission_store/events.csv"
  : > "${out_root}/mission_store/products_manifest.csv"
  : > "${out_root}/mission_store/downlink_queue.csv"
  RUN_ARG="${out_root}/mission_store/telemetry_cycles.csv"
  CFG_PATH_FOR_DASH="${CFG_PATH}"
  echo "Dashboard live mode against: ${RUN_ARG}"
else
  echo "Dashboard review mode against: ${RUN_ARG}"
fi

if [[ "${REFRESH_GIVEN}" != "1" ]]; then
  if [[ "${MODE}" == "review" ]]; then
    REFRESH_MS="1000"
  else
    REFRESH_MS="200"
  fi
fi

# Fail fast with a clear hint instead of showing a Python traceback.
if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
need = ["pyqtgraph", "PyQt5"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  echo "ERROR: Missing dashboard dependencies. Run: python3 -m pip install -r tools/telemetry_dashboard/requirements.txt" >&2
  exit 2
fi

pushd "${REPO_ROOT}" >/dev/null
if [[ -n "${CFG_PATH_FOR_DASH}" ]]; then
  exec python3 tools/telemetry_dashboard/dashboard.py "${RUN_ARG}" --refresh-ms "${REFRESH_MS}" --config-path "${CFG_PATH_FOR_DASH}"
else
  exec python3 tools/telemetry_dashboard/dashboard.py "${RUN_ARG}" --refresh-ms "${REFRESH_MS}"
fi
