#!/usr/bin/env bash
set -euo pipefail

echo "Stopping dashboard-related processes (targeted)..."

pids=$(ps -eo pid,cmd | grep -E 'tools/telemetry_dashboard/dashboard.py|scripts/run_dashboard.sh|python3 .*telemetry_dashboard/dashboard.py' | grep -v grep | awk '{print $1}')
if [[ -z "${pids}" ]]; then
  echo "No dashboard processes found."
  exit 0
fi

echo "Found PIDs: ${pids}"
for p in ${pids}; do
  kill "${p}" 2>/dev/null || true
done
sleep 1

left=$(ps -eo pid,cmd | grep -E 'tools/telemetry_dashboard/dashboard.py|scripts/run_dashboard.sh|python3 .*telemetry_dashboard/dashboard.py' | grep -v grep | awk '{print $1}')
if [[ -n "${left}" ]]; then
  echo "Force-stopping remaining PIDs: ${left}"
  for p in ${left}; do
    kill -9 "${p}" 2>/dev/null || true
  done
fi

echo "Done."
