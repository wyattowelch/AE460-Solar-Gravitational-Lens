#!/usr/bin/env bash
set -euo pipefail

echo "== Dashboard / Demo process check =="
echo "-- dashboard processes --"
ps -eo pid,ppid,stat,%cpu,%mem,cmd | grep -E 'telemetry_dashboard/dashboard.py|run_dashboard.sh' | grep -v grep || true

echo "-- sim processes --"
ps -eo pid,ppid,stat,%cpu,%mem,cmd | grep -E 'sgl_pi_flight|sgl_jetson_service|run_local_demo.sh|run_tcp_pi.sh|run_tcp_jetson.sh' | grep -v grep || true

echo "-- top repo-related CPU users --"
ps -eo pid,ppid,stat,%cpu,%mem,cmd --sort=-%cpu | grep -E 'sgl_obc_pi_jetson_full|telemetry_dashboard|sgl_pi_flight|sgl_jetson_service|run_local_demo|run_dashboard|python3' | head -n 20 || true

echo "-- system top CPU snapshot --"
ps -eo pid,ppid,stat,%cpu,%mem,cmd --sort=-%cpu | head -n 20
