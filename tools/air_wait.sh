#!/usr/bin/env bash
# Wait for an AIR run to finish (polls `databricks air get`; `--watch` streaming can lag, SPEC OQ-15),
# then print the tail of its logs.   Usage: tools/air_wait.sh <JOB_RUN_ID> [profile] [tail-lines]
set -uo pipefail
RUN_ID=$1; PROFILE=${2:-${DATABRICKS_CONFIG_PROFILE:-am-rag-fevm}}; TAIL=${3:-400}
while true; do
  STATUS=$(databricks air get "$RUN_ID" -p "$PROFILE" -o json 2>/dev/null \
    | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d["status"], d.get("duration_seconds"))')
  case "${STATUS%% *}" in
    PENDING|QUEUED|RUNNING|STARTING|"") sleep 30 ;;
    *) break ;;
  esac
done
echo "AIR_RUN $RUN_ID finished: status=${STATUS%% *} duration_s=${STATUS##* }"
databricks air logs "$RUN_ID" -p "$PROFILE" --tail "$TAIL" 2>&1
