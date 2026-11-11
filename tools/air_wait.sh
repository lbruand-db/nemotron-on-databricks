#!/usr/bin/env bash
# Wait for an AIR run to finish, then print the tail of its logs.
# Polls status instead of relying on `--watch` streaming, which can lag (SPEC OQ-15), and reports the
# underlying task state: `air get` says RUNNING while the task is still PENDING on GPU capacity (OQ-16).
#   Usage: tools/air_wait.sh <JOB_RUN_ID> [profile] [tail-lines]
set -uo pipefail
RUN_ID=$1; PROFILE=${2:-${DATABRICKS_CONFIG_PROFILE:-am-rag-fevm}}; TAIL=${3:-400}
LAST=""
while true; do
  STATUS=$(databricks air get "$RUN_ID" -p "$PROFILE" -o json 2>/dev/null \
    | python3 -c 'import sys,json; d=json.loads((lambda t: t[t.find("{"):])(sys.stdin.read()))["data"]; print(d["status"], d.get("duration_seconds"))')
  TASK=$(databricks jobs get-run "$RUN_ID" -p "$PROFILE" -o json 2>/dev/null \
    | python3 -c 'import sys,json; t=(json.loads((lambda t: t[t.find("{"):])(sys.stdin.read())).get("tasks") or [{}])[0]; print(t.get("state",{}).get("life_cycle_state","?"))')
  if [[ "$TASK" != "$LAST" ]]; then echo "$(date -u +%T) run=${STATUS%% *} task=$TASK elapsed_s=${STATUS##* }"; LAST=$TASK; fi
  case "${STATUS%% *}" in
    PENDING|QUEUED|RUNNING|STARTING|"") sleep 30 ;;
    *) break ;;
  esac
done
echo "AIR_RUN $RUN_ID finished: status=${STATUS%% *} duration_s=${STATUS##* }"
databricks air logs "$RUN_ID" -p "$PROFILE" --tail "$TAIL" 2>&1
