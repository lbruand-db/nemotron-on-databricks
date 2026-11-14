#!/usr/bin/env bash
# Wait for an AIR run to finish, then print the tail of its logs.
# Polls status instead of relying on `--watch` streaming, which can lag (SPEC OQ-15), and reports the
# underlying task state: `air get` says RUNNING while the task is still PENDING on GPU capacity (OQ-16).
# A CLI error marked `"retryable": false` (e.g. unknown run ID) exits 2 at once. Other unreadable output is retried
# (the CLI sometimes prints non-JSON), but after MAX_FAILS polls in a row (default 10, ~5 min) the script prints
# the CLI output and exits 2: expired login, wrong profile.
#   Usage: tools/air_wait.sh <JOB_RUN_ID> [profile] [tail-lines]
set -uo pipefail
RUN_ID=$1; PROFILE=${2:-${DATABRICKS_CONFIG_PROFILE:-am-rag-fevm}}; TAIL=${3:-400}
MAX_FAILS=${MAX_FAILS:-10}
LAST=""; FAILS=0
# Prints "<status> <duration_s>", "FATAL" for a non-retryable CLI error, or nothing for unreadable output.
PARSE_STATUS='import sys, json
t = sys.stdin.read()
try:
    j = json.loads(t[t.find("{"):])
    if isinstance(j.get("error"), dict) and j["error"].get("retryable") is False:
        print("FATAL")
    else:
        print(j["data"]["status"], j["data"].get("duration_seconds"))
except (ValueError, KeyError, TypeError, AttributeError):
    pass'
PARSE_TASK='import sys, json
t = sys.stdin.read()
try:
    task = (json.loads(t[t.find("{"):]).get("tasks") or [{}])[0]
    print(task.get("state", {}).get("life_cycle_state", "?"))
except (ValueError, AttributeError, TypeError):
    print("?")'
while true; do
  RAW=$(databricks air get "$RUN_ID" -p "$PROFILE" -o json 2>&1)
  STATUS=$(python3 -c "$PARSE_STATUS" <<< "$RAW")
  if [[ -z "$STATUS" || "$STATUS" == FATAL ]]; then
    FAILS=$((FAILS + 1))
    if [[ "$STATUS" == FATAL ]] || (( FAILS >= MAX_FAILS )); then
      echo "AIR_WAIT_ERROR: could not read the status of run $RUN_ID ($FAILS tries). Last CLI output:" >&2
      echo "$RAW" | tail -20 >&2
      exit 2
    fi
    sleep 30; continue
  fi
  FAILS=0
  TASK=$(databricks jobs get-run "$RUN_ID" -p "$PROFILE" -o json 2>/dev/null | python3 -c "$PARSE_TASK")
  if [[ "$TASK" != "$LAST" ]]; then echo "$(date -u +%T) run=${STATUS%% *} task=$TASK elapsed_s=${STATUS##* }"; LAST=$TASK; fi
  case "${STATUS%% *}" in
    PENDING|QUEUED|RUNNING|STARTING) sleep 30 ;;
    *) break ;;
  esac
done
echo "AIR_RUN $RUN_ID finished: status=${STATUS%% *} duration_s=${STATUS##* }"
databricks air logs "$RUN_ID" -p "$PROFILE" --tail "$TAIL" 2>&1
