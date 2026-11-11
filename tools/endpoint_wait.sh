#!/usr/bin/env bash
# Wait for a Model Serving endpoint to become READY (or fail / time out), printing state changes and
# counting server restarts (a crash loop shows up as repeated "SERVER RESTARTED" with no traceback).
#   Usage: tools/endpoint_wait.sh <ENDPOINT> [profile] [max-minutes]
set -uo pipefail
EP=$1; PROFILE=${2:-${DATABRICKS_CONFIG_PROFILE:-am-rag-fevm}}; MAX=${3:-45}
LAST=""; START=$(date +%s)
while (( $(date +%s) - START < MAX * 60 )); do
  OUT=$(databricks serving-endpoints get "$EP" -p "$PROFILE" -o json 2>&1 | python3 -c '
import sys, json
d = json.loads((lambda t: t[t.find("{"):])(sys.stdin.read())); s = d.get("state", {})
ents = (d.get("pending_config") or d.get("config") or {}).get("served_entities", [])
e = ents[0] if ents else {}
st = e.get("state") or {}
print(s.get("ready"), s.get("config_update"), e.get("name", "-"), st.get("deployment"),
      (st.get("deployment_state_message") or "").replace(" ", "_")[:120])')
  read -r READY UPDATE ENTITY DEPLOY MSG <<<"$OUT"
  if [[ "$OUT" != "$LAST" ]]; then echo "$(date -u +%T) ready=$READY update=$UPDATE deploy=$DEPLOY msg=${MSG//_/ }"; LAST=$OUT; fi
  if [[ "$READY" == READY && "$UPDATE" == NOT_UPDATING ]]; then echo "ENDPOINT_READY $EP"; exit 0; fi
  if [[ "$UPDATE" == UPDATE_FAILED || "$DEPLOY" == *FAILED* ]]; then echo "ENDPOINT_FAILED $EP"; break; fi
  sleep 60
done
echo "--- server log tail ($ENTITY), restarts:"
databricks serving-endpoints logs "$EP" "$ENTITY" -p "$PROFILE" -o json 2>/dev/null \
  | python3 -c 'import sys,json; l=json.loads((lambda t: t[t.find("{"):])(sys.stdin.read())).get("logs",""); print(l.count("SERVER RESTARTED")); print(l[-4000:])'
exit 1
