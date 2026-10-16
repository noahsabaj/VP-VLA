#!/bin/bash
# Several planner-only evals at once, each with its own SAM3 server (planner_eval.sh; its tasks also run at the
# same time). Each eval needs about 4 GB of GPU memory (its SAM3 server and 4 sims).
# Usage: planner_evals.sh GPU PORT_BASE AT_ONCE SET...   (SET = NAME:LEVEL:LAYOUTS, e.g. pseed_paper_harsh:harsh:paper)
#   AT_ONCE evals run at the same time; eval i uses port PORT_BASE + i. SAM3_ARGS, PLANNER_SEED, TASKS_AT_ONCE and
#   RGB_NOISE go to every eval. Progress: each eval's NAME/status.log; at the end, a line in $RUNS_DIR/status.log.
[ $# -ge 4 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; PORT=$2; AT_ONCE=$3
shift 3
LOG=$RUNS_DIR/status.log
items=()
for set in "$@"; do
  IFS=: read -r name level layouts <<< "$set"
  [ -n "$name" ] && [ -n "$level" ] && [ -n "$layouts" ] || { echo "SET must be NAME:LEVEL:LAYOUTS, not '$set'"; exit 1; }
  items+=("$set:$((PORT + ${#items[@]}))")
done
one_eval() {
  local name level layouts port
  IFS=: read -r name level layouts port <<< "$1"
  bash "$EXPERIMENTS/planner_eval.sh" $GPU $port $name $level $layouts > /dev/null 2>&1
}
run_pool $AT_ONCE 5 one_eval "${items[@]}"
log "planner_evals: done ($*)"
