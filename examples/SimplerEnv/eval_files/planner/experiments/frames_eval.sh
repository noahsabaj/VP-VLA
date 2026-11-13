#!/bin/bash
# The first frame of every episode of one layout set, for the frontier-model study's plan arm (planner/frontier.py;
# the experiments README, "Frontier-model study"): in each task's folder, frames/KEY.png and a line in frames.jsonl
# with the episode, its instruction and the picture's hash. The robot does not move, and no model server runs. The
# frames then go to the machine that holds the API key (frontier_answers.py), and the answers come back for
# planner_eval.sh with VLM_PLAN_ANSWERS.
# Usage: frames_eval.sh GPU NAME LAYOUTS
#   NAME: the run folder ($RUNS_DIR/NAME/TASK). LAYOUTS: paper | heldout | seedN | iidN (layout_args in common.sh).
# Environment:
#   TASK_SET, INSTRUCTIONS  as in planner_eval.sh: the tasks, and the reworded instructions that go into frames.jsonl.
#   TASKS_AT_ONCE  tasks at the same time (default 4).
[ $# -eq 3 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; NAME=$2; LAYOUTS=$3
LS=$(layout_args "$LAYOUTS") || exit 1
OUT=$(run_dir "$NAME")
mkdir -p "$OUT"
LOG=$OUT/status.log
log "START $NAME (first frames, ${TASK_SET:-simpler} tasks, $LAYOUTS layouts, ${INSTRUCTIONS:-default} instructions)" \
  "on GPU $GPU (commit $(git rev-parse --short HEAD))"

# task TASK: the first frames of all of the task's episodes into $OUT/TASK
task() {
  local n rc got
  n=$(layout_episodes "$LAYOUTS" "$1")
  sim $GPU "$1" 0 "$n" "$OUT/$1" --controller frames --no-action-plots $LS
  rc=$?
  got=$(grep -c "" "$OUT/$1/frames.jsonl" 2>/dev/null)
  [ $rc = 0 ] && [ "${got:-0}" != "$n" ] && rc=1  # an episode without its frame
  log "$NAME $1 exit=$rc frames=${got:-0} of $n"
}
run_pool ${TASKS_AT_ONCE:-4} 2 task "${TASKS[@]}"
log "DONE $NAME"
