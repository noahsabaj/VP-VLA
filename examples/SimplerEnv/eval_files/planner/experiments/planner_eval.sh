#!/bin/bash
# Planner-only eval of one layout set at one depth-noise level, with the first-frame selector features in each
# task's planner_stats.jsonl. The 4 tasks run at the same time on one SAM3 server.
# Usage: planner_eval.sh GPU PORT NAME LEVEL LAYOUTS
#   NAME: the run folder ($RUNS_DIR/NAME/TASK). LEVEL: none | realistic | harsh. LAYOUTS: paper | heldout | seedN
#   | iidN (layout_args in common.sh).
# Environment:
#   SAM3_ARGS      SAM3 server options (default: none, as in every planner-only eval; the fast SAM3 settings
#                  change planner episodes).
#   PLANNER_SEED   --planner-seed: MPlib's random steps seeded per episode (default 0; empty = unseeded).
#   TASKS_AT_ONCE  tasks at the same time (default 4; 1 = one after the other). With the noise and planner seeds
#                  an episode gives the same result either way.
#   RGB_NOISE      none (default) | realistic | harsh | same: a real color camera's errors in the picture
#                  (camera_noise.py); same = the depth noise's level (one camera at LEVEL).
#   INSTRUCTIONS   reworded instructions (instruction_sets.py; default: the task's own).
#   TASK_SET       simpler (default: SimplerEnv's 4 tasks) | harder (the harder scenes; LAYOUTS=paper only).
#   PLAN_CHECK     1: the VLM plan check (planner/plan_check.py) on each episode's first frame, from a VLM server
#                  on PORT + 50 (PLAN_CHECK_VLM_ARGS; default: the VLA's fast VLM settings). It adds the feature
#                  vlm_plan_yes and the marked pictures (TASK/plan_check/); the planner's episodes stay the same.
#                  Default 0.
# Several evals at once: planner_evals.sh.
[ $# -eq 5 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; PORT=$2; NAME=$3; LEVEL=$4; LAYOUTS=$5
LS=$(layout_args "$LAYOUTS") || exit 1
PS=${PLANNER_SEED-0}
RGB_LEVEL=${RGB_NOISE:-none}
[ "$RGB_LEVEL" = same ] && RGB_LEVEL=$LEVEL
RGB=$(rgb_args "$RGB_LEVEL") || exit 1
OUT=$(run_dir "$NAME")
mkdir -p "$OUT"
LOG=$OUT/status.log

SERVER_ENV=(CUDA_VISIBLE_DEVICES=$GPU)
trap stop_servers EXIT
start_server $PORT "$OUT/sam3.log" "$SAM3_PYTHON" examples/Robocasa_tabletop/visual_prompt_utility/sam3_server.py \
  --port $PORT $SAM3_ARGS || exit 1
wait_port $PORT 600 || { log "$NAME: SAM3 server never opened port $PORT"; exit 1; }
CHECK=()
if [ "${PLAN_CHECK:-0}" = 1 ]; then
  VARGS=${PLAN_CHECK_VLM_ARGS-"--attn sdpa --seed-per-query --compile"}
  start_server $((PORT + 50)) "$OUT/vlm.log" "$SAM3_PYTHON" examples/SimplerEnv/eval_files/vlm_server_subtask_simpler.py \
    --port $((PORT + 50)) $VARGS || exit 1
  wait_port $((PORT + 50)) || { log "$NAME: VLM server never opened port $((PORT + 50))"; exit 1; }
  CHECK=(--plan-check-port $((PORT + 50)))
fi
log "START $NAME ($LEVEL depth, $RGB_LEVEL RGB, ${TASK_SET:-simpler} tasks, $LAYOUTS layouts, ${INSTRUCTIONS:-default} instructions, SAM3_ARGS=[$SAM3_ARGS], PLANNER_SEED=[$PS]," \
  "TASKS_AT_ONCE=${TASKS_AT_ONCE:-4}, PLAN_CHECK=${PLAN_CHECK:-0} [$VARGS]) on GPU $GPU (commit $(git rev-parse --short HEAD))"

task() {
  local out=$OUT/$1 t0=$SECONDS rc
  sim $GPU $1 0 "$(layout_episodes "$LAYOUTS" "$1")" "$out" --controller planner --planner mplib \
    --target-source camera --depth-noise $LEVEL --robot-mask joints --noise-seed 0 ${PS:+--planner-seed $PS} $LS $RGB \
    --sam3-port $PORT "${CHECK[@]}"
  rc=$?
  log "$NAME $1 exit=$rc episodes=$(grep -c "" "$out/planner_stats.jsonl" 2>/dev/null)" \
    "$(grep "Average success" "$out/sim.log" | tail -1 | xargs) ($(( (SECONDS - t0) / 60 )) min)"
}
run_pool ${TASKS_AT_ONCE:-4} 2 task "${TASKS[@]}"
stop_servers
log "DONE $NAME"
