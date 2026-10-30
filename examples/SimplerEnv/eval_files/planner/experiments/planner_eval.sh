#!/bin/bash
# Planner-only eval of one layout set at one depth-noise level, with the first-frame selector features in each
# task's planner_stats.jsonl. The 4 tasks run at the same time on one SAM3 server.
# Usage: planner_eval.sh GPU PORT NAME LEVEL LAYOUTS
#   NAME: the run folder ($RUNS_DIR/NAME/TASK). LEVEL: none | realistic | harsh, or harsh_xS for the difficulty sweep
#   (harsh's errors times S: harsh_x0.25, harsh_x0.5, harsh_x0.75, harsh_x1.5, harsh_x2; planner/geometry.py).
#   LAYOUTS: paper | heldout | seedN | iidN (layout_args in common.sh).
# Environment:
#   SAM_VERSION    sam3.1 (default) | sam3: the SAM server's model (common.sh).
#   SAM3_ARGS      SAM server options (default: none, as in every planner-only eval; the fast SAM3 settings
#                  change planner episodes).
#   PLANNER_SEED   --planner-seed: MPlib's random steps seeded per episode (default 0; empty = unseeded).
#   TASKS_AT_ONCE  tasks at the same time (default 4; 1 = one after the other). With the noise and planner seeds
#                  an episode gives the same result either way.
#   RGB_NOISE      none (default) | realistic | harsh | same: a real color camera's errors in the picture
#                  (camera_noise.py); same = the depth noise's level (one camera at LEVEL; not with harsh_xS).
#   INSTRUCTIONS   reworded instructions (instruction_sets.py; default: the task's own).
#   TASK_SET       simpler (default: SimplerEnv's 4 tasks) | harder (the harder scenes; LAYOUTS=paper or iidN).
#   PLAN_CHECK     1: the VLM plan check (planner/plan_check.py) on each episode's first frame, from a VLM server
#                  on PORT + 50 (PLAN_CHECK_VLM_ARGS; default: the VLA's fast VLM settings). It adds the feature
#                  vlm_plan_yes and the marked pictures (TASK/plan_check/); the planner's episodes stay the same.
#                  Default 0.
#   PLAN_CHECK_MARKS  with PLAN_CHECK=1: how the object is marked, comma-separated (plan_check.MARKS: tint,
#                  outline; default tint). Each mark is one more question and feature (outline: vlm_plan_yes_outline).
#   TASK_SOURCE    rule (default: the fixed rule, one pick-and-place) | vlm: the VLM plan (planner/vlm_plan.py),
#                  from a VLM server on PORT + 50 as for PLAN_CHECK (not with it): the VLM lists the pick-and-place
#                  moves and the planner does them in order | vp: VP-VLA's VLM marks from the same server (its
#                  subtasks and the object and target that it names in each; planner/vp_marks.py).
#   VLM_PLAN_MIN_SCORE  with TASK_SOURCE=vlm: a name whose chosen SAM mask scores below this is not found
#                  (--vlm-plan-min-score; default 0: no check).
#   MISSING_DEPTH  none (default) | target: in each episode, with chance 1/2, the place target gives no depth, as glass
#                  or black cloth would (planner/geometry.py MissingDepth; not in SimplerEnv's eggplant task).
#   ENERGY         1: an energy run (TASKS_AT_ONCE must be 1): each episode's GPU energy, the idle power and the null
#                  episodes (energy_setup and energy_baseline in common.sh). Default 0.
# Several evals at once: planner_evals.sh.
[ $# -eq 5 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; PORT=$2; NAME=$3; LEVEL=$4; LAYOUTS=$5
LS=$(layout_args "$LAYOUTS") || exit 1
PS=${PLANNER_SEED-0}
TS=${TASK_SOURCE:-rule}
case $TS in rule | vlm | vp) ;; *) echo "TASK_SOURCE must be rule, vlm or vp"; exit 1 ;; esac
MD=${MISSING_DEPTH:-none}
case $MD in none | target) ;; *) echo "MISSING_DEPTH must be none or target"; exit 1 ;; esac
[ "${PLAN_CHECK:-0}" = 1 ] && [ "$TS" != rule ] && { echo "PLAN_CHECK=1 is for the fixed rule, not TASK_SOURCE=$TS"; exit 1; }
RGB_LEVEL=${RGB_NOISE:-none}
[ "$RGB_LEVEL" = same ] && RGB_LEVEL=$LEVEL
RGB=$(rgb_args "$RGB_LEVEL") || exit 1
OUT=$(run_dir "$NAME")
mkdir -p "$OUT"
LOG=$OUT/status.log
energy_setup "${TASKS_AT_ONCE:-4}" || exit 1

SERVER_ENV=(CUDA_VISIBLE_DEVICES=$GPU)
trap stop_servers EXIT
start_sam_server $PORT "$OUT/sam3.log" $SAM3_ARGS || exit 1
wait_port $PORT 600 || { log "$NAME: SAM server never opened port $PORT"; exit 1; }
CHECK=()
if [ "${PLAN_CHECK:-0}" = 1 ] || [ "$TS" != rule ]; then
  VARGS=${PLAN_CHECK_VLM_ARGS-"--attn sdpa --seed-per-query --compile"}
  start_server $((PORT + 50)) "$OUT/vlm.log" "$SAM3_PYTHON" examples/SimplerEnv/eval_files/vlm_server_subtask_simpler.py \
    --port $((PORT + 50)) $VARGS || exit 1
  wait_port $((PORT + 50)) || { log "$NAME: VLM server never opened port $((PORT + 50))"; exit 1; }
  CHECK=(--plan-check-port $((PORT + 50)) --plan-check-marks "${PLAN_CHECK_MARKS:-tint}")
  [ "$TS" = vlm ] && CHECK=(--task-source vlm --vlm-plan-port $((PORT + 50)) --vlm-plan-min-score ${VLM_PLAN_MIN_SCORE:-0})
  [ "$TS" = vp ] && CHECK=(--task-source vp --vlm-plan-port $((PORT + 50)))
fi
energy_baseline $GPU "$OUT" "$LAYOUTS" $(for t in "${TASKS[@]}"; do echo "$t:0:$(layout_episodes "$LAYOUTS" "$t")"; done) ||
  exit 1
log "START $NAME (ENERGY=${ENERGY:-0}, $LEVEL depth, $RGB_LEVEL RGB, ${TASK_SET:-simpler} tasks, $LAYOUTS layouts, ${INSTRUCTIONS:-default} instructions, SAM_VERSION=$SAM_VERSION SAM3_ARGS=[$SAM3_ARGS], PLANNER_SEED=[$PS]," \
  "TASKS_AT_ONCE=${TASKS_AT_ONCE:-4}, PLAN_CHECK=${PLAN_CHECK:-0} PLAN_CHECK_MARKS=${PLAN_CHECK_MARKS:-tint} TASK_SOURCE=$TS VLM_PLAN_MIN_SCORE=${VLM_PLAN_MIN_SCORE:-0} MISSING_DEPTH=$MD [$VARGS]) on GPU $GPU (commit $(git rev-parse --short HEAD))"
nvidia-smi dmon -i $GPU -s pum -o DT -d 5 > "$OUT/dmon.txt" 2>&1 &
DMON=$!

task() {
  local out=$OUT/$1 t0=$SECONDS rc
  sim $GPU $1 0 "$(layout_episodes "$LAYOUTS" "$1")" "$out" --controller planner --planner mplib \
    --target-source camera --depth-noise $LEVEL --robot-mask joints --noise-seed 0 ${PS:+--planner-seed $PS} $LS $RGB \
    --sam3-port $PORT --missing-depth $MD "${CHECK[@]}"
  rc=$?
  log "$NAME $1 exit=$rc episodes=$(grep -c "" "$out/planner_stats.jsonl" 2>/dev/null)" \
    "$(grep "Average success" "$out/sim.log" | tail -1 | xargs) ($(( (SECONDS - t0) / 60 )) min)"
}
run_pool ${TASKS_AT_ONCE:-4} 2 task "${TASKS[@]}"
kill $DMON 2>/dev/null
stop_servers
log "DONE $NAME"
