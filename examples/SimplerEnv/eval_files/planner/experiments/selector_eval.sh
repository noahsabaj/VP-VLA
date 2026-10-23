#!/bin/bash
# Closed-loop ASM eval: on the first frame of each episode the saved ASM gives the episode to the planner or to
# the VLA. One server set for the VLA (policy, SAM3, VLM; as in vla_eval.sh) and one SAM3 server for the planner,
# shared by K sim clients.
# Usage: selector_eval.sh GPU PORT_BASE NAME LAYOUTS MODELS K [JOB...]
#   MODELS: the folder with the ASM models, MODELS/sel_v3_LEVEL.json (from selector.py --save), or a file name with
#   {} for the level (e.g. two_model/asm_{}.json). JOB = LEVEL:TASK:START:END (LEVEL: none | realistic | harsh) writes
#   to $RUNS_DIR/NAME/LEVEL/TASK/eps_START_END.
#   Default jobs: every episode of the layout set at each level of DEPTHS (default: harsh realistic none), in
#   chunks of CHUNK_SIZE episodes (default: all of a task's episodes in one job).
#   Ports: policy PORT_BASE, SAM PORT_BASE + 1, VLM PORT_BASE + 2, planner SAM PORT_BASE + 3.
# Environment: FAST, MPS and SAM_VERSION (both SAM servers) as in vla_eval.sh. PLANNER_SAM3_ARGS: the planner's
#   SAM server options (default: none, as in planner_eval.sh). PLANNER_SEED as in planner_eval.sh: give it the
#   same value as the planner-only evals that the closed-loop eval is compared with. RGB_NOISE as in
#   planner_eval.sh (same = each job's depth
#   level). INSTRUCTIONS as in vla_eval.sh. PLAN_CHECK=1: the VLM plan check (planner/plan_check.py) on the first
#   frame, from the VLA's VLM server; a "no" gives the episode to the VLA before the ASM decides. TASK_SOURCE as in
#   planner_eval.sh, from the VLA's VLM server (an ASM trained on records with the same TASK_SOURCE), and
#   VLM_PLAN_MIN_SCORE as in planner_eval.sh.
[ $# -ge 6 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; PB=$2; NAME=$3; LAYOUTS=$4; MODELS=$(run_dir "$5"); K=$6
shift 6
case $MODELS in *"{}"*) ;; *) MODELS=$MODELS/sel_v3_{}.json ;; esac
LS=$(layout_args "$LAYOUTS") || exit 1
JOBS=("$@")
if [ ${#JOBS[@]} -eq 0 ]; then
  for level in ${DEPTHS:-harsh realistic none}; do
    for chunk in $(all_chunks "$LAYOUTS" "${CHUNK_SIZE:-$(layout_episodes "$LAYOUTS")}"); do
      JOBS+=("$level:$chunk")
    done
  done
fi
PS=${PLANNER_SEED-0}
case ${RGB_NOISE:-none} in
  none | realistic | harsh | same) ;;
  *) echo "RGB_NOISE must be none, realistic, harsh or same, not '$RGB_NOISE'" >&2; exit 1 ;;
esac
OUT=$(run_dir "$NAME")
mkdir -p "$OUT"
LOG=$OUT/status.log
fast_defaults

SERVER_ENV=(CUDA_VISIBLE_DEVICES=$GPU)
trap stop_servers EXIT
if [ "${MPS:-1}" = 1 ]; then start_mps $GPU "$OUT/mps_log" || { log "MPS daemon did not start"; exit 1; }; fi
T0=$SECONDS
start_vla_servers $PB "$OUT" &&
  start_sam_server $((PB + 3)) "$OUT/planner_sam3.log" $PLANNER_SAM3_ARGS &&
  wait_port $PB && wait_port $((PB + 1)) && wait_port $((PB + 2)) && wait_port $((PB + 3)) ||
  { log "servers did not start"; exit 1; }
log "START K=$K MPS=${MPS:-1} LAYOUTS=$LAYOUTS MODELS=$MODELS RGB_NOISE=${RGB_NOISE:-none}" \
  "INSTRUCTIONS=${INSTRUCTIONS:-default} PLANNER_SEED=[$PS] PLAN_CHECK=${PLAN_CHECK:-0} TASK_SOURCE=${TASK_SOURCE:-rule} VLM_PLAN_MIN_SCORE=${VLM_PLAN_MIN_SCORE:-0}" \
  "SAM_VERSION=$SAM_VERSION SAM3_ARGS=[$SAM3_ARGS] PLANNER_SAM3_ARGS=[$PLANNER_SAM3_ARGS] VLM_ARGS=[$VLM_ARGS] POLICY_ARGS=[$POLICY_ARGS]" \
  "SIM_ARGS=[$SIM_ARGS]" \
  "on GPU $GPU, commit $(git log -1 --format=%h), servers up in $((SECONDS - T0)) s: ${JOBS[*]}"
nvidia-smi dmon -i $GPU -s um -d 5 > "$OUT/dmon.txt" 2>&1 &
DMON=$!

CHECK=()
[ "${PLAN_CHECK:-0}" = 1 ] && CHECK=(--plan-check-port $((PB + 2)))
[ "${TASK_SOURCE:-rule}" = vlm ] && CHECK+=(--task-source vlm --vlm-plan-port $((PB + 2)) --vlm-plan-min-score ${VLM_PLAN_MIN_SCORE:-0})

job() {
  local level task start end out t0=$SECONDS rc stats rgb
  IFS=: read -r level task start end <<< "$1"
  rgb=${RGB_NOISE:-none}
  [ "$rgb" = same ] && rgb=$level
  rgb=$(rgb_args "$rgb")
  out=$OUT/$level/$task/eps_${start}_$end
  stats=$out/planner_stats.jsonl
  sim $GPU $task $start $end "$out" --controller selector --selector "${MODELS//\{\}/$level}" --planner mplib \
    --target-source camera --depth-noise $level --robot-mask joints --noise-seed 0 ${PS:+--planner-seed $PS} \
    --port $PB --sam3-port $((PB + 1)) --vlm-port $((PB + 2)) --planner-sam3-port $((PB + 3)) $LS $rgb \
    "${CHECK[@]}" $VP_ARGS $SIM_ARGS
  rc=$?
  log "job $1 exit=$rc tries=$(grep -c "" "$stats" 2>/dev/null)" \
    "vla_picks=$(grep -c '"selector_choice": "vla"' "$stats" 2>/dev/null)" \
    "$(grep "Average success" "$out/sim.log" | tail -1 | xargs) ($((SECONDS - t0)) s)"
}
run_pool $K 2 job "${JOBS[@]}"
kill $DMON 2>/dev/null
stop_servers
log "DONE K=$K in $((SECONDS - T0)) s"
