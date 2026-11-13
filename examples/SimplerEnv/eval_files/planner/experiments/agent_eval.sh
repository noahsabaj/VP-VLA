#!/bin/bash
# An eval of the frontier agent on one layout set (planner/agent.py; the experiments README, "Frontier-model study"):
# the frontier model drives the arm with tools. In each task's folder: agent_stats.jsonl (one line per episode),
# transcripts/ (each episode's conversation and its pictures) and the evaluator's own files. No SAM server and no VLM
# server run; the model answers through the relay on the machine that holds the API key (agent_relay.py), which this
# server reaches through an SSH tunnel that that machine opens. This server holds no key.
# Usage: agent_eval.sh GPU NAME LAYOUTS [CHUNK...]
#   NAME: the run folder ($RUNS_DIR/NAME/TASK). LAYOUTS: paper | heldout | seedN | iidN (layout_args in common.sh).
#   CHUNK: TASK:START:END, only episodes START to END - 1 of TASK, into $RUNS_DIR/NAME/TASK/eps_START_END (the rest of
#   a set after an outage stopped its eval: no episode is run a second time). Default: every episode of each task.
# Environment:
#   AGENT_RELAY    the relay's address: unix:PATH (the tunnel's Unix socket) or HOST:PORT. Needed with AGENT_MODEL=relay.
#   AGENT_MODEL    relay (default) | scripted: a made-up model for a pipeline check, which sends no request.
#   TASK_SET, INSTRUCTIONS  as in planner_eval.sh.
#   TASKS_AT_ONCE  tasks at the same time (default 4); each holds its simulator while the model answers.
#   PLANNER_SEED   --planner-seed: MPlib's random steps seeded per episode (default 0; empty = unseeded).
[ $# -ge 3 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; NAME=$2; LAYOUTS=$3
shift 3
for job in "$@"; do
  [[ $job =~ ^([^:]+):[0-9]+:[0-9]+$ && " ${TASKS[*]} " == *" ${BASH_REMATCH[1]} "* ]] ||
    { echo "a chunk is TASK:START:END with a task of TASK_SET=${TASK_SET:-simpler}, not '$job'" >&2; exit 1; }
done
JOBS=("$@")
[ ${#JOBS[@]} -gt 0 ] || JOBS=("${TASKS[@]}")
LS=$(layout_args "$LAYOUTS") || exit 1
PS=${PLANNER_SEED-0}
AM=${AGENT_MODEL:-relay}
case $AM in
  relay) [ -n "${AGENT_RELAY:-}" ] || { echo "AGENT_MODEL=relay needs AGENT_RELAY (unix:PATH or HOST:PORT)"; exit 1; }
         MODEL_ARGS=(--agent-model relay --agent-relay "$AGENT_RELAY") ;;
  scripted) MODEL_ARGS=(--agent-model scripted) ;;
  *) echo "AGENT_MODEL must be relay or scripted"; exit 1 ;;
esac
OUT=$(run_dir "$NAME")
mkdir -p "$OUT"
LOG=$OUT/status.log
log "START $NAME (the frontier agent, AGENT_MODEL=$AM, ${TASK_SET:-simpler} tasks, $LAYOUTS layouts," \
  "${INSTRUCTIONS:-default} instructions, PLANNER_SEED=[$PS], TASKS_AT_ONCE=${TASKS_AT_ONCE:-4}) on GPU $GPU" \
  "(commit $(git rev-parse --short HEAD)): ${JOBS[*]}"

# task JOB: all of a task's episodes (JOB = TASK) into $OUT/TASK, or a chunk (TASK:START:END) into
# $OUT/TASK/eps_START_END
task() {
  local t s e out t0=$SECONDS rc got
  IFS=: read -r t s e <<< "$1"
  out=$OUT/$t
  if [ -n "$s" ]; then out=$out/eps_${s}_$e; else s=0 e=$(layout_episodes "$LAYOUTS" "$t"); fi
  sim $GPU $t $s $e "$out" --controller agent --planner mplib ${PS:+--planner-seed $PS} --no-action-plots $LS \
    "${MODEL_ARGS[@]}"
  rc=$?
  got=$(grep -c "" "$out/agent_stats.jsonl" 2>/dev/null)
  [ $rc = 0 ] && [ "${got:-0}" != $((e - s)) ] && rc=1  # an episode without its record
  log "$NAME $1 exit=$rc episodes=${got:-0} of $((e - s)) ($(( (SECONDS - t0) / 60 )) min)"
}
run_pool ${TASKS_AT_ONCE:-4} 2 task "${JOBS[@]}"
log "DONE $NAME"
