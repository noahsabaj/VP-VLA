#!/bin/bash
# VLA-only eval with one server set (policy, SAM3, VLM) shared by K sim clients. Episodes are deterministic, so
# splitting them into chunks gives the same videos as one client.
# Usage: vla_eval.sh GPU PORT_BASE NAME LAYOUTS K [CHUNK...]
#   NAME: the run folder; chunk TASK:START:END writes to $RUNS_DIR/NAME/TASK/eps_START_END. LAYOUTS: paper |
#   heldout | seedN | iidN (layout_args in common.sh). Default chunks: every episode of the layout set, CHUNK_SIZE
#   (default 8) per chunk.
#   Ports: policy PORT_BASE, SAM3 PORT_BASE + 1, VLM PORT_BASE + 2.
# Environment:
#   FAST=1 (default): the fast settings (fast_defaults in common.sh) for each of SAM3_ARGS, VLM_ARGS, POLICY_ARGS
#     and SIM_ARGS that is not set; FAST=0: the paper's settings.
#   SEED_OFFSET=N: the VLM's --seed-offset (another repeatable seed per query, for reruns; default 0).
#   HIDDEN=1: save the policy's first-step hidden states in NAME/hidden (for the linear probe).
#   PERTURB=K: with HIDDEN=1, also save the first-step actions for K changed copies of the input (vla_perturbation.py).
#   MPS=1 (default): the servers share the GPU through an MPS daemon of this run only (start_mps in common.sh);
#     MPS=0: no daemon.
#   SAM_VERSION=sam3 (default, the paper's setup) | sam3.1: the segmentation model of the visual prompts
#     (start_vla_servers in common.sh).
#   RGB_NOISE=none (default) | realistic | harsh: a real color camera's errors in the picture (camera_noise.py).
[ $# -ge 5 ] || { awk 'NR > 1 { if (!/^#/) exit; print }' "$0"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPU=$1; PB=$2; NAME=$3; LAYOUTS=$4; K=$5
shift 5
LS=$(layout_args "$LAYOUTS") || exit 1
CHUNKS=("$@")
[ ${#CHUNKS[@]} -gt 0 ] || mapfile -t CHUNKS < <(all_chunks "$LAYOUTS" "${CHUNK_SIZE:-8}")
OUT=$(run_dir "$NAME")
mkdir -p "$OUT"
LOG=$OUT/status.log
fast_defaults
[ "${SEED_OFFSET:-0}" != 0 ] && VLM_ARGS="$VLM_ARGS --seed-offset $SEED_OFFSET"
RGB=$(rgb_args "${RGB_NOISE:-none}") || exit 1
[ "${HIDDEN:-0}" = 1 ] && SIM_ARGS="$SIM_ARGS --vla-hidden-dir $OUT/hidden"
if [ "${PERTURB:-0}" != 0 ]; then
  [ "${HIDDEN:-0}" = 1 ] || { echo "PERTURB needs HIDDEN=1" >&2; exit 1; }
  SIM_ARGS="$SIM_ARGS --vla-perturb $PERTURB"
fi

SERVER_ENV=(CUDA_VISIBLE_DEVICES=$GPU)
trap stop_servers EXIT
if [ "${MPS:-1}" = 1 ]; then start_mps $GPU "$OUT/mps_log" || { log "MPS daemon did not start"; exit 1; }; fi
T0=$SECONDS
start_vla_servers $PB "$OUT" && wait_port $PB && wait_port $((PB + 1)) && wait_port $((PB + 2)) ||
  { log "servers did not start"; exit 1; }
log "START K=$K MPS=${MPS:-1} LAYOUTS=$LAYOUTS RGB_NOISE=${RGB_NOISE:-none} SAM_VERSION=${SAM_VERSION:-sam3} SAM3_ARGS=[$SAM3_ARGS] VLM_ARGS=[$VLM_ARGS] POLICY_ARGS=[$POLICY_ARGS]" \
  "SIM_ARGS=[$SIM_ARGS] on GPU $GPU, commit $(git log -1 --format=%h), servers up in $((SECONDS - T0)) s: ${CHUNKS[*]}"
nvidia-smi dmon -i $GPU -s um -d 5 > "$OUT/dmon.txt" 2>&1 &
DMON=$!

chunk() {
  local task start end out t0=$SECONDS rc
  IFS=: read -r task start end <<< "$1"
  out=$OUT/$task/eps_${start}_$end
  sim $GPU $task $start $end "$out" --controller vla --port $PB --sam3-port $((PB + 1)) --vlm-port $((PB + 2)) \
    $LS $RGB $VP_ARGS $SIM_ARGS
  rc=$?
  log "chunk $1 exit=$rc videos=$(find "$out" -name "*.mp4" | wc -l)" \
    "$(grep "Average success" "$out/sim.log" | tail -1 | xargs) ($((SECONDS - t0)) s)"
}
run_pool $K 2 chunk "${CHUNKS[@]}"
kill $DMON 2>/dev/null
stop_servers
log "DONE K=$K in $((SECONDS - T0)) s"
