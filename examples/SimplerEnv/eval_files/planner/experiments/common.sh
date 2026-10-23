#!/bin/bash
# Shared parts of the eval runners (planner_eval.sh, planner_evals.sh, vla_eval.sh, selector_eval.sh, sim_task.sh).
# Source it.
#
# It needs, from the environment: RUNS_DIR (the folder for the run folders) and what examples/eval_env.sh
# resolves (CONDA_ENVS_ROOT or the *_PYTHON paths, and SIMPLERENV_PATH). CKPT: the VP-VLA SimplerEnv checkpoint
# (default: the one in playground/). The working folder becomes the repository root.

EXPERIMENTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$EXPERIMENTS/../../../../.." && pwd)
source "$REPO/examples/eval_env.sh"
vpvla_require STARVLA_PYTHON SAM3_PYTHON SIMPLER_PYTHON
vpvla_require_dir SIMPLERENV_PATH
: "${RUNS_DIR:?set RUNS_DIR to the folder for the run folders}"
# The segmentation model of every SAM server (the VLA's visual prompts and the planner's masks): sam3.1 (the
# default since 2026-10-23: SAM 3.1's detector, from the sam31 env) | sam3 (transformers Sam3Model, the paper's
# setup and every eval before 2026-10-23).
SAM_VERSION=${SAM_VERSION:-sam3.1}
case $SAM_VERSION in
  sam3) SAM_PYTHON=$SAM3_PYTHON ;;
  sam3.1) vpvla_require SAM31_PYTHON; SAM_PYTHON=$SAM31_PYTHON ;;
  *) echo "SAM_VERSION must be sam3 or sam3.1, not '$SAM_VERSION'" >&2; exit 1 ;;
esac
CKPT=${CKPT:-$REPO/playground/Checkpoints/VP-VLA-OXE/checkpoints/steps_70000_pytorch_model.pt}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# The tasks of an eval (TASK_SET): simpler = SimplerEnv's 4 WidowX tasks (the default); harder = the 5 harder scenes
# (eval_files/harder_scenes.py), with their fixed layouts (LAYOUTS=paper) or a random one per episode (LAYOUTS=iidN).
case ${TASK_SET:-simpler} in
  simpler) TASKS=(PutSpoonOnTableClothInScene-v0 PutCarrotOnPlateInScene-v0
                  StackGreenCubeOnYellowCubeBakedTexInScene-v0 PutEggplantInBasketScene-v0) ;;
  harder) TASKS=(ClutteredSpoonOnTowel-v0 ClutteredCarrotOnPlate-v0 TwoStepSpoonAndCarrot-v0 CarrotOnNearPlate-v0
                 CarrotOnFarPlate-v0) ;;
  *) echo "TASK_SET must be simpler or harder, not '$TASK_SET'" >&2; exit 1 ;;
esac
# The VLA's visual prompts and VLM, as in the paper's SimplerEnv eval.
VP_ARGS="--use-sam3 --target-object-prompt-mode crosshair --target-location-prompt-mode box --use-vlm
         --vlm-query-on-gripper-change --vlm-decompose-subtasks"

# log MESSAGE: a line with the time, on stdout and in $LOG.
log() { echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }

# run_dir NAME: a run folder: NAME in $RUNS_DIR, or NAME itself when it is an absolute path.
run_dir() { case $1 in /*) echo "$1" ;; *) echo "$RUNS_DIR/$1" ;; esac; }

port_in_use() { ss -ltn | grep -q ":$1 "; }

# wait_port PORT [SECONDS]: until a server listens on PORT (default: up to 900 s).
wait_port() {
  local t=0
  while [ $t -lt "${2:-900}" ]; do port_in_use "$1" && return 0; sleep 3; t=$((t + 3)); done
  return 1
}

# layout_args LAYOUTS: the sim options of a layout set: paper = SimplerEnv's 24 fixed layouts per task (the test
# set), heldout = --layout-seed 1 (the training set), seedN = --layout-seed N (seed2: the held-out test set),
# iidN = a random layout of its own for each episode from layout seed N (iid11: the two-model ASM's training set,
# iid12: its test set).
layout_args() {
  case $1 in
    paper) ;;
    heldout) echo "--layout-seed 1" ;;
    seed[0-9]*) echo "--layout-seed ${1#seed}" ;;
    iid[0-9]*) echo "--layout-seed ${1#iid} --layout-per-episode" ;;
    *) echo "LAYOUTS must be paper, heldout, seedN or iidN, not '$1'" >&2; return 1 ;;
  esac
}

# rgb_args LEVEL: the sim options of the RGB camera noise (camera_noise.py; noise seed 0, the default): none (no
# options, as before the noise existed), realistic or harsh.
rgb_args() {
  case $1 in
    none) ;;
    realistic | harsh) echo "--rgb-noise $1" ;;
    *) echo "RGB_NOISE must be none, realistic or harsh, not '$1'" >&2; return 1 ;;
  esac
}

# layout_episodes LAYOUTS [TASK]: episodes per task in the layout set (iidN: IID_EPISODES, default 100). The harder
# scenes' fixed layouts (paper): 96 per kitchen task, 32 per two-plate task (harder_scenes.episodes).
layout_episodes() {
  case $1 in
    paper)
      case $2 in
        Cluttered*-v0 | TwoStep*-v0) echo 96 ;;
        CarrotOn*Plate-v0) echo 32 ;;
        *) echo 24 ;;
      esac ;;
    iid[0-9]*) echo "${IID_EPISODES:-100}" ;;
    *) echo 48 ;;
  esac
}

# all_chunks LAYOUTS SIZE: TASK:START:END for every episode of the layout set, in chunks of SIZE, the tasks taking
# turns.
all_chunks() {
  local n s t max=0
  for t in "${TASKS[@]}"; do n=$(layout_episodes "$1" "$t"); [ "$n" -gt $max ] && max=$n; done
  for ((s = 0; s < max; s += $2)); do
    for t in "${TASKS[@]}"; do
      n=$(layout_episodes "$1" "$t")
      [ $s -lt "$n" ] && echo "$t:$s:$((s + $2 < n ? s + $2 : n))"
    done
  done
}

# fast_defaults: the fast settings of the VLA's servers and sims, for each of SAM3_ARGS, VLM_ARGS, POLICY_ARGS and
# SIM_ARGS that is not set; FAST=0 leaves them empty (the paper's settings). Checked on 2026-10-10 against the 5
# evals with the paper's settings on the test set: the same success rate (McNemar p = 0.69), 4.8x faster per GPU.
# The VLM is seeded per query, so its answers do not depend on the queries before (the closed-loop eval needs it).
# SAM 3.1 always runs under bf16 autocast and has no --fp16 or --compile, so its fast setting is --tf32 only.
fast_defaults() {
  if [ "${FAST:-1}" = 1 ]; then
    if [ "$SAM_VERSION" = sam3.1 ]; then : "${SAM3_ARGS=--tf32}"; else : "${SAM3_ARGS=--fp16 --tf32 --compile}"; fi
    : "${VLM_ARGS=--attn sdpa --seed-per-query --compile}" \
      "${POLICY_ARGS=--attn sdpa --compile}" "${SIM_ARGS=--sam3-batch-prompts --no-action-plots}"
  fi
}

# sim GPU TASK START END OUT ARGS...: episodes START..END-1 of TASK with the paper's SimplerEnv settings. OUT is
# replaced; it gets the videos, the planner records and sim.log. ARGS choose the controller and its servers.
# INSTRUCTIONS=NAME: the reworded instructions of instruction_sets.py instead of the task's own (default: none).
sim() {
  local gpu=$1 task=$2 start=$3 end=$4 out=$5
  shift 5
  local robot=widowx scene=bridge_table_1_v1 overlay=bridge_real_eval_1.png x=0.147 y=0.028 steps=120
  if [ "$task" = PutEggplantInBasketScene-v0 ]; then
    robot=widowx_sink_camera_setup scene=bridge_table_1_v2 overlay=bridge_sink.png x=0.127 y=0.06
  fi
  [ "$task" = TwoStepSpoonAndCarrot-v0 ] && steps=240  # two pick-and-places (harder_scenes.py)
  [ -n "$out" ] || return 1
  rm -rf "$out" && mkdir -p "$out" || return 1
  CUDA_VISIBLE_DEVICES=$gpu "$SIMPLER_PYTHON" examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py \
    --ckpt-path "$CKPT" --robot $robot --policy-setup widowx_bridge --control-freq 5 --sim-freq 500 \
    --max-episode-steps $steps --env-name "$task" --scene-name $scene \
    --rgb-overlay-path "$SIMPLERENV_PATH/ManiSkill2_real2sim/data/real_inpainting/$overlay" \
    --robot-init-x $x $x 1 --robot-init-y $y $y 1 --obj-variation-mode episode --obj-episode-range "$start" "$end" \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir "$out" ${INSTRUCTIONS:+--instruction-set "$INSTRUCTIONS"} "$@" > "$out/sim.log" 2>&1
}

# Servers. The runner sets SERVER_ENV=(CUDA_VISIBLE_DEVICES=GPU); start_mps changes it to the MPS daemon.
SERVER_PIDS=()
MPS_PIPE=""

# start_mps GPU LOGDIR: an NVIDIA MPS daemon of this run only (its own pipe folder, so no other process or user
# connects to it), so that the servers' kernels can run at the same time. The sims do not use it. Tested on
# 2026-10-14: the same videos, hidden states and records as without it; VLA chunks 1.16x faster, the closed-loop
# eval 1.22x.
start_mps() {
  local uuid
  MPS_PIPE=$(mktemp -d /tmp/vpvla_mps_XXXXXX) || return 1
  mkdir -p "$2"
  MPS_ENV=(CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$2")
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1")
  env CUDA_VISIBLE_DEVICES="$uuid" "${MPS_ENV[@]}" nvidia-cuda-mps-control -d || return 1
  SERVER_ENV=(CUDA_VISIBLE_DEVICES=0 "${MPS_ENV[@]}")  # the daemon shows only this GPU, as device 0
}

# start_server PORT LOGFILE COMMAND...: a server in the background. It stops if PORT is already taken, because
# wait_port would then pass with somebody else's server.
start_server() {
  local port=$1 logfile=$2
  shift 2
  if port_in_use "$port"; then log "port $port is already in use"; return 1; fi
  env "${SERVER_ENV[@]}" "$@" > "$logfile" 2>&1 &
  SERVER_PIDS+=($!)
}

# start_sam_server PORT LOGFILE ARGS...: a SAM server of SAM_VERSION in the background.
start_sam_server() {
  local port=$1 logfile=$2
  shift 2
  start_server $port "$logfile" "$SAM_PYTHON" examples/Robocasa_tabletop/visual_prompt_utility/sam3_server.py \
    --port $port --sam-version $SAM_VERSION "$@"
}

# start_vla_servers PORT_BASE DIR: the VLA's servers, logs in DIR: policy (PORT_BASE, POLICY_ARGS), SAM (+1,
# SAM_VERSION, SAM3_ARGS) and VLM (+2, VLM_ARGS). The VLM server is in the sam3 env.
start_vla_servers() {
  local pb=$1 dir=$2
  start_server $pb "$dir/policy.log" "$STARVLA_PYTHON" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $pb --use_bf16 $POLICY_ARGS &&
  start_sam_server $((pb + 1)) "$dir/sam3.log" $SAM3_ARGS &&
  start_server $((pb + 2)) "$dir/vlm.log" "$SAM3_PYTHON" examples/SimplerEnv/eval_files/vlm_server_subtask_simpler.py \
    --port $((pb + 2)) $VLM_ARGS
}

# stop_servers: the servers, then the MPS daemon. Runners call it before their last log line and on exit.
stop_servers() {
  if [ ${#SERVER_PIDS[@]} -gt 0 ]; then
    kill "${SERVER_PIDS[@]}" 2>/dev/null
    wait "${SERVER_PIDS[@]}" 2>/dev/null
    SERVER_PIDS=()
  fi
  if [ -n "$MPS_PIPE" ]; then
    echo quit | env "${MPS_ENV[@]}" nvidia-cuda-mps-control
    sleep 2
    rm -rf "$MPS_PIPE"
    MPS_PIPE=""
  fi
}

# run_pool K DELAY FUNCTION ITEM...: FUNCTION ITEM for every item in the background, K at a time, DELAY seconds
# apart (so they do not all start at once); returns when all are done.
run_pool() {
  local k=$1 delay=$2 fn=$3 base item pids=()
  shift 3
  base=$(jobs -rp | wc -l)  # the servers and monitors that already run
  for item in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge $((base + k)) ]; do wait -n; done
    "$fn" "$item" &
    pids+=($!)
    sleep "$delay"
  done
  wait "${pids[@]}"
}
