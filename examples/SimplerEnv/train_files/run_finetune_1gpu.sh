#!/bin/bash
# Fine-tunes the released VP-VLA SimplerEnv checkpoint on one 48 GB GPU with VP-VLA's own training code
# (starVLA/training/train_starvla_visual_prompt.py), for the planner-VLA training study (experiments README,
# "Fine-tuning"):
#
#   DATA_ROOT=... VP_DIR=... [GPU=0] bash examples/SimplerEnv/train_files/run_finetune_1gpu.sh RUN_ID DATA_MIX
#
# DATA_MIX: a mixture in starVLA/dataloader/gr00t_lerobot/mixtures.py (the study's: vla_p, vla_pl, vla_plt, vla_pltd,
# vla_s, and the learning curve's vla_p450 and vla_p1125). Its sets are in DATA_ROOT and their visual prompts in VP_DIR
# (select_demos.py --write SET --data-root DATA_ROOT --visual-prompts VP_DIR, or planner_demos_to_lerobot.py --out
# DATA_ROOT/SET --visual-prompts VP_DIR). The run goes to RUN_ROOT/RUN_ID in the layout that the evals read:
# config.yaml, dataset_statistics.json and checkpoints/steps_N_pytorch_model.pt (the evals' CKPT).
#
# The model and every training setting not named here are the released run's (CONFIG: the config.yaml of CKPT's
# run): QwenOFT (Qwen3-VL-4B and an MLP action head on the VLM's action tokens), the camera picture with and without
# the visual prompts, the learning rates (action head 1e-4, VLM 1e-5) with their warm-up and cosine decay, and the
# visual-prompt co-training loss (weight 0.1). Changed for one GPU:
# - the vision encoder, the word embeddings (tied to the output layer) and the lowest FROZEN_LAYERS (24) of the 36
#   language-model layers are frozen; the upper layers and the action head train (1.28 of 4.50 billion weights);
# - one process (DeepSpeed ZeRO-2, bf16) with BATCH (16) samples per step, against 8 GPUs x 32 in the released run;
# - MAX_STEPS (20000) steps, WARMUP (1000) of them warm-up, a checkpoint every SAVE_INTERVAL (MAX_STEPS) steps;
# - the co-training examples come from the training batch (use_inline_vp), as the recorded sets have no extracted
#   frames.
# The checkpoint loads module by module with strict key checks (reload_modules), so a model that does not match it
# stops the run. Also: CKPT (as in the eval runners), RUN_ROOT (playground/Checkpoints), BASE_VLM (the Qwen3-VL folder;
# default CONFIG's), LR_ACTION and LR_VLM (default CONFIG's), WORKERS (4 data loader workers), PORT (29517, for the
# process group), DRY_RUN=1 (check and print the command, start nothing). wandb logs offline, in the run folder. The
# script refuses a GPU that another process uses (ALLOW_BUSY_GPU=1 starts anyway).
set -euo pipefail
USAGE="usage: DATA_ROOT=... VP_DIR=... bash $0 RUN_ID DATA_MIX"
RUN_ID=${1:?$USAGE}
DATA_MIX=${2:?$USAGE}
: "${DATA_ROOT:?$USAGE}"
: "${VP_DIR:?$USAGE}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$REPO/examples/eval_env.sh"
vpvla_require STARVLA_PYTHON
cd "$REPO"

CKPT=${CKPT:-$REPO/playground/Checkpoints/VP-VLA-OXE/checkpoints/steps_70000_pytorch_model.pt}
CONFIG=${CONFIG:-$(dirname "$(dirname "$CKPT")")/config.yaml}
RUN_ROOT=${RUN_ROOT:-$REPO/playground/Checkpoints}
GPU=${GPU:-0}
BATCH=${BATCH:-16}
FROZEN_LAYERS=${FROZEN_LAYERS:-24}
MAX_STEPS=${MAX_STEPS:-20000}
WARMUP=${WARMUP:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-$MAX_STEPS}
WORKERS=${WORKERS:-4}
PORT=${PORT:-29517}
fail() { echo "ERROR: $*" >&2; exit 1; }

ACCELERATE=$(dirname "$STARVLA_PYTHON")/accelerate
[ -x "$ACCELERATE" ] || fail "no accelerate next to STARVLA_PYTHON ($ACCELERATE)"
[ -f "$CKPT" ] || fail "no checkpoint $CKPT (CKPT)"
[ -f "$CONFIG" ] || fail "no training config $CONFIG (CONFIG)"
# The trainer saves only at multiples of SAVE_INTERVAL, so the last steps would be lost.
[ $((MAX_STEPS % SAVE_INTERVAL)) = 0 ] || fail "MAX_STEPS ($MAX_STEPS) must be a multiple of SAVE_INTERVAL ($SAVE_INTERVAL)"
[ "$FROZEN_LAYERS" -ge 0 ] && [ "$FROZEN_LAYERS" -le 36 ] || fail "FROZEN_LAYERS must be 0 to 36 (Qwen3-VL-4B has 36)"
RUN_DIR=$RUN_ROOT/$RUN_ID
[ -e "$RUN_DIR" ] && fail "$RUN_DIR exists: use another RUN_ID"

# The mixture's sets (mixtures.py imports nothing from starVLA, so it loads without the training packages).
SETS=$("$STARVLA_PYTHON" - "$DATA_MIX" <<'P'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("mixtures", "starVLA/dataloader/gr00t_lerobot/mixtures.py")
mixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixtures)
mix = mixtures.DATASET_NAMED_MIXTURES.get(sys.argv[1])
if mix is None:
    sys.exit(f"ERROR: no mixture {sys.argv[1]} in starVLA/dataloader/gr00t_lerobot/mixtures.py")
print(" ".join(name for name, _, _ in mix))
P
)
for s in $SETS; do
  [ -f "$DATA_ROOT/$s/meta/info.json" ] || fail "no training set $DATA_ROOT/$s (planner_demos_to_lerobot.py --out)"
  [ -d "$VP_DIR/$s" ] || fail "no visual prompts $VP_DIR/$s (planner_demos_to_lerobot.py --visual-prompts)"
done

LM=qwen_vl_interface.model.model.language_model
FREEZE="qwen_vl_interface.model.model.visual,$LM.embed_tokens"
for ((i = 0; i < FROZEN_LAYERS; i++)); do FREEZE+=",$LM.layers.$i"; done

CMD=("$ACCELERATE" launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 --main_process_port "$PORT"
  starVLA/training/train_starvla_visual_prompt.py --config_yaml "$CONFIG"
  --is_debug false --run_root_dir "$RUN_ROOT" --run_id "$RUN_ID" --wandb_entity null
  --use_inline_vp true
  --datasets.vla_data.dataset_py visual_prompt_datasets_inline_vp
  --datasets.vla_data.data_root_dir "$DATA_ROOT"
  --datasets.vla_data.visual_prompt_dir "$VP_DIR"
  --datasets.vla_data.data_mix "$DATA_MIX"
  --datasets.vla_data.per_device_batch_size "$BATCH"
  --datasets.vla_data.num_workers "$WORKERS"
  --trainer.pretrained_checkpoint "$CKPT"
  --trainer.reload_modules qwen_vl_interface,action_model
  --trainer.freeze_modules "$FREEZE"
  --trainer.max_train_steps "$MAX_STEPS"
  --trainer.num_warmup_steps "$WARMUP"
  --trainer.save_interval "$SAVE_INTERVAL")
[ -n "${BASE_VLM:-}" ] && CMD+=(--framework.qwenvl.base_vlm "$BASE_VLM")
[ -n "${LR_ACTION:-}" ] && CMD+=(--trainer.learning_rate.action_model "$LR_ACTION")
[ -n "${LR_VLM:-}" ] && CMD+=(--trainer.learning_rate.qwen_vl_interface "$LR_VLM")

echo "sets of $DATA_MIX: $SETS"
echo "CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

if [ "${ALLOW_BUSY_GPU:-0}" != 1 ] && [ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)" ]; then
  fail "GPU $GPU is in use (nvidia-smi); ALLOW_BUSY_GPU=1 starts anyway"
fi
mkdir -p "$RUN_DIR"
{
  echo "git $(git rev-parse HEAD)"
  echo "sets of $DATA_MIX: $SETS"
  echo "CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
} > "$RUN_DIR/launch.txt"
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
