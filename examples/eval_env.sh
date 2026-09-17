#!/bin/bash
# Shared environment resolution for VP-VLA evaluation scripts.
#
# Every value below is overridable from the environment, so the same scripts run
# unmodified locally and on a cluster:
#
#   CONDA_ENVS_ROOT=/shared/envs SIMPLERENV_PATH=/scratch/SimplerEnv \
#     bash examples/SimplerEnv/eval_files/auto_eval_scripts/run_eval.sh ckpt.pt
#
# Resolution order: existing environment value > default below.

: "${CONDA_ENVS_ROOT:=${HOME}/miniconda3/envs}"

: "${STARVLA_PYTHON:=${CONDA_ENVS_ROOT}/starVLA/bin/python}"
: "${SAM3_PYTHON:=${CONDA_ENVS_ROOT}/sam3/bin/python}"
: "${SIMPLER_PYTHON:=${CONDA_ENVS_ROOT}/simpler_env/bin/python}"
: "${ROBOCASA_PYTHON:=${CONDA_ENVS_ROOT}/robocasa/bin/python}"

: "${SIMPLERENV_PATH:=${HOME}/Development/SimplerEnv}"

# GPU count: honour CUDA_VISIBLE_DEVICES when set, else ask the driver.
if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _vpvla_devs <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS=${#_vpvla_devs[@]}
  else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
  fi
fi
[ "${NUM_GPUS:-0}" -ge 1 ] 2>/dev/null || NUM_GPUS=1
export NUM_GPUS

# Fail fast with a readable message instead of dying 30s into server startup.
vpvla_require() {
  local ok=1 v p
  for v in "$@"; do
    p="${!v}"
    if [ ! -x "${p}" ]; then
      echo "ERROR: \$${v} -> '${p}' is not an executable python" >&2
      ok=0
    fi
  done
  if [ "${ok}" -ne 1 ]; then
    echo "" >&2
    echo "Override via environment, e.g.:" >&2
    echo "  CONDA_ENVS_ROOT=/path/to/conda/envs bash \$0 ..." >&2
    exit 1
  fi
}

vpvla_require_dir() {
  local v p
  for v in "$@"; do
    p="${!v}"
    [ -d "${p}" ] || { echo "ERROR: \$${v} -> '${p}' is not a directory" >&2; exit 1; }
  done
}

vpvla_banner() {
  echo "--- VP-VLA eval configuration ---"
  echo "  NUM_GPUS         : ${NUM_GPUS}"
  echo "  CUDA_VISIBLE_DEV : ${CUDA_VISIBLE_DEVICES:-<unset>}"
  for v in "$@"; do echo "  $(printf '%-17s' "${v}"): ${!v}"; done
  echo "  git HEAD         : $(git rev-parse --short HEAD 2>/dev/null || echo '<not a repo>')"
  echo "---------------------------------"
}
