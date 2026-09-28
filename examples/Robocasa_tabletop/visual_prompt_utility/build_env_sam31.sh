#!/bin/bash
# Build the sam31 conda env for `sam3_server.py --sam-version sam3.1`.
#
#   CONDA_ENVS_ROOT=/path/to/envs bash examples/Robocasa_tabletop/visual_prompt_utility/build_env_sam31.sh
#
# SAM 3.1 loads only through Meta's `sam3` package, which wants Python >= 3.12,
# torch >= 2.7, numpy < 2 and setuptools < 81 (it imports pkg_resources). The
# existing `sam3` env runs transformers 5.x with numpy 2, so SAM 3.1 gets its own
# env and the VLM server stays where it is.
#
# The checkpoint (facebook/sam3.1, sam3.1_multiplex.pt, 3.5 GB) is gated: accept
# the license on Hugging Face and log in (`hf auth login`) before step 4.
set -u

CONDA=${CONDA:-conda}
ENV_PREFIX=${SAM31_ENV_PREFIX:-${CONDA_ENVS_ROOT:-${HOME}/miniconda3/envs}/sam31}
SAM3_REPO=https://github.com/facebookresearch/sam3.git
SAM3_COMMIT=2345a4ad109ac29c569da749c91d84f10dc08c40  # main on 2026-09-18
SAM3_SRC=${SAM3_SRC:-${ENV_PREFIX}/src/sam3}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stage() { echo ""; echo "############ $* ############"; }
fail() { echo "!!! FAILED: $*"; exit 1; }

stage "1/5  create env (python 3.12) at ${ENV_PREFIX}"
if [ -d "${ENV_PREFIX}" ]; then echo "exists, reusing"; else
  ${CONDA} create -p "${ENV_PREFIX}" python=3.12 -y || fail "conda create"
fi
PY="${ENV_PREFIX}/bin/python"
${PY} -m pip install -q --upgrade pip wheel "setuptools<81" || fail "pip bootstrap"

stage "2/5  torch 2.10.0 (CUDA 12.8)"
${PY} -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 || fail torch

stage "3/5  Meta sam3 @ ${SAM3_COMMIT:0:7} + server deps"
if [ ! -d "${SAM3_SRC}/.git" ]; then git clone "${SAM3_REPO}" "${SAM3_SRC}" || fail "git clone"; fi
git -C "${SAM3_SRC}" fetch -q origin && git -C "${SAM3_SRC}" checkout -q "${SAM3_COMMIT}" || fail "checkout"
${PY} -m pip install -e "${SAM3_SRC}" || fail "sam3"
${PY} -m pip install -q websockets msgpack pillow opencv-python-headless || fail "server deps"

stage "4/5  download sam3.1_multiplex.pt (HF_HOME=${HF_HOME:-~/.cache/huggingface})"
${PY} -c "from huggingface_hub import hf_hub_download as d; print(d('facebook/sam3.1', 'sam3.1_multiplex.pt'))" \
  || fail "download (license accepted and logged in?)"

stage "5/5  smoke test: load SAM3.1 and find a red square by text"
cd "${HERE}" && ${PY} - <<'PYEOF'
import time, numpy as np, torch
from sam31_backend import SAM31Model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAM31Model(device=device)
if device == "cuda":
    print("  VRAM used: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

a = np.full((360, 360, 3), 128, np.uint8); a[120:240, 120:240] = [220, 30, 30]
res = model.segment(a, "a red square")
t0 = time.time(); res = model.segment(a, "a red square"); dt = time.time() - t0
print(f"  detections: {res['num_masks']}  ({dt*1000:.0f} ms per call on {device})")
if res["num_masks"]:
    ys, xs = np.nonzero(res["masks"][0])
    print(f"  top mask: {len(xs)} px, centroid=({xs.mean():.0f},{ys.mean():.0f}) [truth ~(180,180)], score {res['scores'][0]:.3f}")
    print(f"  box: {res['boxes'][0].round().tolist()} [truth ~(120,120,240,240)]")
    print("  SAM3.1 SMOKE TEST: PASS")
else:
    print("  SAM3.1 SMOKE TEST: no detection (check the load warnings above)")
PYEOF
