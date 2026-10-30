"""The run folders of the experiments: one folder per eval, all in $RUNS_DIR.

Planner-only evals are named {prefix}_{layouts}_{level}: layouts = heldout (the training set, layout seed 1),
paper (the test set, SimplerEnv's 24 layouts per task), seed2 (the held-out test set, layout seed 2), or iid11 and
iid12 (the two-model ASM study's training and test sets: a random layout of its own for each episode), and
level = the depth noise (none, realistic, harsh). The VLA-only evals are named below. The camera-noise study's
evals are cam_{layouts}_{level} (planner-only) and vla_{layouts}_{level} (VLA-only), level = the camera's (RGB noise
for both, and the depth noise for the planner); its ASMs are cam_model/asm_{level}.json. The reworded-instruction
study's evals are words_{set}_paper (planner-only) and vla_words_{set}_paper (VLA-only); the harder-scene study's
are harder_paper and vla_harder_paper. The VLM plan-check study adds check_paper_none (planner-only, with the plan
check). The VLM plan-check v2 study's are named in score_plan_check_v2 (check2_*, and vla_*_iid13). The evals with
SAM 3.1 are pseed31_{layouts}_harsh (planner-only, the fixed rule) and vla31_{layouts} (VLA-only); the VLM-plan
study adds vlmplan_{layouts}_{level} (planner-only, the VLM plan) and closed_loop_vlmplan_iid15, with the ASMs
vlmplan_model/asm_{level}.json, and the role split's closed-loop eval handoff_vlmplan_iid15 (score_vlm_plan). The
difficulty sweep's depth levels harsh_xS (harsh's errors times S) give vlmplan_iid15_harsh_xS (score_sweep); the
missing-depth study's evals are miss_{layouts}_none (score_missing_depth); the harder-task system study's are named
in score_harder_system (vlmplan_* and pseed31_* of each instruction group, with harsh depth). The energy study's
energy runs are named in score_energy (energy_*, each with its null episodes in energy_*_null), with the ASMs
energy_model/asm_{level}.json.
"""

import os
from typing import Iterable, List

from .. import selector as s

LEVELS = ["none", "realistic", "harsh"]
LEVEL_NAMES = {"none": "clean", "realistic": "realistic", "harsh": "harsh"}
SHORT = dict(zip(s.TASKS, ["Spoon", "Carrot", "Stack", "Eggplant"], strict=True))
# Every selector feature except plan_time, which is wall-clock time and differs on every run.
CHECK = [n for n in s.FEATURES_V1 + [n for n in s.FEATURES_V2 if n not in s.FEATURES_V1] if n != "plan_time"]

# VLA-only evals with the fast settings (vla_eval.sh), one per layout set; the training set and test set ones are
# split over two folders (one per GPU). The reruns with VLM seed offsets 1 and 2 are rep1_{set} and rep2_{set}.
FAST_VLA = {"heldout": ["heldout_vla_g0", "heldout_vla_g1"], "paper": ["fast3_train_g0", "fast3_train_g1"],
            "seed2": ["seed2_vla"], "iid11": ["vla_iid11"], "iid12": ["vla_iid12"]}
# The test set with SimplerEnv's own instructions and SAM 3.1: planner-only (planner seed 0, clean camera) and
# VLA-only (fast settings, VLM seed offset 0). The reworded-instruction, harder-scene and VLM plan-check studies use
# them since their changes of 2026-10-23, instead of pseed_paper_none and FAST_VLA["paper"] (SAM 3).
OWN31_PLANNER, OWN31_VLA = "check_paper_none", ["sam31_paper"]
# The 5 VLA-only evals of the test set with the paper's settings (links to where they were written).
EXACT_VLA = [f"exact_vla_{i}" for i in range(1, 6)]
# The VLA's first-step files with the action spread (vla_eval.sh with HIDDEN=1 PERTURB=8): those of the two-model
# study's VLA-only evals, and of 1-step episodes of the test set.
PERTURB = {"iid11": "vla_iid11/hidden", "iid12": "vla_iid12/hidden", "paper": "perturb_paper/hidden"}
# The two-model study's ASMs (selector.py --rule two-model, trained on iid11), {} for the depth level.
TWO_MODEL = "two_model/asm_{}.json"


def path(name: str) -> str:
    """A run folder: `name` in $RUNS_DIR, or `name` itself when it is an absolute path."""
    if os.path.isabs(name):
        return name
    root = os.environ.get("RUNS_DIR")
    if not root:
        raise SystemExit("set RUNS_DIR to the folder with the run folders")
    return os.path.join(root, name)


def paths(names: Iterable[str]) -> List[str]:
    return [path(n) for n in names]
