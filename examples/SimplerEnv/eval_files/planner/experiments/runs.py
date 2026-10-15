"""The run folders of the experiments: one folder per eval, all in $RUNS_DIR.

Planner-only evals are named {prefix}_{layouts}_{level}: layouts = heldout (the training set, layout seed 1),
paper (the test set, SimplerEnv's 24 layouts per task) or seed2 (the held-out test set, layout seed 2), and
level = the depth noise (none, realistic, harsh). The VLA-only evals are named below.
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
            "seed2": ["seed2_vla"]}
# The 5 VLA-only evals of the test set with the paper's settings (links to where they were written).
EXACT_VLA = [f"exact_vla_{i}" for i in range(1, 6)]


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
