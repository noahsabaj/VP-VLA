"""The VLM router: a baseline in which the VLM, not the ASM, chooses the method for each episode.

A router built on a vision-language model alone gives each episode to a method from the instruction and the first
camera frame, without the planner's first look. Here the VLM gets the first frame, the instruction and a description
of the two methods, and is asked once per method whether that method will complete the instruction. Its P(yes) (from
its first answer token, no sampling, as in the plan check) is the method's score, and the episode goes to the planner
when the planner's score is at least the VLA's (`choose`): the two-model ASM's rule, with the VLM's answers in place of
the two success models.

Two versions of the question (VERSIONS), each one more pair of questions and features:

* described: the two methods as their designers would describe them, with the depth camera's noise;
* rates: the same, and each method's successes on this task in the training set (TRAINING_SUCCESSES), as a router
  that remembers past episodes would know them.

The planner-only eval asks them on each first frame (planner_client.py, --vlm-router-port); the planner's episodes
stay the same. The VLM-router study scores them offline (experiments/score_vlm_router.py). Needs only numpy; the VLM
server answers (vlm_server_subtask_simpler.py, message "yes_no").
"""

from typing import Optional, Tuple

import numpy as np

METHODS = ("planner", "vla")
VERSIONS = ("described", "rates")
NAMES = {"planner": "motion planner", "vla": "VLA"}
# Each method's successes in the 100 training episodes of each task (iid11 with harsh depth and SAM 3.1:
# pseed31_iid11_harsh and vla31_iid11), fixed with the VLM-router study's plan on 2026-10-31.
TRAINING_EPISODES = 100
TRAINING_SUCCESSES = {
    "PutSpoonOnTableClothInScene-v0": {"planner": 54, "vla": 74},
    "PutCarrotOnPlateInScene-v0": {"planner": 51, "vla": 63},
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": {"planner": 54, "vla": 22},
    "PutEggplantInBasketScene-v0": {"planner": 46, "vla": 78},
}
DESCRIPTION = (
    "Two methods can control the robot arm. One of them will do the whole task alone, from the start:\n"
    "- The motion planner finds the object and the target in the camera picture, measures where they are with a "
    "depth camera, and plans a path that grasps the object and puts it on the target. It is precise when the depth "
    "is right, but this depth camera is very noisy, and a depth error can make it miss the object or the target.\n"
    "- The VLA is a neural network trained on many robot demonstrations. It turns the camera picture and the "
    "instruction directly into arm motions and does not use depth. Its motions are less precise, and it often fails "
    "when the object lies at an unusual angle."
)


def _check(method: str, version: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, not {version!r}")


def feature(method: str, version: str) -> str:
    """The record feature with the VLM's P(yes) for this method and version: vlm_router_{method}_yes for the
    described version, vlm_router_{method}_yes_rates for the rates version."""
    _check(method, version)
    return f"vlm_router_{method}_yes" + ("" if version == "described" else f"_{version}")


def question(instruction: str, method: str, version: str, task: Optional[str] = None) -> str:
    """The question the VLM gets with the first frame. The rates version needs the task (its training successes)."""
    _check(method, version)
    rates = ""
    if version == "rates":
        if task not in TRAINING_SUCCESSES:
            raise ValueError(f"the rates version knows SimplerEnv's 4 tasks only, not {task!r}")
        n = TRAINING_SUCCESSES[task]
        rates = (f"In {TRAINING_EPISODES} earlier tries of this task, the motion planner completed it {n['planner']} "
                 f"times and the VLA {n['vla']} times.\n")
    return (f'The robot is told: "{instruction}"\n'
            "The image is the robot's camera view before it moves.\n"
            f"{DESCRIPTION}\n{rates}"
            f"Will the {NAMES[method]} complete the instruction? Answer yes or no.")


def scores(record: dict, version: str) -> Tuple[float, float]:
    """The VLM's P(yes) for the planner and for the VLA in a planner record, in this version."""
    f = record.get("features", {})
    names = [feature(m, version) for m in METHODS]
    if any(f.get(n) is None for n in names):
        raise ValueError(f"no {' or '.join(names)} in this record (run the planner with --vlm-router-port)")
    return float(f[names[0]]), float(f[names[1]])


def choose(p_planner, p_vla) -> np.ndarray:
    """True (the planner) where the planner's P(yes) is at least the VLA's, as the two-model ASM breaks a tie."""
    return np.asarray(p_planner, float) >= np.asarray(p_vla, float)
