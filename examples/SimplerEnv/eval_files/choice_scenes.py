"""The choice scenes (planner/experiments/README.md, "Instruction study"): one scene, the same layouts, and four
instructions that pair each of its two objects with each of its two targets. Two of the pairings are in the training
sets of this repo (the spoon on the towel, the carrot on the plate) and two are in none (the spoon on the plate, the
carrot on the towel), so a controller that solves all four at an episode index does what the instruction says, and one
that is tied to the trained pairings does not.

The scene is the kitchen scene of harder_scenes.py (the spoon, the towel, the carrot and the plate) with its random
layouts: episode N of a layout seed has the same objects in the same places as episode N of the kitchen tasks, in
all four tasks. There are no fixed layouts. Like harder_scenes.py, this file says what the scenes are and needs no
simulator; harder_scene_envs.py builds them in ManiSkill2_real2sim.
"""

from typing import Dict, Tuple

import numpy as np

from . import harder_scenes
from .harder_scenes import CARROT, PLATE, SPOON, STEPS, TOWEL, Task

# A kitchen task of harder_scenes.py: all of them have the same layout at a layout seed and episode, so any one gives
# the scene's layout, in the order of its objects.
KITCHEN_TASK = "ClutteredSpoonOnTowel-v0"

# Each task: the object to move, its target, then the other two objects. The goal is checked as the kitchen scene's
# own: the spoon as SimplerEnv's spoon goal (part of it may lie off the target), the carrot as SimplerEnv's carrot
# goal (all of it on the target, touching nothing else but the robot). The logged pair, which is not part of success,
# is the same object on the other target with the same check: on a new pairing it says whether the object ended on
# the target that it was trained with.
TASKS: Dict[str, Task] = {
    "ChoiceSpoonOnTowel-v0": Task(
        "kitchen", (SPOON, TOWEL, CARROT, PLATE), ("spoon", "towel", "carrot", "plate"),
        ((0, 1, False),), ((0, 3, False),), "put the spoon on the towel", STEPS),
    "ChoiceCarrotOnPlate-v0": Task(
        "kitchen", (CARROT, PLATE, SPOON, TOWEL), ("carrot", "plate", "spoon", "towel"),
        ((0, 1, True),), ((0, 3, True),), "put carrot on plate", STEPS),
    "ChoiceSpoonOnPlate-v0": Task(
        "kitchen", (SPOON, PLATE, TOWEL, CARROT), ("spoon", "plate", "towel", "carrot"),
        ((0, 1, False),), ((0, 2, False),), "put the spoon on the plate", STEPS),
    "ChoiceCarrotOnTowel-v0": Task(
        "kitchen", (CARROT, TOWEL, SPOON, PLATE), ("carrot", "towel", "spoon", "plate"),
        ((0, 1, True),), ((0, 3, True),), "put the carrot on the towel", STEPS),
}
# The pairings that the training sets of this repo have (SimplerEnv's spoon and carrot tasks, and the kitchen
# scene's), and the two that none has.
TRAINED = ("ChoiceSpoonOnTowel-v0", "ChoiceCarrotOnPlate-v0")
NEW = ("ChoiceSpoonOnPlate-v0", "ChoiceCarrotOnTowel-v0")
# Each object's two tasks: its trained pairing, then its new one.
OBJECT_TASKS = {"spoon": ("ChoiceSpoonOnTowel-v0", "ChoiceSpoonOnPlate-v0"),
                "carrot": ("ChoiceCarrotOnPlate-v0", "ChoiceCarrotOnTowel-v0")}
SHORT = dict(zip(TASKS, ["Spoon on towel", "Carrot on plate", "Spoon on plate", "Carrot on towel"], strict=True))


def random_layout(task: str, seed: int, episode: int) -> Tuple[np.ndarray, np.ndarray]:
    """The kitchen scene's random layout for an episode of a layout seed (harder_scenes.random_layout): ([objects, 2]
    xy, [objects, 4] turns), in the order of the task's objects. The four tasks, and the kitchen tasks of
    harder_scenes.py, share it."""
    xy, quats = harder_scenes.random_layout(KITCHEN_TASK, seed, episode)
    kitchen = harder_scenes.TASKS[KITCHEN_TASK].objects
    order = [kitchen.index(o) for o in TASKS[task].objects]
    return xy[order], quats[order]
