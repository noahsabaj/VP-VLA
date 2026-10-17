"""The harder WidowX scenes (planner/experiments/README.md, "Harder-scene study"): more objects on the table of
SimplerEnv's spoon and carrot tasks, and instructions that the planner's fixed rule cannot read. This file says
what the scenes are (objects, layouts, goals, instructions) and needs no simulator; harder_scene_envs.py builds them
in ManiSkill2_real2sim.

Two scenes, on SimplerEnv's spoon and carrot table with its camera and robot:
- kitchen: the spoon, the towel, the carrot and the plate. Three tasks with the same layouts: episode N of each has
  the same objects in the same places. The one-step tasks keep SimplerEnv's own instructions, so only the two
  extra objects change; the two-step task needs both pick-and-places in one episode.
- two plates: the carrot and two identical plates, one in the row nearer the robot and one in the far row. The
  instruction names the target only by where it is. The two tasks have the same layouts with the target swapped.

Each object starts on one corner of a square grid at the centre of SimplerEnv's own layouts (its half-edges 0.075 m,
as in the spoon and carrot tasks, and 0.1 m, as in the cube task), so no two objects start on top of each other.
"""

from itertools import permutations
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

SPOON = "bridge_spoon_generated_modified"
TOWEL = "table_cloth_generated_shorter"
CARROT = "bridge_carrot_generated_modified"
PLATE = "bridge_plate_objaverse_larger"

CENTER = np.array([-0.16, 0.0])  # the centre of SimplerEnv's layouts (simulator frame; the robot is at x = 0.147)
HALF_EDGES = (0.075, 0.1)
# Corner order: the far row (smaller x, away from the robot), then the near row; in each row y = -h, then +h.
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
FAR, NEAR = (0, 1), (2, 3)
# The turns of SimplerEnv's own tasks (rad, about the vertical): spoon 0 or 90 deg, carrot 180 or -90 deg.
KITCHEN_YAWS = [(0.0, 0.0, np.pi, 0.0), (np.pi / 2, 0.0, -np.pi / 2, 0.0)]  # spoon, towel, carrot, plate
CARROT_YAWS = (np.pi, -np.pi / 2)


class Task(NamedTuple):
    scene: str
    objects: Tuple[str, ...]  # model ids: the object to move first, its target second (as SimplerEnv's tasks)
    names: Tuple[str, ...]  # one short name per object, for the episode stats
    goals: Tuple[Tuple[int, int, bool], ...]  # (object, target, all of it on the target); all must hold at the end
    logged: Tuple[Tuple[int, int, bool], ...]  # pairs only logged, not part of success
    instruction: str
    max_steps: int


STEPS = 120  # as every eval of SimplerEnv's 4 tasks in this repo; 120 per pick-and-place
TASKS: Dict[str, Task] = {
    "ClutteredSpoonOnTowel-v0": Task(
        "kitchen", (SPOON, TOWEL, CARROT, PLATE), ("spoon", "towel", "carrot", "plate"),
        ((0, 1, False),), (), "put the spoon on the towel", STEPS),
    "ClutteredCarrotOnPlate-v0": Task(
        "kitchen", (CARROT, PLATE, SPOON, TOWEL), ("carrot", "plate", "spoon", "towel"),
        ((0, 1, True),), (), "put carrot on plate", STEPS),
    "TwoStepSpoonAndCarrot-v0": Task(
        "kitchen", (SPOON, TOWEL, CARROT, PLATE), ("spoon", "towel", "carrot", "plate"),
        ((0, 1, False), (2, 3, True)), (), "put the spoon on the towel and the carrot on the plate", 2 * STEPS),
    "CarrotOnNearPlate-v0": Task(
        "two_plates", (CARROT, PLATE, PLATE), ("carrot", "plate", "other_plate"),
        ((0, 1, True),), ((0, 2, True),), "put the carrot on the plate that is closer to the robot", STEPS),
    "CarrotOnFarPlate-v0": Task(
        "two_plates", (CARROT, PLATE, PLATE), ("carrot", "plate", "other_plate"),
        ((0, 1, True),), ((0, 2, True),), "put the carrot on the plate that is farther from the robot", STEPS),
}
SHORT = dict(zip(TASKS, ["Cluttered spoon", "Cluttered carrot", "Two-step", "Near plate", "Far plate"], strict=True))
# The spoon goal leaves part of the spoon off the towel, as SimplerEnv's spoon task; the carrot must lie on the plate
# and touch nothing else but the robot, as SimplerEnv's carrot task.


def corners(half_edge: float) -> np.ndarray:
    return CENTER + SIGNS * half_edge


def quat(yaw: float) -> np.ndarray:
    """(w, x, y, z) of a turn about the vertical, as transforms3d's euler2quat(0, 0, yaw)."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def kitchen_layouts() -> List[np.ndarray]:
    """[48] x [4 objects (spoon, towel, carrot, plate), 2]: each grid, each way to put the 4 objects on its corners."""
    return [corners(h)[list(p)] for h in HALF_EDGES for p in permutations(range(4))]


def two_plate_layouts() -> List[np.ndarray]:
    """[16] x [3 (carrot, near plate, far plate), 2]: each grid, each corner of the near row and of the far row for
    the plates, and each corner left for the carrot."""
    out = []
    for h in HALF_EDGES:
        c = corners(h)
        for near in NEAR:
            for far in FAR:
                for carrot in (k for k in range(4) if k not in (near, far)):
                    out.append(c[[carrot, near, far]])
    return out


def layouts(task: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The task's xy configs ([objects, 2] each) and turn configs ([objects, 4] each), in the order of its objects.
    Episode N takes xy config N // 2 and turn config N % 2, as SimplerEnv's PutOnBridgeInSceneEnv indexes them."""
    t = TASKS[task]
    if t.scene == "kitchen":
        order = [(SPOON, TOWEL, CARROT, PLATE).index(o) for o in t.objects]
        xys = [xy[order] for xy in kitchen_layouts()]
        quats = [np.stack([quat(yaws[i]) for i in order]) for yaws in KITCHEN_YAWS]
    else:
        # the target is object 1: the near-row plate, or the far-row plate (then the plates swap places)
        swap = [0, 1, 2] if task == "CarrotOnNearPlate-v0" else [0, 2, 1]
        xys = [xy[swap] for xy in two_plate_layouts()]
        quats = [np.stack([quat(y), quat(0.0), quat(0.0)]) for y in CARROT_YAWS]
    return xys, quats


def episodes(task: str) -> int:
    xys, quats = layouts(task)
    return len(xys) * len(quats)
