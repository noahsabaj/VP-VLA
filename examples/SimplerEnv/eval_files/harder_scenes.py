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
  The instructions say "camera", not "robot": the robot's base is below the camera picture, but its arm comes in
  from the top, so "closer to the robot" could be read as the far row. The near row is nearer both.

Each object starts on one corner of a square grid at the centre of SimplerEnv's own layouts (its half-edges 0.075 m,
as in the spoon and carrot tasks, and 0.1 m, as in the cube task), so no two objects start on top of each other.
These fixed layouts are the `paper` layout set. A layout seed gives random ones instead, one per episode
(`random_layout`, the iidN layout sets): the same grid with a random half-edge, each object moved a little off its
corner, and the spoon and the carrot turned at random.
"""

from itertools import permutations
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

SPOON = "bridge_spoon_generated_modified"
TOWEL = "table_cloth_generated_shorter"
CARROT = "bridge_carrot_generated_modified"
PLATE = "bridge_plate_objaverse_larger"

CENTER = np.array([-0.16, 0.0])  # the centre of SimplerEnv's layouts (simulator frame; the robot is at x = 0.147)
# Where the camera is (simulator frame, xy): ManiSkill2_real2sim's WidowX 3rd_view_camera, 0.16 m to the side of
# the robot's base, which is at (0.147, 0.028) and turned 180 deg.
CAMERA_XY = np.array([0.147, 0.188])
HALF_EDGES = (0.075, 0.1)
# Corner order: the far row (smaller x, away from the robot), then the near row; in each row y = -h, then +h.
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
FAR, NEAR = (0, 1), (2, 3)
# The turns of SimplerEnv's own tasks (rad, about the vertical): spoon 0 or 90 deg, carrot 180 or -90 deg.
KITCHEN_YAWS = [(0.0, 0.0, np.pi, 0.0), (np.pi / 2, 0.0, -np.pi / 2, 0.0)]  # spoon, towel, carrot, plate
CARROT_YAWS = (np.pi, -np.pi / 2)
# Length and width on the table (m), from ManiSkill2_real2sim's data/custom/info_bridge_custom_v0.json.
FOOTPRINT = {SPOON: (0.14, 0.035), TOWEL: (0.1, 0.1), CARROT: (0.115, 0.032), PLATE: (0.11, 0.11)}
# Random layouts: the grid's half-edge range (no corner object leaves the cube task's area, SimplerEnv's widest),
# how far each object moves off its corner (in x and in y), and the gap kept between any two objects.
RANDOM_HALF_EDGES = (0.075, 0.09)
JITTER = 0.01
MIN_GAP = 0.02


class Task(NamedTuple):
    scene: str
    objects: Tuple[str, ...]  # model ids: the object to move first, its target second (as SimplerEnv's tasks)
    names: Tuple[str, ...]  # one short name per object, for the episode stats
    goals: Tuple[Tuple[int, int, bool], ...]  # (object, target, all of it on the target); all must hold at the end
    logged: Tuple[Tuple[int, int, bool], ...]  # pairs only logged, not part of success
    instruction: str
    max_steps: int
    z_offset: float = 0.02  # how far (m) the object's bottom may be above the target's top, as SimplerEnv's evaluate


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
        ((0, 1, True),), ((0, 2, True),), "put the carrot on the plate that is closer to the camera", STEPS),
    "CarrotOnFarPlate-v0": Task(
        "two_plates", (CARROT, PLATE, PLATE), ("carrot", "plate", "other_plate"),
        ((0, 1, True),), ((0, 2, True),), "put the carrot on the plate that is farther from the camera", STEPS),
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


def fixed_layout_group(task: str, episode: int) -> int:
    """The xy config of a fixed layout (layouts): the episodes of one xy config differ only in the turns."""
    return episode // len(KITCHEN_YAWS if TASKS[task].scene == "kitchen" else CARROT_YAWS)


def extent(model: str, yaw: float, sizes: Dict[str, Tuple[float, float]] = FOOTPRINT) -> np.ndarray:
    """The x and y size (m) of the box around the object's footprint (`sizes`) when it is turned by yaw."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return np.array([[c, s], [s, c]]) @ sizes[model]


def apart(models, xys, yaws, gap: float = MIN_GAP, sizes: Dict[str, Tuple[float, float]] = FOOTPRINT) -> bool:
    """Is every two objects' gap at least `gap` along x or along y (their footprints' boxes)?"""
    ext = [extent(m, y, sizes) for m, y in zip(models, yaws, strict=True)]
    return all((np.abs(xys[i] - xys[j]) - (ext[i] + ext[j]) / 2).max() >= gap
               for i in range(len(models)) for j in range(i))


def random_layout(task: str, seed: int, episode: int) -> Tuple[np.ndarray, np.ndarray]:
    """One random layout for an episode of a layout seed: ([objects, 2] xy, [objects, 4] turns), in the order of the
    task's objects. The objects take the grid's corners as in the fixed layouts (kitchen: in a random order; two
    plates: one plate in each row, the carrot on a corner left), on a grid with a random half-edge in
    RANDOM_HALF_EDGES; each moves up to JITTER off its corner in x and y, and the spoon and the carrot get a random
    turn. Drawn again until every two objects are MIN_GAP apart (and the near-row plate is at least 1 cm nearer the
    camera than the other). The generator is seeded with (seed, episode), so the tasks of a scene share the layout
    (the two plate tasks with the target swapped), and an episode's layout does not depend on the others."""
    t = TASKS[task]
    rng = np.random.default_rng([seed, episode])
    for _ in range(1000):
        c = corners(rng.uniform(*RANDOM_HALF_EDGES))
        if t.scene == "kitchen":
            models = (SPOON, TOWEL, CARROT, PLATE)
            xy = c[rng.permutation(4)]
            yaws = np.array([rng.uniform(-np.pi, np.pi), 0.0, rng.uniform(-np.pi, np.pi), 0.0])
        else:
            models = (CARROT, PLATE, PLATE)  # the carrot, the near-row plate, the far-row plate
            near, far = int(rng.choice(NEAR)), int(rng.choice(FAR))
            carrot = int(rng.choice([k for k in range(4) if k not in (near, far)]))
            xy = c[[carrot, near, far]]
            yaws = np.array([rng.uniform(-np.pi, np.pi), 0.0, 0.0])
        xy = xy + rng.uniform(-JITTER, JITTER, size=xy.shape)
        if not apart(models, xy, yaws):
            continue
        if t.scene == "two_plates" and np.linalg.norm(xy[2] - CAMERA_XY) - np.linalg.norm(xy[1] - CAMERA_XY) < 0.01:
            continue
        break
    else:
        raise RuntimeError(f"{task}: no layout found for seed {seed}, episode {episode}")
    if t.scene == "kitchen":
        order = [models.index(o) for o in t.objects]
    else:  # the target is object 1: the near-row plate, or the far-row plate (then the plates swap places)
        order = [0, 1, 2] if task == "CarrotOnNearPlate-v0" else [0, 2, 1]
    return xy[order], np.stack([quat(y) for y in yaws[order]])
