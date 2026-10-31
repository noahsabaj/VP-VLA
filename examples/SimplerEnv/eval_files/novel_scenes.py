"""The new test scenes (T5 of the marriage study; planner/experiments/README.md, "Marriage study"): eight WidowX tasks
that no model trains on, two of each kind (KINDS):
- new objects: a sponge and a soda can from SimplerEnv's Google-robot objects, which no training scene has;
- new pairs: an object and a target that no training task pairs, one of them in the eggplant task's sink scene;
- reasoning in clutter: the object to move is named by what it is not, or by which other object it is next to;
- long tasks: two and three pick-and-places in one instruction.

Like harder_scenes.py, this file says what the scenes are (objects, layouts, goals, instructions) and needs no
simulator; harder_scene_envs.py builds them in ManiSkill2_real2sim. The table scenes use the spoon and carrot task's
table, camera and robot; the sink scene uses the eggplant task's. The layouts are random only, one per episode
(random_layout; the iidN layout sets), drawn as the harder scenes' are: each object on one corner of a square grid at
the centre of SimplerEnv's layouts, moved a little off it and turned at random.
"""

import zlib
from typing import Dict, Tuple

import numpy as np

from .harder_scenes import (CARROT, CENTER, FOOTPRINT, JITTER, PLATE, RANDOM_HALF_EDGES, SPOON, STEPS, TOWEL, Task,
                            apart, corners, quat)

SPONGE = "sponge"
CAN = "opened_coke_can"  # opened: a light can (SimplerEnv's full cans weigh 0.4 kg)
EGGPLANT = "eggplant"
GREEN = "green_cube_3cm"
YELLOW = "yellow_cube_3cm"
BASKET = "dummy_sink_target_plane"  # the sink scene's yellow basket: an invisible plane in it, as in the eggplant task
# Models from SimplerEnv's Google-robot objects (data/custom/info_pick_custom_v0.json), not from its Bridge ones.
PICK_MODELS = (SPONGE, CAN)
# Length and width on the table (m) when not turned, from ManiSkill2_real2sim's model infos; the can stands upright.
SIZES = {**FOOTPRINT, SPONGE: (0.07, 0.1), CAN: (0.066, 0.066), EGGPLANT: (0.078, 0.037), GREEN: (0.03, 0.03),
         YELLOW: (0.03, 0.03)}
TURNED = (SPOON, CARROT, EGGPLANT, SPONGE, CAN, GREEN, YELLOW)  # the objects that get a random turn
# The can model lies along its y axis; this turn stands it up, as SimplerEnv's upright cans (euler (pi/2, 0, 0)).
UPRIGHT = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
# The three-step scene: four objects on the corners of a grid with a half-edge of WIDE, the two cubes side by side
# at its centre (DELTA from it in y), each moved up to WIDE_JITTER: no object leaves the cube task's area.
WIDE, WIDE_JITTER, DELTA = 0.095, 0.005, 0.035
# The sink scene, as SimplerEnv's eggplant task: the object starts near SINK_START, up to SINK_SPAN off it in x and y,
# turned by up to SINK_TURN; the basket's plane is at BASKET_XY.
SINK_START, SINK_SPAN, SINK_TURN = np.array([-0.105, 0.206]), np.array([0.01, 0.015]), np.pi / 4
BASKET_XY = np.array([-0.125, 0.025])
NEARER_BY = 0.02  # the carrot by the towel is at least this much nearer the towel than the other carrot (centres)

TASKS: Dict[str, Task] = {
    "NovelSpongeOnPlate-v0": Task(
        "table", (SPONGE, PLATE, CARROT, TOWEL), ("sponge", "plate", "carrot", "towel"),
        ((0, 1, True),), (), "put the sponge on the plate", STEPS),
    "NovelCanOnTowel-v0": Task(
        "table", (CAN, TOWEL, SPOON, PLATE), ("can", "towel", "spoon", "plate"),
        ((0, 1, False),), (), "put the soda can on the towel", STEPS),
    "NovelEggplantOnPlate-v0": Task(
        "table", (EGGPLANT, PLATE, CARROT, TOWEL), ("eggplant", "plate", "carrot", "towel"),
        ((0, 1, True),), (), "put the eggplant on the plate", STEPS),
    "NovelCarrotInBasket-v0": Task(
        "sink", (CARROT, BASKET), ("carrot", "basket"),
        ((0, 1, False),), (), "put the carrot into the yellow basket", STEPS, 0.06),
    "NovelNotOrangeOnTowel-v0": Task(
        "table", (EGGPLANT, TOWEL, CARROT, PLATE), ("eggplant", "towel", "carrot", "plate"),
        ((0, 1, False),), ((2, 1, False),), "put the vegetable that is not orange on the towel", STEPS),
    "NovelCarrotByTowelOnPlate-v0": Task(
        "table", (CARROT, PLATE, CARROT, TOWEL), ("carrot", "plate", "other_carrot", "towel"),
        ((0, 1, True),), ((2, 1, True),), "put the carrot that is closer to the towel on the plate", STEPS),
    "NovelStackOnPlate-v0": Task(
        "table", (YELLOW, PLATE, GREEN, CARROT), ("yellow_cube", "plate", "green_cube", "carrot"),
        ((0, 1, False), (2, 0, False)), (), "put the yellow cube on the plate, then stack the green cube on it",
        2 * STEPS),
    "NovelThreeStep-v0": Task(
        "table", (SPOON, TOWEL, CARROT, PLATE, GREEN, YELLOW),
        ("spoon", "towel", "carrot", "plate", "green_cube", "yellow_cube"),
        ((0, 1, False), (2, 3, True), (4, 5, False)), (),
        "put the spoon on the towel, the carrot on the plate, and the green cube on the yellow cube", 3 * STEPS),
}
KINDS = dict(zip(TASKS, ["new object"] * 2 + ["new pair"] * 2 + ["reasoning"] * 2 + ["long"] * 2, strict=True))
SHORT = dict(zip(TASKS, ["Sponge", "Soda can", "Eggplant on plate", "Carrot in basket", "Not orange",
                         "Carrot by towel", "Stack on plate", "Three steps"], strict=True))
# The goals follow SimplerEnv's own: an object that must lie on its target and touch nothing else (True) as its carrot
# task, the spoon and the cubes only on their target as its spoon and cube tasks, and the basket as its eggplant task
# (the plane in the basket, the object up to 6 cm above it). The logged pairs (the wrong object on the target) are
# not part of success.


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The product of two (w, x, y, z) quaternions: turn b, then a."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def turn(model: str, yaw: float) -> np.ndarray:
    """The object's start turn: by yaw about the vertical (the can stood up first)."""
    return qmul(quat(yaw), UPRIGHT) if model == CAN else quat(yaw)


def random_layout(task: str, seed: int, episode: int) -> Tuple[np.ndarray, np.ndarray]:
    """One random layout for an episode of a layout seed: ([objects, 2] xy, [objects, 4] turns), in the order of the
    task's objects. Table scenes: the objects on the corners of the grid in a random order (the three-step scene: its
    cubes at the centre, in a random order), the grid's half-edge in RANDOM_HALF_EDGES (three-step: WIDE), each object
    up to JITTER off its place (three-step: WIDE_JITTER), and the objects in TURNED turned at random. Drawn again until
    every two objects are MIN_GAP apart and the carrot by the towel is NEARER_BY nearer it than the other carrot. The
    sink scene: the object near SINK_START, the basket's plane at BASKET_XY. The generator is seeded with (seed,
    episode, the task's crc32), so each task has layouts of its own and an episode's layout does not depend on the
    others."""
    t = TASKS[task]
    rng = np.random.default_rng([seed, episode, zlib.crc32(task.encode())])
    if t.scene == "sink":
        xy = np.stack([SINK_START + rng.uniform(-SINK_SPAN, SINK_SPAN), BASKET_XY])
        return xy, np.stack([quat(rng.uniform(-SINK_TURN, SINK_TURN)), quat(0.0)])
    cubes = CENTER + np.array([[0.0, -DELTA], [0.0, DELTA]])
    for _ in range(1000):
        if len(t.objects) == 6:
            xy = np.concatenate([corners(WIDE)[rng.permutation(4)], cubes[rng.permutation(2)]])
            xy = xy + rng.uniform(-WIDE_JITTER, WIDE_JITTER, size=xy.shape)
        else:
            xy = corners(rng.uniform(*RANDOM_HALF_EDGES))[rng.permutation(4)]
            xy = xy + rng.uniform(-JITTER, JITTER, size=xy.shape)
        yaws = np.array([rng.uniform(-np.pi, np.pi) if m in TURNED else 0.0 for m in t.objects])
        if not apart(t.objects, xy, yaws, sizes=SIZES):
            continue
        if task == "NovelCarrotByTowelOnPlate-v0":
            near, other = (np.linalg.norm(xy[i] - xy[3]) for i in (0, 2))
            if other - near < NEARER_BY:
                continue
        break
    else:
        raise RuntimeError(f"{task}: no layout found for seed {seed}, episode {episode}")
    return xy, np.stack([turn(m, y) for m, y in zip(t.objects, yaws, strict=True)])
