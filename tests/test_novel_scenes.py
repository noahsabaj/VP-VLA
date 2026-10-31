"""CPU tests for the new test scenes (eval_files/novel_scenes.py)."""

import re
from pathlib import Path

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files import instruction_sets
from examples.SimplerEnv.eval_files import novel_scenes as ns

COMMON_SH = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments/common.sh"


def rotate(q, v):
    """v turned by the (w, x, y, z) quaternion q."""
    w, x, y, z = q
    return (
        np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        @ v
    )


def episode(task, seed, e):
    """[(model, xy, yaw, quat)] of every object in episode e of a layout seed."""
    xy, q = ns.random_layout(task, seed, e)
    return [(m, xy[i], 2 * np.arctan2(q[i, 3], q[i, 0]), q[i]) for i, m in enumerate(ns.TASKS[task].objects)]


def test_tasks():
    assert len(ns.TASKS) == 8 and set(ns.TASKS).isdisjoint(hs.TASKS)
    assert list(ns.KINDS.values()) == ["new object"] * 2 + ["new pair"] * 2 + ["reasoning"] * 2 + ["long"] * 2
    assert list(ns.SHORT) == list(ns.TASKS)
    seen = {t.instruction for t in hs.TASKS.values()} | {
        i for s in instruction_sets.INSTRUCTION_SETS.values() for i in s
    }
    assert len({t.instruction for t in ns.TASKS.values()} - seen) == 8  # new instructions
    for name, t in ns.TASKS.items():
        assert len(t.objects) == len(t.names) == len(set(t.names)), name
        assert all(m in ns.SIZES or m == ns.BASKET for m in t.objects), name
        assert all(0 <= i < len(t.objects) for a, b, _ in t.goals + t.logged for i in (a, b)), name
        assert t.goals[0][:2] == (0, 1)  # objects 0 and 1 are the source and target SimplerEnv's code reads
        assert t.max_steps == len(t.goals) * hs.STEPS, name  # 120 per pick-and-place
        assert t.z_offset == (0.06 if ns.BASKET in t.objects else 0.02), name  # the basket as in the eggplant task
    assert [n for n, t in ns.TASKS.items() if t.scene == "sink"] == ["NovelCarrotInBasket-v0"]
    assert {m for t in ns.TASKS.values() for m in t.objects} & set(ns.PICK_MODELS) == {ns.SPONGE, ns.CAN}
    assert not any(m in ns.PICK_MODELS for t in hs.TASKS.values() for m in t.objects)


def test_random_layouts():
    for task, t in ns.TASKS.items():
        for e in range(200):
            objs = episode(task, 18, e)
            assert all(np.isclose(np.linalg.norm(q), 1) for *_, q in objs), (task, e)
            if t.scene == "sink":  # the object near the eggplant's start, the basket's plane where it always is
                (_, xy, yaw, _), (_, plane, _, _) = objs
                assert np.all(np.abs(xy - ns.SINK_START) <= ns.SINK_SPAN) and abs(yaw) <= np.pi / 4, e
                assert np.allclose(plane, ns.BASKET_XY)
                continue
            for m, xy, _, _ in objs:  # the cube task's area, SimplerEnv's widest
                assert -0.2601 <= xy[0] <= -0.0599 and abs(xy[1]) <= 0.1001, (task, e, m)
            ext = [np.abs([[np.cos(y), np.sin(y)], [np.sin(y), np.cos(y)]]) @ ns.SIZES[m] for m, _, y, _ in objs]
            for i in range(len(objs)):
                for j in range(i):
                    gap = np.abs(objs[i][1] - objs[j][1]) - (ext[i] + ext[j]) / 2
                    assert gap.max() >= 0.02, (task, e, objs[i][0], objs[j][0])
            for m, _, _, q in objs:
                if m == ns.CAN:  # standing: the can model's long axis (its y) points up
                    assert np.allclose(rotate(q, [0, 1, 0]), [0, 0, 1])
                elif m not in ns.TURNED:
                    assert np.allclose(q, [1, 0, 0, 0])
        assert np.ptp([episode(task, 18, e)[0][2] for e in range(50)]) > 1  # the object gets turns of all sizes


def test_reasoning_layouts():
    for e in range(200):  # the carrot to move is the one nearer the towel, by 2 cm or more
        (_, carrot, _, _), _, (_, other, _, _), (_, towel, _, _) = episode("NovelCarrotByTowelOnPlate-v0", 18, e)
        assert np.linalg.norm(other - towel) - np.linalg.norm(carrot - towel) >= ns.NEARER_BY, e
    three = [episode("NovelThreeStep-v0", 18, e) for e in range(50)]  # the cubes in the middle of the grid
    assert all(np.abs(objs[k][1] - hs.CENTER).max() <= ns.DELTA + ns.WIDE_JITTER for objs in three for k in (4, 5))


def test_layouts_are_fixed_by_seed_episode_and_task():
    task = "NovelSpongeOnPlate-v0"
    a, b = ns.random_layout(task, 18, 5), ns.random_layout(task, 18, 5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.allclose(a[0], ns.random_layout(task, 18, 6)[0])
    assert not np.allclose(a[0], ns.random_layout(task, 19, 5)[0])
    # each task its own layouts, also with the same models
    assert not np.allclose(
        ns.random_layout("NovelNotOrangeOnTowel-v0", 18, 5)[0], ns.random_layout("NovelEggplantOnPlate-v0", 18, 5)[0]
    )


def test_runners_match_the_scenes():
    text = COMMON_SH.read_text()
    novel = re.search(r"novel\) TASKS=\(([^)]*)\)", text).group(1).split()
    assert novel == list(ns.TASKS)
    assert "PutEggplantInBasketScene-v0 | NovelCarrotInBasket-v0)\n      robot=widowx_sink_camera_setup" in text
    steps = {
        "NovelStackOnPlate-v0": "TwoStepSpoonAndCarrot-v0 | NovelStackOnPlate-v0) steps=240",
        "NovelThreeStep-v0": "NovelThreeStep-v0) steps=360",
    }
    for task, line in steps.items():
        assert line in text and line.endswith(str(ns.TASKS[task].max_steps))
    assert all(ns.TASKS[t].max_steps == hs.STEPS for t in ns.TASKS if t not in steps)
    assert '"${TASK_SET:-simpler}" = novel ] && [[ $1 != iid[0-9]* ]]' in text  # random layouts only
