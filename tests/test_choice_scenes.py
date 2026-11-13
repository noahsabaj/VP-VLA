"""CPU tests for the choice scenes (eval_files/choice_scenes.py): the instruction study's four tasks, the layouts
that they share with the kitchen scene of the harder scenes, and the runners' task set."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import choice_scenes as cs
from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files import novel_scenes as ns

EXPERIMENTS = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments"
COMMON_SH = EXPERIMENTS / "common.sh"
needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="the runners are bash scripts"
)
KITCHEN = [name for name, t in hs.TASKS.items() if t.scene == "kitchen"]
MODELS = {"spoon": hs.SPOON, "towel": hs.TOWEL, "carrot": hs.CARROT, "plate": hs.PLATE}
# The plan's table: each task's instruction, its goal and its logged pair (the object, the target, and whether all of
# the object must lie on the target and touch nothing else).
PLAN = {
    "ChoiceSpoonOnTowel-v0": ("put the spoon on the towel", ("spoon", "towel", False), ("spoon", "plate", False)),
    "ChoiceCarrotOnPlate-v0": ("put carrot on plate", ("carrot", "plate", True), ("carrot", "towel", True)),
    "ChoiceSpoonOnPlate-v0": ("put the spoon on the plate", ("spoon", "plate", False), ("spoon", "towel", False)),
    "ChoiceCarrotOnTowel-v0": ("put the carrot on the towel", ("carrot", "towel", True), ("carrot", "plate", True)),
}


def named(task, pair):
    """A goal or logged pair with its objects' names instead of their indices."""
    return (task.names[pair[0]], task.names[pair[1]], pair[2])


def test_the_task_table():
    assert list(cs.TASKS) == list(PLAN) == list(cs.SHORT)
    assert set(cs.TASKS).isdisjoint(hs.TASKS) and set(cs.TASKS).isdisjoint(ns.TASKS)
    for name, (instruction, goal, logged) in PLAN.items():
        t = cs.TASKS[name]
        assert t.instruction == instruction, name
        assert t.scene == "kitchen" and t.max_steps == hs.STEPS == 120 and t.z_offset == 0.02, name
        assert len(t.goals) == len(t.logged) == 1, name
        assert named(t, t.goals[0]) == goal and named(t, t.logged[0]) == logged, name
        assert t.goals[0][:2] == (0, 1), name  # objects 0 and 1 are the source and target SimplerEnv's code reads
        assert sorted(t.objects) == sorted(MODELS.values()), name  # the kitchen scene's four objects
        assert t.objects == tuple(MODELS[n] for n in t.names), name
        # the form that the planner's fixed rule reads: the object before "on", the target after it
        assert re.fullmatch(rf"put (the )?{goal[0]} on (the )?{goal[1]}", instruction), name
    assert len({t.instruction for t in cs.TASKS.values()}) == 4


def test_the_episode_stats_names():
    names = {n: [hs.pair_name(t, p) for p in t.goals + t.logged] for n, t in cs.TASKS.items()}
    assert names == {
        "ChoiceSpoonOnTowel-v0": ["spoon_on_towel", "spoon_on_plate"],
        "ChoiceCarrotOnPlate-v0": ["carrot_on_plate", "carrot_on_towel"],
        "ChoiceSpoonOnPlate-v0": ["spoon_on_plate", "spoon_on_towel"],
        "ChoiceCarrotOnTowel-v0": ["carrot_on_towel", "carrot_on_plate"],
    }


def test_trained_and_new_pairings():
    assert (*cs.TRAINED, *cs.NEW) == tuple(cs.TASKS)
    # the trained pairings are the kitchen scene's own one-step tasks: the same instruction, objects and goal
    for choice, kitchen in zip(cs.TRAINED, ("ClutteredSpoonOnTowel-v0", "ClutteredCarrotOnPlate-v0"), strict=True):
        a, b = cs.TASKS[choice], hs.TASKS[kitchen]
        assert (a.instruction, a.objects, a.names, a.goals) == (b.instruction, b.objects, b.names, b.goals)
    # a new pairing's logged pair is the goal of its object's trained pairing, checked in the same way
    assert list(cs.OBJECT_TASKS) == ["spoon", "carrot"]
    for obj, (trained, new) in cs.OBJECT_TASKS.items():
        assert trained in cs.TRAINED and new in cs.NEW
        assert cs.TASKS[trained].names[0] == cs.TASKS[new].names[0] == obj
        assert named(cs.TASKS[new], cs.TASKS[new].logged[0]) == named(cs.TASKS[trained], cs.TASKS[trained].goals[0])
        assert named(cs.TASKS[trained], cs.TASKS[trained].logged[0]) == named(cs.TASKS[new], cs.TASKS[new].goals[0])
    # no instruction of a new pairing is one of the harder or the new test scenes
    seen = {t.instruction for t in [*hs.TASKS.values(), *ns.TASKS.values()]}
    assert not {cs.TASKS[n].instruction for n in cs.NEW} & seen


def by_model(tasks, layout, task):
    """{model: (xy, turn)} of a layout, which is in the order of the task's objects."""
    xy, quats = layout
    assert xy.shape == (4, 2) and quats.shape == (4, 4)
    return {m: (tuple(xy[i]), tuple(quats[i])) for i, m in enumerate(tasks[task].objects)}


def test_layouts_are_shared_with_the_kitchen_tasks():
    assert cs.KITCHEN_TASK in KITCHEN and len(KITCHEN) == 3
    for seed in (13, 17, 19, 98):
        for e in range(60):
            kitchen = [by_model(hs.TASKS, hs.random_layout(t, seed, e), t) for t in KITCHEN]
            choice = [by_model(cs.TASKS, cs.random_layout(t, seed, e), t) for t in cs.TASKS]
            # exactly equal, not only close: every task has each object in the same place with the same turn
            assert all(scene == kitchen[0] for scene in [*kitchen, *choice]), (seed, e)
    # a trained pairing has its kitchen task's object order, so its layout is that task's as it is
    for choice, kitchen in zip(cs.TRAINED, ("ClutteredSpoonOnTowel-v0", "ClutteredCarrotOnPlate-v0"), strict=True):
        a, b = cs.random_layout(choice, 19, 7), hs.random_layout(kitchen, 19, 7)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_layouts_follow_the_object_order():
    for e in range(20):
        base_xy, base_q = hs.random_layout("ClutteredSpoonOnTowel-v0", 19, e)  # spoon, towel, carrot, plate
        spoon, towel, carrot, plate = range(4)
        xy, q = cs.random_layout("ChoiceSpoonOnPlate-v0", 19, e)  # spoon, plate, towel, carrot
        assert np.array_equal(xy, base_xy[[spoon, plate, towel, carrot]])
        assert np.array_equal(q, base_q[[spoon, plate, towel, carrot]])
        xy, q = cs.random_layout("ChoiceCarrotOnTowel-v0", 19, e)  # carrot, towel, spoon, plate
        assert np.array_equal(xy, base_xy[[carrot, towel, spoon, plate]])
        assert np.array_equal(q, base_q[[carrot, towel, spoon, plate]])
        # the towel and the plate are not turned; the spoon and the carrot are
        assert np.allclose(q[1], [1, 0, 0, 0]) and np.allclose(q[3], [1, 0, 0, 0])
    a = cs.random_layout("ChoiceSpoonOnPlate-v0", 19, 5)
    assert not np.allclose(a[0], cs.random_layout("ChoiceSpoonOnPlate-v0", 19, 6)[0])
    assert not np.allclose(a[0], cs.random_layout("ChoiceSpoonOnPlate-v0", 98, 5)[0])


def test_runners_match_the_scenes():
    text = COMMON_SH.read_text()
    choice = re.search(r"choice\) TASKS=\(([^)]*)\)", text).group(1).split()
    assert choice == list(cs.TASKS)
    assert "TASK_SET must be simpler, harder, novel or choice" in text
    assert '"${TASK_SET:-simpler}" = choice ] && [[ $1 != iid[0-9]* ]]' in text  # random layouts only
    # the runners' default scene, robot and 120 steps: no line gives a choice scene others
    assert not [line for line in text.splitlines() if "Choice" in line and ("steps=" in line or "robot=" in line)]
    assert all(t.max_steps == hs.STEPS for t in cs.TASKS.values())


@needs_bash
def test_the_runners_take_the_choice_scenes(tmp_path):
    true = shutil.which("true")
    env = {
        **os.environ,
        **dict.fromkeys(("STARVLA_PYTHON", "SAM3_PYTHON", "SAM31_PYTHON", "SIMPLER_PYTHON"), true),
        "SIMPLERENV_PATH": str(tmp_path),
        "RUNS_DIR": str(tmp_path / "runs"),
        "TASK_SET": "choice",
        "IID_EPISODES": "48",
    }

    def run(*args):
        return subprocess.run(["bash", *map(str, args)], env=env, capture_output=True, text=True, timeout=60)

    # the 4 tasks take turns, 48 episodes each, with a layout per episode of the layout seed
    result = run("-c", f'source "{COMMON_SH}" && all_chunks iid19 24 && layout_args iid19')
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:-1] == [f"{t}:{s}:{s + 24}" for s in (0, 24) for t in cs.TASKS]
    assert lines[-1] == "--layout-seed 19 --layout-per-episode"
    # random layouts only, and only the set's own tasks as chunks: the runners stop before they start anything
    for args, error in [
        ((EXPERIMENTS / "planner_eval.sh", 0, 10320, "x", "none", "paper"), "LAYOUTS=iidN, not 'paper'"),
        ((EXPERIMENTS / "vla_eval.sh", 0, 10400, "x", "seed2", 4), "LAYOUTS=iidN, not 'seed2'"),
        (
            (EXPERIMENTS / "planner_eval.sh", 0, 10320, "x", "none", "iid19", "ClutteredSpoonOnTowel-v0:0:8"),
            "a task of TASK_SET=choice, not 'ClutteredSpoonOnTowel-v0:0:8'",
        ),
    ]:
        result = run(*args)
        assert result.returncode != 0 and error in result.stderr, (args, result.stderr)
    assert not (tmp_path / "runs").exists()
