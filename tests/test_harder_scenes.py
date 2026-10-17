"""CPU tests for the harder scenes (eval_files/harder_scenes.py) and their scorer."""

import re
from pathlib import Path

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.planner.experiments import score_harder_scenes as sh

KITCHEN = ["ClutteredSpoonOnTowel-v0", "ClutteredCarrotOnPlate-v0", "TwoStepSpoonAndCarrot-v0"]
NEAR, FAR = "CarrotOnNearPlate-v0", "CarrotOnFarPlate-v0"
# Length and width on the table (m), from ManiSkill2_real2sim's data/custom/info_bridge_custom_v0.json.
FOOTPRINT = {hs.SPOON: (0.14, 0.035), hs.TOWEL: (0.1, 0.1), hs.CARROT: (0.115, 0.032), hs.PLATE: (0.11, 0.11)}
COMMON_SH = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments/common.sh"


def episode(task, e):
    """[(model, xy, yaw)] of every object in episode e, as SimplerEnv's PutOnBridgeInSceneEnv indexes the configs."""
    xys, quats = hs.layouts(task)
    xy, q = xys[e // len(quats)], quats[e % len(quats)]
    return [(m, xy[i], 2 * np.arctan2(q[i, 3], q[i, 0])) for i, m in enumerate(hs.TASKS[task].objects)]


def test_tasks_and_layouts():
    assert sh.TASKS == [*KITCHEN, NEAR, FAR]
    assert [hs.episodes(t) for t in sh.TASKS] == [96, 96, 96, 32, 32]
    assert len({t.instruction for t in hs.TASKS.values()}) == 5
    for name, t in hs.TASKS.items():
        xys, quats = hs.layouts(name)
        assert all(xy.shape == (len(t.objects), 2) for xy in xys), name
        assert all(np.allclose(np.linalg.norm(q, axis=1), 1) for q in quats), name
        assert all(0 <= i < len(t.objects) for a, b, _ in t.goals + t.logged for i in (a, b)), name
        assert t.goals[0][:2] == (0, 1)  # objects 0 and 1 are the source and target SimplerEnv's code reads
    assert hs.TASKS["TwoStepSpoonAndCarrot-v0"].max_steps == 2 * hs.TASKS[KITCHEN[0]].max_steps == 240


def test_objects_start_apart_and_in_reach():
    for task in sh.TASKS:
        for e in range(hs.episodes(task)):
            objs = episode(task, e)
            for m, xy, _ in objs:  # the cube task's area, SimplerEnv's widest
                assert -0.2601 <= xy[0] <= -0.0599 and abs(xy[1]) <= 0.1001, (task, e, m)
            ext = [np.abs([[np.cos(y), np.sin(y)], [np.sin(y), np.cos(y)]]) @ FOOTPRINT[m] for m, _, y in objs]
            for i in range(len(objs)):
                for j in range(i):
                    gap = np.abs(objs[i][1] - objs[j][1]) - (ext[i] + ext[j]) / 2
                    assert gap.max() >= 0.02, (task, e, objs[i][0], objs[j][0])


def test_tasks_of_a_scene_share_their_layouts():
    for e in range(96):
        scenes = [{m: (tuple(np.round(xy, 6)), round(y, 6)) for m, xy, y in episode(t, e)} for t in KITCHEN]
        assert scenes[0] == scenes[1] == scenes[2], e
    for e in range(32):
        near, far = episode(NEAR, e), episode(FAR, e)
        assert np.allclose(near[0][1], far[0][1]) and near[0][2] == far[0][2]  # the carrot
        assert np.allclose(near[1][1], far[2][1]) and np.allclose(near[2][1], far[1][1])  # the plates, swapped
        assert near[1][1][0] > near[2][1][0]  # the target is the plate nearer the robot (at x = 0.147)


def test_runners_match_the_scenes():
    text = COMMON_SH.read_text()
    harder = re.search(r"harder\) TASKS=\(([^)]*)\)", text).group(1).split()
    assert harder == sh.TASKS
    assert "Cluttered*-v0 | TwoStep*-v0) echo 96" in text and "CarrotOn*Plate-v0) echo 32" in text
    assert '[ "$task" = TwoStepSpoonAndCarrot-v0 ] && steps=240' in text


def test_vla_stats(tmp_path):  # from the video names
    folder = tmp_path / sh.TWO_STEP  # short paths: Windows allows 260 characters
    folder.mkdir(parents=True)
    (folder / "failure_obj_episode_3_spoon_on_towel_True_carrot_on_plate_False_moved_other_obj_False.mp4").touch()
    near = tmp_path / NEAR
    near.mkdir(parents=True)
    (near / "failure_obj_episode_5_carrot_on_plate_False_carrot_on_other_plate_True_moved_other_obj_True.mp4").touch()
    stats = sh.vla_stats([str(tmp_path)])
    assert stats[sh.TWO_STEP, 3] == {"spoon_on_towel": True, "carrot_on_plate": False, "moved_other_obj": False}
    assert stats[NEAR, 5] == {"carrot_on_plate": False, "carrot_on_other_plate": True, "moved_other_obj": True}


def test_planned_tests(capsys):
    # the VLA wins every layout the rule cannot read; the planner wins the one-step ones
    plan = {(t, e): float(t not in sh.RULE_BREAKING) for t in sh.TASKS for e in range(10)}
    vla = {k: 1 - v for k, v in plan.items()}
    (_, gap, strata, _), (_, cross, _, _) = sh.tests(plan, vla)
    assert len(gap) == 30 and gap.min() == 1 and set(strata) == set(sh.RULE_BREAKING)
    assert len(cross) == 10 and np.allclose(cross, 2)  # one-step lead +1, two-step lead -1
    sh.print_tests(plan, vla)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("1. ") and "+100.0 points" in lines[1] and lines[1].endswith(": significant")
    assert lines[2].startswith("2. ") and "+200.0 points" in lines[2] and lines[2].endswith(": significant")
    vla = dict(plan)  # both the same everywhere: no difference, so test 2 does not count
    sh.print_tests(plan, vla)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].endswith("exact p = 1: not significant")
    assert lines[2].endswith("does not count (a test before it was not significant)")
