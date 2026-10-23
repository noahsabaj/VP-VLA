"""CPU tests for the VLM plan check (planner/plan_check.py) and its scorer."""

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS
from examples.SimplerEnv.eval_files.planner import plan_check
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import score_plan_check as sp


def test_mark_plan():
    image = np.full((40, 60, 3), 100, dtype=np.uint8)
    grasp = np.zeros((40, 60), bool)
    grasp[5:10, 5:10] = True
    place = np.zeros((40, 60), bool)
    place[20:30, 30:50] = True
    out = plan_check.mark_plan(image, grasp, place, border=2)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert out[7, 7, 1] > 150 and out[7, 7, 0] < 100  # the object tinted green
    assert (out[0:3, :] == 100).all() and (out[25, 40] == 100).all()  # elsewhere unchanged, inside the box too
    assert (out[18, 35] == plan_check.TARGET_COLOR).all() and (out[25, 28] == plan_check.TARGET_COLOR).all()
    # an empty target mask: only the tint
    assert (plan_check.mark_plan(image, grasp, np.zeros_like(place))[20:, 20:] == 100).all()


def test_outline_mark():
    image = np.full((40, 60, 3), 100, dtype=np.uint8)
    grasp = np.zeros((40, 60), bool)
    grasp[10:20, 10:20] = True
    place = np.zeros((40, 60), bool)
    out = plan_check.mark_plan(image, grasp, place, border=2, mark="outline")
    assert (out[grasp] == 100).all()  # the object keeps its colour
    ring = np.zeros_like(grasp)
    ring[8:22, 8:22] = True
    ring &= ~grasp
    assert (out[ring] == plan_check.OBJECT_COLOR).all()  # a 2-pixel green line just outside it
    rest = ~(ring | grasp)
    assert (out[rest] == 100).all()
    with pytest.raises(ValueError):
        plan_check.mark_plan(image, grasp, place, mark="glow")


def test_question_and_gate():
    q = plan_check.question("put carrot on plate")
    assert q == (
        'The robot is told: "put carrot on plate"\n'  # the first study's question, word for word
        "The first image is the robot's camera view. The second image is the same view with a motion "
        "planner's plan marked on it: the planner will pick up the object tinted green, put it down on the "
        "place inside the blue box, and then stop. It will do nothing else.\n"
        "Will doing exactly this complete the whole instruction? Answer yes or no."
    )
    assert plan_check.question("put carrot on plate", "outline") == q.replace("tinted green", "outlined in green")
    assert plan_check.gate(0.49) == "vla" and plan_check.gate(0.5) == "planner" and plan_check.gate(1.0) == "planner"
    assert plan_check.gate(0.3, threshold=0.2) == "planner" and plan_check.gate(0.1, threshold=0.2) == "vla"
    with pytest.raises(ValueError):
        plan_check.gate(None)
    assert plan_check.feature("tint") == "vlm_plan_yes" and plan_check.feature("outline") == "vlm_plan_yes_outline"
    with pytest.raises(ValueError):
        plan_check.feature("glow")


def rows(tasks, planner, vla, p_yes, n=10):
    return {
        (t, e): {"record": {}, "planner": planner(t), "vla": vla(t), "p_yes": p_yes(t)} for t in tasks for e in range(n)
    }


def test_planned_tests(capsys):
    # the planner wins the one-step harder tasks, the VLA the others, and the VLM tells them apart: the gate gets all
    breaking = {"TwoStepSpoonAndCarrot-v0", "CarrotOnNearPlate-v0", "CarrotOnFarPlate-v0"}
    harder = rows(
        hs.TASKS,
        lambda t: float(t not in breaking),
        lambda t: float(t in breaking),
        lambda t: 0.1 if t in breaking else 0.9,
    )
    words = {n: rows(s.TASKS, lambda t: 0.0, lambda t: 1.0, lambda t: 0.2) for n in INSTRUCTION_SETS}
    c = sp.columns(harder, sorted(harder))
    assert (c["gate"] == 1).all()
    (_, d1, _, _), (_, d2, _, _), (_, d3, strata, _) = sp.tests(harder, words)
    assert d1.sum() == 30 and d2.sum() == 20 and len(d3) == 40 and (d3 == 4).all() and set(strata) == set(s.TASKS)
    sp.print_tests(harder, words)
    lines = capsys.readouterr().out.splitlines()
    assert all(line.endswith(": significant") for line in lines[1:4])
    # a VLM that always says yes: the gate is the planner, so test 1 fails and the others do not count
    harder = {k: {**r, "p_yes": 0.9} for k, r in harder.items()}
    sp.print_tests(harder, words)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].endswith("exact p = 1: not significant")
    assert lines[2].endswith("(a test before it was not significant)")
    assert lines[3].endswith("(a test before it was not significant)")
