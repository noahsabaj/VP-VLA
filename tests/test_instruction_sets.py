"""CPU tests for the reworded instructions (eval_files/instruction_sets.py) and their scorer."""

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import instruction_sets as ins
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import score_instructions


def test_every_set_has_one_new_instruction_per_task():
    assert ins.TASKS == tuple(s.TASKS)
    assert ins.NAMES[0] == "default"
    for name, texts in ins.INSTRUCTION_SETS.items():
        assert len(texts) == len(ins.TASKS), name
        assert len(set(texts)) == len(texts) and all(t == t.strip().lower() and t for t in texts), name
    every = [t for texts in ins.INSTRUCTION_SETS.values() for t in texts]
    assert len(set(every)) == len(every)


def test_instruction_for():
    assert ins.instruction_for("default", ins.TASKS[0]) is None
    assert ins.instruction_for("order", ins.TASKS[1]) == ins.INSTRUCTION_SETS["order"][1]
    with pytest.raises(ValueError):
        ins.instruction_for("nonsense", ins.TASKS[0])
    with pytest.raises(ValueError):
        ins.instruction_for("order", "OpenTopDrawerCustomInScene-v0")


def test_planned_tests_count_each_layout_once(capsys):
    tasks = np.array([s.TASKS[0]] * 10)
    # columns: own instructions, then 4 reworded sets; the planner drops from 1 to 0, the VLA stays at 1
    plan = np.hstack([np.ones((10, 1)), np.zeros((10, 4))])
    vla = np.ones((10, 5))
    score_instructions.print_tests(plan, vla, tasks)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("1. ") and "+100.0 points" in lines[1] and "10 layouts up / 0 down" in lines[1]
    assert lines[1].endswith("exact p = 0.002: significant")  # 10 layouts all up: 2 / 2**10
    assert lines[2].startswith("2. ") and lines[2].endswith(": significant")
    plan[:, 0] = 0  # the planner fails with the tasks' own instructions too: no drop, so test 1 fails
    score_instructions.print_tests(plan, vla, tasks)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].endswith("exact p = 1: not significant")
    assert lines[2].endswith("does not count (a test before it was not significant)")
