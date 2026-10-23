"""CPU tests for the VLM plan-check v2 study's statistics and scorer (score_plan_check_v2.py)."""

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import score_plan_check_v2 as v2
from examples.SimplerEnv.eval_files.planner.experiments.stats import auc, auc_ranks, delong


def test_auc_ranks_matches_auc():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 60).astype(float)
        p = np.round(rng.random(60), 1)  # many ties
        assert np.isclose(auc_ranks(p, y), auc(p, y))


def naive_delong_variance(a, b, y):
    """DeLong's variance of AUC(a) - AUC(b) from its definition, pair by pair (slow)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def psi(x, i, j):
        return 1.0 if x[i] > x[j] else 0.5 if x[i] == x[j] else 0.0

    v10 = np.array([[np.mean([psi(x, i, j) for j in neg]) for i in pos] for x in (a, b)])
    v01 = np.array([[np.mean([psi(x, i, j) for i in pos]) for j in neg] for x in (a, b)])
    c10, c01 = np.cov(v10), np.cov(v01)
    return (c10[0, 0] + c10[1, 1] - 2 * c10[0, 1]) / len(pos) + (c01[0, 0] + c01[1, 1] - 2 * c01[0, 1]) / len(neg)


def test_delong():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    a = y + rng.normal(0, 1.0, 80)
    b = y + rng.normal(0, 2.0, 80)
    auc_a, auc_b, p = delong(a, b, y)
    assert np.isclose(auc_a, auc(a, y)) and np.isclose(auc_b, auc(b, y))
    from math import erfc, sqrt

    assert np.isclose(p, erfc(abs(auc_a - auc_b) / sqrt(naive_delong_variance(a, b, y)) / sqrt(2)))
    assert np.isclose(delong(b, a, y)[2], p)  # symmetric
    assert delong(a, a, y)[2] == 1.0
    y = np.r_[np.ones(200), np.zeros(200)]
    good, noise = y + rng.normal(0, 0.3, 400), rng.random(400)
    assert delong(good, noise, y)[2] < 1e-10
    with pytest.raises(ValueError):
        delong(a[:3], b[:3], np.array([1, 0, 0]))


def test_fit_threshold():
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    y = np.array([0, 0, 0, 1, 1, 1])
    t, j = v2.fit_threshold(p, y)
    assert t == 0.6 and j == 1.0
    t, j = v2.fit_threshold(np.array([0.2, 0.4, 0.6, 0.8]), np.array([0, 1, 0, 1]))  # J = 0.5 at 0.4 and 0.8
    assert (t, j) == (0.4, 0.5)  # the one nearer 0.5
    with pytest.raises(ValueError):
        v2.fit_threshold(np.array([0.1, 0.9]), np.array([1, 1]))


def group(tasks, planner, vla_ok, p, n=6):
    rows = {(t, e): {"record": {}, "planner": planner(t), "v1": 0.0, "v2": p(t)} for t in tasks for e in range(n)}
    vla = {(t, e): vla_ok(t) for t in tasks for e in range(n)}
    return rows, vla


def test_tests_2_to_4():
    breaking = {"TwoStepSpoonAndCarrot-v0", "CarrotOnNearPlate-v0", "CarrotOnFarPlate-v0"}
    groups, vlas = {}, {}
    groups["harder"], vlas["harder"] = group(
        hs.TASKS,
        lambda t: float(t not in breaking),
        lambda t: float(t in breaking),
        lambda t: 0.1 if t in breaking else 0.9,
    )
    for n in INSTRUCTION_SETS:
        groups[n], vlas[n] = group(s.TASKS, lambda t: 0.0, lambda t: 1.0, lambda t: 0.2)
    (_, d2, _), (_, d3, _), (_, d4, strata) = v2.tests_2_to_4(groups, vlas, t=0.5)
    assert d2.sum() == 18 and d3.sum() == 12 and len(d4) == 24 and (d4 == 4).all()
    assert set(strata) == set(s.TASKS)
    # the threshold from the dev step decides: with t = 0.05 the v2 gate always gives the planner the episode
    (_, d2, _), _, _ = v2.tests_2_to_4(groups, vlas, t=0.05)
    assert d2.sum() == 0


def test_run_names():
    assert v2.planner_run("own", "paper") == "check2_own_paper"
    assert v2.planner_run("synonyms", "iid13") == "check2_words_synonyms_iid13"
    assert v2.vla_run("harder", "iid13") == "vla_harder_iid13"
    assert v2.GROUPS == ["own", *INSTRUCTION_SETS, "harder"]


class FakeVLM:
    """Says yes (0.9) to the outline question and no (0.1) to the tint question."""

    def yes_probability(self, images, question):
        return {"p_yes": 0.9 if "outlined in green" in question else 0.1, "answer_mass": 0.99}


class FakeASM:
    def decide(self, stats, task):
        return {"choice": "planner", "reason": "asm"}


def test_client_checks_each_mark_and_the_first_gates():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner.planner_client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)  # no servers: only what the plan check and the gate read
    c.plan_check, c.plan_check_marks, c.plan_check_threshold = FakeVLM(), ["outline", "tint"], 0.5
    c.task_description, c.log_path, c.episode_key, c.stats = "put carrot on plate", None, None, {"features": {}}
    image, mask = np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20), bool)
    mask[5:8, 5:8] = True
    c._check_plan(image, {"mask": mask}, {"mask": mask})
    assert c.stats["features"] == {"vlm_plan_yes_outline": 0.9, "vlm_plan_yes": 0.1}
    assert c.stats["plan_check_answer_mass_outline"] == c.stats["plan_check_answer_mass"] == 0.99
    c.stats = {"features": {}}
    c._check_plan(image, None, {"mask": mask})  # no object found: no plan to show
    assert c.stats["features"] == {"vlm_plan_yes_outline": 0.0, "vlm_plan_yes": 0.0}

    c.episode_key, c.task_source, c.selector = "PutCarrotOnPlateInScene-v0/None/0", "rule", FakeASM()
    for marks, choice in ((["outline", "tint"], "asm"), (["tint", "outline"], "plan check")):
        c.plan_check_marks, c.phase, c.queue = marks, "approach", [1]
        c.stats = {"features": {"vlm_plan_yes_outline": 0.9, "vlm_plan_yes": 0.1}}
        c._choose()
        assert c.stats["selector_reason"] == choice
        assert (c.phase == "vla") == (choice == "plan check")
