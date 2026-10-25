"""CPU tests for the newer studies' shared parts: the statistics for clustered episodes and several tests
(experiments/stats.py), the baseline routers (experiments/routers.py) and the analyses of what makes the ASM better
(experiments/mechanism.py)."""

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import mechanism, routers, stats
from examples.SimplerEnv.eval_files.planner.experiments.score_two_model import LayoutSet

SPOON, CARROT = s.TASKS[0], s.TASKS[1]


def test_unit_sums_and_cluster_bootstrap():
    v, units = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]), np.array([5, 5, 2, 2, 9, 9])
    assert stats.unit_sums(v, units).tolist() == [2.0, 1.0, 0.0]  # units 2, 5, 9
    # units of one size: the same draws as bootstrap_interval of the per-unit means
    rng = np.random.default_rng(0)
    v, units = rng.integers(0, 2, 120).astype(float), np.repeat(np.arange(30), 4)
    per_unit = stats.unit_sums(v, units) / 4
    assert stats.cluster_bootstrap_interval(v, units) == pytest.approx(stats.bootstrap_interval(per_unit, np.zeros(30)))
    lo, hi = stats.cluster_bootstrap_interval(v, units)
    assert lo < v.mean() < hi


def test_holm():
    assert stats.holm([0.01, 0.04]) == [True, True]  # 0.01 < 0.025, then 0.04 < 0.05
    assert stats.holm([0.03, 0.04]) == [False, False]  # 0.03 is not below 0.025
    assert stats.holm([0.04, 0.001, 0.2]) == [False, True, False]  # 0.001 < 0.0167, then 0.04 is not below 0.025
    assert stats.holm([1.0, 0.002]) == [False, True]


def test_mh_risk_difference():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0], float)
    exposed = np.array([1, 1, 0, 0, 1, 1, 0, 0], bool)
    strata = np.array([SPOON] * 4 + [CARROT] * 4)
    # spoon: 1 - 0 = 1, weight 2 * 2 / 4 = 1; carrot: 0.5 - 0.5 = 0, weight 1
    assert stats.mh_risk_difference(y, exposed, strata) == pytest.approx(0.5)
    assert np.isnan(stats.mh_risk_difference(y[:2], exposed[:2], strata[:2]))  # no stratum with both groups


def test_per_task_choice():
    tasks = np.array([SPOON] * 4 + [CARROT] * 4)
    y_plan = np.array([1, 1, 0, 0, 0, 0, 0, 1], float)
    y_vla = np.array([0, 1, 1, 0, 1, 1, 0, 0], float)
    use = routers.per_task_choice(tasks, y_plan, y_vla, np.array([SPOON, CARROT, SPOON]))
    assert use.tolist() == [True, False, True]  # spoon: a tie (2 and 2), so the planner
    with pytest.raises(ValueError, match="no training episodes of"):
        routers.per_task_choice(tasks, y_plan, y_vla, np.array([s.TASKS[2]]))


def test_knn_choice_uses_the_nearest_episodes_of_the_same_task():
    x = np.array([[0.0], [0.1], [1.0], [1.1], [0.0], [1.0]])
    tasks = np.array([SPOON] * 4 + [CARROT] * 2)
    y_plan = np.array([1, 1, 0, 0, 0, 1], float)  # the spoon's planner succeeds at small x
    y_vla = np.array([0, 0, 1, 1, 1, 0], float)
    test_x, test_tasks = np.array([[0.05], [1.05], [0.05]]), np.array([SPOON, SPOON, CARROT])
    assert routers.knn_choice(x, tasks, y_plan, y_vla, test_x, test_tasks, k=1).tolist() == [True, False, False]
    # k = 2: the carrot's two episodes, where the planner and the VLA tie, so the planner
    assert routers.knn_choice(x, tasks, y_plan, y_vla, test_x, test_tasks, k=2).tolist() == [True, False, True]
    # k above a task's episodes: all of them
    assert routers.knn_choice(x, tasks, y_plan, y_vla, test_x, test_tasks, k=9).tolist() == [True, True, True]


def test_choose_k_and_expected_random():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (200, 1))
    tasks = np.array([SPOON, CARROT] * 100)
    y_plan, y_vla = (x[:, 0] < 0.5).astype(float), (x[:, 0] >= 0.5).astype(float)
    k, scores = routers.choose_k(x, tasks, y_plan, y_vla, np.arange(200) // 2)
    assert len(scores) == len(routers.KS) and k in routers.KS and max(scores) > 0.9
    assert scores[routers.KS.index(k)] == max(scores)
    assert routers.expected_random(0.25, np.array([1.0, 0.0]), np.array([0.0, 0.0])).tolist() == [0.25, 0.0]


def test_complementarity_and_gain_decomposition():
    y_plan = np.array([1, 1, 0, 0, 1, 0], float)
    y_vla = np.array([1, 0, 1, 0, 0, 1], float)
    c = mechanism.complementarity(y_plan, y_vla)
    assert (c["both"], c["planner only"], c["VLA only"], c["neither"]) == (1, 2, 2, 1)
    assert c["oracle gain"] == pytest.approx(2 / 6)  # the share of episodes only the worse method solves
    use = np.array([True, True, True, False, False, False])  # one planner-only episode to the planner, one not
    g = mechanism.gain_decomposition(use, y_plan, y_vla)
    assert (g["planner-only to the planner"], g["VLA-only to the planner"]) == (1, 1)
    assert g["gain over the VLA"] == pytest.approx(0.0) and g["gain over the planner"] == pytest.approx(0.0)


def test_failure_dependence():
    rng = np.random.default_rng(2)
    y_plan = rng.integers(0, 2, 400).astype(float)
    tasks, units = np.array([SPOON, CARROT] * 200), np.arange(400) // 4
    same = mechanism.failure_dependence(y_plan, y_plan.copy(), tasks, units, draws=200)  # fail on the same episodes
    assert same[0] == pytest.approx(-1.0)
    d, lo, hi = mechanism.failure_dependence(y_plan, rng.integers(0, 2, 400).astype(float), tasks, units, draws=200)
    assert lo < d < hi and lo < 0 < hi  # independent failures


def _layout_set(n, rng, seed):
    """Episodes where the planner succeeds in 70% of the episodes with a plan, and the VLA with a small turn."""
    tasks = np.array([s.TASKS[i % 4] for i in range(n)])
    records, y_plan, y_vla = [], [], []
    for i in range(n):
        plan_ok, turn = bool(rng.random() < 0.6), float(rng.uniform(0, 1.5))
        records.append(
            {
                "obj_episode_id": i // 4,
                "layout_seed": seed,
                "layout_per_episode": True,
                "features": {"grasp_turn_rad": turn, "grasp_reach_m": 0.1},
                "plans": [{"success": plan_ok, "ee_path_length": 0.4, "ee_straight_length": 0.3}],
            }
        )
        y_plan.append(float(plan_ok and rng.random() < 0.7))
        y_vla.append(float(turn < 0.75))
    keys = [(t, i // 4) for i, t in enumerate(tasks)]
    return LayoutSet(keys, records, tasks, np.array(y_plan), np.array(y_vla))


def test_ablations_and_learning_curve(capsys):
    rng = np.random.default_rng(3)
    train, test = _layout_set(400, rng, 11), _layout_set(200, rng, 15)
    asm = mechanism.fit(train, s.PLANNER_FEATURES, s.VLA_FEATURES)
    gated = np.zeros(200, bool)
    rows = dict(mechanism.ablations(train, test, asm, gated))
    assert next(iter(rows)) == "the ASM" and "P(planner) without plan_ok" in rows
    assert "P(VLA) without grasp_turn_rad" in rows
    assert rows["the ASM"] > rows["P(VLA): the task columns only"]  # the turn tells the VLA's success
    curve = mechanism.learning_curve(train, test, asm, gated)
    assert [share for share, _ in curve] == list(mechanism.FRACTIONS)
    assert len(curve[-1][1]) == 1 and curve[-1][1][0] == pytest.approx(rows["the ASM"])
    assert all(len(got) <= mechanism.DRAWS for _, got in curve)
    use = np.where(gated, False, np.greater_equal(*asm.probs(test.records, test.tasks)))
    mechanism.report(train, test, asm, use, gated, np.array([e for _, e in test.keys]))
    out = capsys.readouterr().out
    assert "### What makes the ASM better (planned secondary results, no tests)" in out
    assert "Complementarity: " in out and "Learning curve " in out and "The ASM minus the oracle: " in out
