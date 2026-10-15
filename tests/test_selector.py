"""
CPU tests for the adaptive selector (planner/selector.py) and the experiments' statistics (numpy only).

The loaders read the folder layout the evals write: <task>/planner_stats.jsonl for the planner and
SimplerEnv's video names for the VLA, so the tests build small folders of that kind.
"""

import json
import math

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import stats

SPOON, CARROT = s.TASKS[0], s.TASKS[1]


def _record(episode, success, plan_ok=True, ratio=1.5, **extra):
    plan = {"success": plan_ok, "time": 0.1}
    if plan_ok:
        plan.update(ee_path_length=0.3 * ratio, ee_straight_length=0.3)
    return {
        "mode": "planner",
        "obj_episode_id": episode,
        "success": success,
        "features_version": s.FEATURES_VERSION,
        "features": {"grasp_sam3_score": 0.5},
        "plans": [plan],
        "depth_noise": "harsh",
        **extra,
    }


def _write_planner(root, task, records):
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _write_video(root, task, episode, success):
    folder = root / task / "ckpt" / "scene"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{'success' if success else 'failure'}_obj_episode_{episode}_moved_True.mp4").write_bytes(b"")


def test_row_features_plan_ok_and_path_ratio():
    x = s.row_features(_record(0, True, ratio=2.0), ["plan_ok", "plan_path_ratio", "grasp_sam3_score", "missing"])
    assert x.tolist() == pytest.approx([1.0, 2.0, 0.5, 0.0])
    # no plan: the ratio is 1 (no detour known), not a division by zero
    assert s.row_features(_record(0, False, plan_ok=False), s.FEATURES_V3).tolist() == [0.0, 1.0]


def test_heuristic_uses_planner():
    x = np.array([[1.0, 0.3], [1.0, 0.29], [0.0, 0.9]])
    assert s.heuristic_uses_planner(x, s.HEURISTIC_FEATURES).tolist() == [True, False, False]


def test_logistic_regression_learns_and_round_trips():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] > 0).astype(float)
    model = s.LogisticRegression().fit(x, y)
    assert ((model.prob(x) >= 0.5) == (y == 1)).mean() > 0.95
    copy = s.LogisticRegression.from_json(json.loads(json.dumps(model.to_json())))
    np.testing.assert_array_equal(copy.prob(x), model.prob(x))
    np.testing.assert_array_equal(copy.use_planner(x), model.prob(x) >= 0.5)


def test_boosted_trees_learn_and_round_trip():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(200, 2))
    y = (x[:, 1] > 0.5).astype(float)
    model = s.BoostedTrees().fit(x, y)
    assert ((model.prob(x) >= 0.5) == (y == 1)).mean() > 0.95
    copy = s.BoostedTrees.from_json(json.loads(json.dumps(model.to_json())))
    np.testing.assert_array_equal(copy.prob(x), model.prob(x))


def test_planner_records_refuse_wrong_joins(tmp_path):
    _write_planner(tmp_path / "ok", SPOON, [_record(0, True), _record(1, False)])
    assert sorted(s.load_planner(str(tmp_path / "ok"))) == [(SPOON, 0), (SPOON, 1)]
    bad = {
        "mode": [_record(0, True, mode="selector")],
        "duplicate": [_record(0, True), _record(0, False)],
        "settings": [_record(0, True), _record(1, True, depth_noise="none")],
    }
    for name, records in bad.items():
        _write_planner(tmp_path / name, SPOON, records)
        with pytest.raises(SystemExit):
            s.planner_records(str(tmp_path / name))


def test_load_vla_averages_runs_and_refuses_two_outcomes(tmp_path):
    for run, success in (("a", True), ("b", False)):
        _write_video(tmp_path / run, SPOON, 3, success)
    assert s.load_vla([str(tmp_path / "a"), str(tmp_path / "b")]) == {(SPOON, 3): 0.5}
    _write_video(tmp_path / "a", SPOON, 3, False)
    with pytest.raises(SystemExit):
        s.load_vla([str(tmp_path / "a")])


def test_evaluate_saves_a_model_the_eval_can_load(tmp_path, capsys):
    planner, vla = tmp_path / "planner", tmp_path / "vla"
    for task in (SPOON, CARROT):
        records = [_record(e, success=e % 3 != 0, ratio=1.0 + (e % 3 == 0)) for e in range(12)]
        _write_planner(planner, task, records)
        for e in range(12):
            _write_video(vla, task, e, e % 2 == 0)
    path = tmp_path / "model.json"
    s.evaluate(str(planner), [str(vla)], save=str(path))
    assert "leave one task out" in capsys.readouterr().out
    saved = json.loads(path.read_text())
    assert list(saved) == ["model", "features", "mean", "std", "w", "b", "threshold", "trained_on"]
    assert saved["features"] == s.FEATURES_V3
    model, names = s.load_saved(str(path))
    assert names == s.FEATURES_V3
    assert model.trained_on == {"depth_noise": "harsh"}


def test_leave_one_task_out_never_sees_the_task():
    tasks = np.array([SPOON, SPOON, CARROT])

    def predict(train, test):
        assert not (train & test).any()
        return np.full(test.sum(), float(train.sum()))

    assert s.leave_one_task_out(tasks, predict).tolist() == [1.0, 1.0, 2.0]


def test_task_rates():
    y, tasks = np.array([1.0, 0.0, 1.0]), np.array([SPOON, SPOON, CARROT])
    assert s.task_rate(y, tasks, np.array([CARROT, SPOON, s.TASKS[2]])).tolist() == pytest.approx([1.0, 0.5, 2 / 3])
    assert s.other_layouts_rate(np.array([1.0, 0.0, 1.0])).tolist() == [0.5, 1.0, 0.5]


def test_mcnemar_and_sign_flip_agree_on_binary_outcomes():
    base = np.array([0, 0, 0, 0, 1, 1, 0, 1], float)
    new = np.array([1, 1, 1, 0, 1, 0, 0, 1], float)
    better, worse, p = stats.mcnemar(base, new)
    assert (better, worse) == (3, 1)
    assert p == pytest.approx(2 * (1 + 4) / 2**4)
    assert stats.sign_flip(new - base, 1) == pytest.approx(p)
    assert stats.mcnemar(base, base) == (0, 0, 1.0)


def test_auc_counts_ties_half_and_takes_mean_labels():
    assert stats.auc(np.array([0.9, 0.5, 0.5, 0.1]), np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.875)
    assert stats.auc(np.array([0.3, 0.2, 0.1]), np.array([1.0, 0.5, 0.0])) == 1.0
    assert math.isnan(stats.auc(np.array([0.3, 0.2]), np.array([1.0, 1.0])))


def test_fleiss_kappa():
    assert stats.fleiss_kappa(np.array([[1, 1], [0, 0], [1, 1], [0, 0]], float)) == pytest.approx(1.0)
    assert stats.fleiss_kappa(np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)) == pytest.approx(-1.0)
