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
from examples.SimplerEnv.eval_files.planner.experiments import score_two_model, stats

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


def _two_model_record(episode, success, turn, plan_ok=True, reach=0.1):
    r = _record(episode, success, plan_ok=plan_ok)
    r["features"].update(grasp_turn_rad=turn, grasp_reach_m=reach)
    return r


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
        "rgb": [_record(0, True, rgb_noise="none"), _record(1, True, rgb_noise="harsh")],
    }
    for name, records in bad.items():
        _write_planner(tmp_path / name, SPOON, records)
        with pytest.raises(SystemExit):
            s.planner_records(str(tmp_path / name))
    # records written before the RGB noise existed have no rgb_noise and join those with "none"
    _write_planner(tmp_path / "old", SPOON, [_record(0, True), _record(1, True, rgb_noise="none")])
    assert s.planner_settings(s.planner_records(str(tmp_path / "old"))) == {"depth_noise": "harsh", "rgb_noise": "none"}


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


def test_brier_ece_and_calibration_bins():
    p = np.array([0.05, 0.05, 0.1, 0.95, 0.95, 1.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert stats.brier(p, y) == pytest.approx(np.mean((p - y) ** 2))
    bins = stats.calibration_bins(p, y)
    # p = 0.1 starts the second bin; p = 1 is in the last bin
    assert [b[0] for b in bins] == pytest.approx([0.0, 0.1, 0.9])
    assert [b[2] for b in bins] == [2, 1, 3]
    assert stats.ece(p, y) == pytest.approx((2 * 0.05 + 0.1 + 3 * (1 - (0.95 + 0.95 + 1.0) / 3)) / 6)
    assert stats.ece(np.full(4, 0.5), np.array([1.0, 0.0, 1.0, 0.0])) == 0.0


def test_bootstrap_interval_keeps_each_task_whole():
    v = np.array([1.0] * 10 + [0.0] * 30)
    tasks = np.array([SPOON] * 10 + [CARROT] * 30)
    # each draw takes 10 Spoon and 30 Carrot episodes, so the mean is always 0.25
    assert stats.bootstrap_interval(v, tasks) == pytest.approx((0.25, 0.25))
    lo, hi = stats.bootstrap_interval(np.array([1.0, 0.0] * 20), np.array([SPOON] * 40))
    assert lo < 0.5 < hi


def test_success_model_plain_units_give_the_same_probabilities():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 2)) * [1.0, 5.0] + [0.5, 2.0]
    tasks = np.array([SPOON, CARROT, s.TASKS[2]] * 20)
    y = (x[:, 0] + rng.normal(size=60) > 0.5).astype(float)
    model = s.SuccessModel(["a", "b"]).fit(x, tasks, y)
    intercepts, slopes = model.plain()
    logit = np.array([intercepts[t] for t in tasks]) + x @ np.array([slopes["a"], slopes["b"]])
    np.testing.assert_allclose(model.prob(x, tasks), s.sigmoid(logit), rtol=1e-9)
    copy = s.SuccessModel.from_json(json.loads(json.dumps(model.to_json())))
    np.testing.assert_array_equal(copy.prob(x, tasks), model.prob(x, tasks))
    with pytest.raises(ValueError):  # a weight for each feature and task column
        s.SuccessModel.from_json({**model.to_json(), "tasks": s.TASKS[:3]})


def test_success_model_keeps_its_own_task_columns():
    rng = np.random.default_rng(3)
    x, tasks = rng.normal(size=(40, 1)), np.array([CARROT, "Other-v0"] * 20)
    y = (x[:, 0] > 0).astype(float)
    model = s.SuccessModel(["a"], tasks=["Other-v0", CARROT]).fit(x, tasks, y)
    copy = s.SuccessModel.from_json(json.loads(json.dumps(model.to_json())))
    assert copy.tasks == ["Other-v0", CARROT] and list(copy.plain()[0]) == ["Other-v0", CARROT]
    np.testing.assert_array_equal(copy.prob(x, tasks), model.prob(x, tasks))
    with pytest.raises(ValueError, match="no task column for " + SPOON):
        model.prob(x[:1], np.array([SPOON]))
    with pytest.raises(ValueError, match="no task column for Other-v0"):
        s.SuccessModel(["a"]).fit(x, tasks, y)  # the default columns: TASKS


def test_two_model_asm_trains_saves_and_decides(tmp_path, capsys):
    rng = np.random.default_rng(0)
    planner, vla = tmp_path / "planner", tmp_path / "vla"
    for task in (SPOON, CARROT):
        records = []
        for e in range(40):  # the planner succeeds when it has a plan, the VLA when the turn is small
            turn, plan_ok = float(rng.uniform(0, np.pi / 2)), bool(e % 2)
            records.append(_two_model_record(e, success=plan_ok, turn=turn, plan_ok=plan_ok))
            _write_video(vla, task, e, turn < np.pi / 4)
        _write_planner(planner, task, records)
    path = tmp_path / "asm.json"
    s.save_two_model(str(planner), [str(vla)], str(path))
    assert "P(VLA succeeds): logit" in capsys.readouterr().out
    asm = s.load_asm(str(path))
    assert isinstance(asm, s.TwoModelASM)
    assert asm.trained_on == {"depth_noise": "harsh"}
    assert asm.planner.features == s.PLANNER_FEATURES and asm.vla.features == s.VLA_FEATURES
    assert asm.planner.plain()[1]["plan_ok"] > 0
    assert asm.vla.plain()[1]["grasp_turn_rad"] < 0
    small_turn = asm.decide(_two_model_record(0, True, turn=0.1, plan_ok=False), SPOON)
    assert set(small_turn) == {"p_planner", "p_vla", "choice"}
    assert small_turn["choice"] == "vla" and small_turn["p_vla"] > small_turn["p_planner"]
    assert asm.decide(_two_model_record(0, True, turn=1.4, plan_ok=True), CARROT)["choice"] == "planner"


def test_load_asm_reads_a_threshold_model_as_before(tmp_path):
    planner, vla = tmp_path / "planner", tmp_path / "vla"
    records = {
        task: [_record(e, success=e % 3 != 0, ratio=1.0 + (e % 3 == 0)) for e in range(12)] for task in (SPOON, CARROT)
    }
    for task, rs in records.items():
        _write_planner(planner, task, rs)
        for e in range(12):
            _write_video(vla, task, e, e % 2 == 0)
    path = tmp_path / "model.json"
    s.evaluate(str(planner), [str(vla)], save=str(path))
    asm = s.load_asm(str(path))
    assert isinstance(asm, s.ThresholdASM)
    model, names = s.load_saved(str(path))
    for r in records[SPOON]:  # the decision the selector mode made before load_asm
        p = float(model.prob(s.row_features(r, names)[None])[0])
        assert asm.decide(r, SPOON) == {"p_planner": p, "choice": "planner" if p >= model.threshold else "vla"}


def test_matched_layouts_leaves_out_unmatched_episodes(tmp_path, capsys):
    _write_planner(tmp_path / "p", SPOON, [_record(0, True), _record(1, False)])
    for e in (1, 2):
        _write_video(tmp_path / "v", SPOON, e, True)
    _, vla, keys = s.matched_layouts(str(tmp_path / "p"), [str(tmp_path / "v")])
    assert keys == [(SPOON, 1)] and vla[SPOON, 1] == 1.0
    out = capsys.readouterr().out
    assert "1 layouts planner only" in out and "1 layouts VLA only" in out


def _layout_set(rng, n):
    keys, records, tasks, y_plan, y_vla = [], [], [], [], []
    for task in s.TASKS:
        for e in range(n):
            turn, plan_ok = float(rng.uniform(0, np.pi / 2)), bool(rng.random() < 0.6)
            records.append(_two_model_record(e, plan_ok and rng.random() < 0.9, turn, plan_ok=plan_ok))
            keys.append((task, e))
            tasks.append(task)
            y_plan.append(float(records[-1]["success"]))
            y_vla.append(float(rng.random() < 1 - turn / 2))
    return score_two_model.LayoutSet(keys, records, np.array(tasks), np.array(y_plan), np.array(y_vla))


def test_score_two_model_rows(capsys):
    rng = np.random.default_rng(3)
    train, test = _layout_set(rng, 40), _layout_set(rng, 30)
    asm, rows, picks, probs = score_two_model.score(train, test, {}, {}, "iid12")
    p_plan, p_vla = asm.probs(test.records, test.tasks)
    np.testing.assert_array_equal(rows[score_two_model.ASM], np.where(p_plan >= p_vla, test.y_plan, test.y_vla))
    assert picks[score_two_model.ASM] == int((p_plan < p_vla).sum())
    assert picks["planner-only"] == 0 and picks["VLA-only"] == len(test.keys)
    assert list(rows)[-1] == "oracle" and all(np.all(rows["oracle"] >= v) for v in rows.values())
    np.testing.assert_array_equal(probs["P(VLA)"][0], p_vla)
    score_two_model.print_tests(rows, train, test, planned=True)
    assert "1. vs " in capsys.readouterr().out


def test_planned_tests_stop_at_the_first_that_is_not_significant(capsys):
    st = score_two_model
    y = np.array([1.0, 0.0] * 20)
    ls = st.LayoutSet([(SPOON, e) for e in range(40)], [{}] * 40, np.array([SPOON] * 40), y, 1 - y)
    rows = {st.ASM: y, "planner-only": y, "VLA-only": 1 - y, st.RULE_A: 1 - y, st.RULE_C: y, "oracle": np.ones(40)}
    st.print_tests(rows, ls, ls, planned=True)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("1. vs planner-only") and lines[1].endswith("p = 1: not significant")
    assert lines[2].endswith("does not count (a test before it was not significant)")
    assert lines[3].endswith("does not count (a test before it was not significant)")


def test_score_two_model_load_takes_one_vla_eval_per_level(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    _write_planner(tmp_path / "cam_iid11_harsh", SPOON, [_record(0, True), _record(1, False)])
    _write_video(tmp_path / "vla_iid11_harsh", SPOON, 0, False)
    _write_video(tmp_path / "vla_iid11_harsh", SPOON, 1, True)
    ls = score_two_model.load("cam", "iid11", "harsh", "vla_{set}_{level}")
    assert ls.keys == [(SPOON, 0), (SPOON, 1)]
    assert ls.y_plan.tolist() == [1.0, 0.0] and ls.y_vla.tolist() == [0.0, 1.0]


def test_first_step_reads_the_hidden_state_and_the_action_spread(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    root = tmp_path / score_two_model.runs.PERTURB["iid12"]
    root.mkdir(parents=True)
    hidden = np.arange(12.0).reshape(3, 4)  # middle, 3/4 and last layer
    actions = np.stack([np.zeros((2, 7)), np.ones((2, 7))])  # 2 copies: standard deviation 0.5 everywhere
    np.savez(root / f"{SPOON}_e12_5.npz", hidden=hidden, perturbed_actions=actions)
    out = score_two_model.first_step("iid12")
    assert list(out) == [(SPOON, 5)]
    np.testing.assert_array_equal(out[SPOON, 5][0], hidden[2])
    assert out[SPOON, 5][1] == pytest.approx(np.log(0.5))
    assert score_two_model.first_step("seed2") == {}
    np.savez(root / f"{CARROT}_e12_1.npz", hidden=hidden)  # no perturbed actions
    with pytest.raises(SystemExit):
        score_two_model.first_step("iid12")
