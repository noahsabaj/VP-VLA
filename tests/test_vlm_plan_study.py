"""CPU tests for the VLM-plan study: the VLM plan's gate before the ASM, the choice of P(planner)'s features, and the
study's scorer (experiments/score_vlm_plan.py)."""

import json
import sys

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import check_closed_loop, score_vlm_plan

SPOON, CARROT = s.TASKS[0], s.TASKS[1]
GATED = {"choice": "vla", "reason": "VLM plan: not pick-and-place"}
SAM_FEATURES = [*s.PLANNER_FEATURES, "grasp_sam3_score", "place_sam3_score"]


def _record(episode, success, *, plan_ok=True, turn=0.5, sam=0.5, feasible=True, source="vlm"):
    """A planner-only record; with source vlm, its VLM plan (feasible: one move; else no first look at all)."""
    r = {
        "mode": "planner",
        "obj_episode_id": episode,
        "success": success,
        "features_version": s.FEATURES_VERSION,
        "features": {"grasp_sam3_score": sam, "place_sam3_score": sam, "grasp_turn_rad": turn, "grasp_reach_m": 0.1},
        "plans": [{"success": plan_ok, "time": 0.1, "ee_path_length": 0.45, "ee_straight_length": 0.3}],
        "depth_noise": "harsh",
        "task_source": source,
    }
    if source == "vlm":
        r.update(vlm_plan={"feasible": feasible, "steps": [], "error": None}, vlm_plan_feasible=feasible)
        if feasible:
            r["vlm_plan"]["steps"] = [{"object": "spoon", "target": "towel"}]
        else:
            r.update(success=False, features={}, plans=[])
    return r


def _write_planner(root, task, records):
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _write_video(root, task, episode, success):
    folder = root / task / "ckpt" / "scene"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{'success' if success else 'failure'}_obj_episode_{episode}_moved_True.mp4").write_bytes(b"")


def _write_set(root, name, episodes, rng, source="vlm"):
    """Planner-only and VLA-only runs of one layout set: the planner succeeds with a plan and a good SAM score, the VLA
    with a small turn; every 7th episode's VLM plan is not pick-and-place."""
    for task in s.TASKS:
        records = []
        for e in range(episodes):
            turn, sam, plan_ok = float(rng.uniform(0, 1.5)), float(rng.uniform(0, 1)), bool(rng.random() < 0.8)
            feasible = source == "rule" or e % 7 != 3
            records.append(
                _record(e, plan_ok and sam > 0.3, plan_ok=plan_ok, turn=turn, sam=sam, feasible=feasible, source=source)
            )
            if source == "vlm":
                _write_video(root / f"vla31_{name}", task, e, turn < 0.75)
        prefix = "vlmplan" if source == "vlm" else "pseed31"
        _write_planner(root / f"{prefix}_{name}_harsh", task, records)


def test_vlm_plan_gate():
    assert s.vlm_plan_gate(_record(0, True, source="rule")) is None
    assert s.vlm_plan_gate(_record(0, True)) is None
    assert s.vlm_plan_gate(_record(0, False, feasible=False)) == GATED
    assert s.vlm_plan_gate({}) is None  # a record from before the VLM plan


def test_grouped_folds_keep_each_episode_index_in_one_fold():
    episodes = np.tile(np.arange(30), 4)  # the 4 tasks of each episode index
    folds = s.grouped_folds(episodes, 10, seed=0)
    assert all(len(set(folds[episodes == e].tolist())) == 1 for e in range(30))
    assert np.bincount(folds).tolist() == [12] * 10
    np.testing.assert_array_equal(folds, s.grouped_folds(episodes, 10, seed=0))
    assert not np.array_equal(folds, s.grouped_folds(episodes, 10, seed=1))


def test_choose_features_takes_the_lowest_cross_validated_log_loss():
    rng = np.random.default_rng(0)
    sam = rng.uniform(0, 1, 200)
    records = [_record(e % 50, bool(v > 0.5), sam=float(v)) for e, v in enumerate(sam)]
    tasks, episodes = np.array([s.TASKS[e // 50] for e in range(200)]), np.arange(200) % 50
    y = np.array([float(r["success"]) for r in records])
    features, losses = s.choose_features(records, tasks, y, [s.PLANNER_FEATURES, SAM_FEATURES], episodes)
    assert features == SAM_FEATURES and losses[1] < losses[0]
    assert losses[0] == pytest.approx(s.cv_logloss(records, tasks, y, s.PLANNER_FEATURES, episodes))
    features, losses = s.choose_features(records, tasks, y, [["plan_ok"], ["plan_ok"]], episodes)
    assert features == ["plan_ok"] and losses[0] == losses[1]  # a tie: the first


def test_train_two_model_leaves_out_gated_episodes_and_uses_the_chosen_features(tmp_path, monkeypatch, capsys):
    _write_set(tmp_path, "iid11", 30, np.random.default_rng(1))
    trained = {}
    fit = s.fit_two_model

    def spy(records, *args, **kwargs):
        trained["episodes"] = [r["obj_episode_id"] for r in records]
        return fit(records, *args, **kwargs)

    monkeypatch.setattr(s, "fit_two_model", spy)
    path = tmp_path / "asm.json"
    candidates = ["plan_ok,plan_path_ratio", "plan_ok,plan_path_ratio,grasp_sam3_score,place_sam3_score"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selector",
            "--rule",
            "two-model",
            "--planner-runs",
            str(tmp_path / "vlmplan_iid11_harsh"),
            "--vla-runs",
            str(tmp_path / "vla31_iid11"),
            "--save",
            str(path),
            "--planner-candidates",
            *candidates,
        ],
    )
    s.main()
    out = capsys.readouterr().out
    assert "cross-validated log-loss" in out and "-> plan_ok, plan_path_ratio, grasp_sam3_score, place_sam3_score" in out
    assert len(trained["episodes"]) == 4 * (30 - 4) and 3 not in trained["episodes"]  # episodes 3, 10, 17, 24 gated
    asm = s.load_asm(str(path))
    assert asm.planner.features == SAM_FEATURES and asm.vla.features == s.VLA_FEATURES
    assert asm.trained_on["task_source"] == "vlm"


def test_planner_candidates_need_the_two_model_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["selector", "--planner-runs", str(tmp_path), "--vla-runs", str(tmp_path), "--planner-candidates", "plan_ok"],
    )
    with pytest.raises(SystemExit):
        s.main()


def test_check_closed_loop_names_a_choice_made_before_the_asm():
    assert check_closed_loop.probs(GATED) == "VLM plan: not pick-and-place"
    live = {f"selector_{k}": v for k, v in GATED.items()}
    assert check_closed_loop.probs(live, "selector_") == "VLM plan: not pick-and-place"
    assert check_closed_loop.probs({"p_planner": 0.25, "p_vla": 0.5, "choice": "vla"}) == "P 0.2500, P(VLA) 0.5000"


class FakeASM:
    def decide(self, stats, task):
        return {"choice": "planner", "reason": "asm"}


def test_client_gives_an_episode_with_no_pick_and_place_plan_to_the_vla():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner.planner_client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)  # no servers: only what the gate reads
    c.plan_check, c.episode_key, c.selector = None, f"{SPOON}/e15/0", FakeASM()
    for feasible, reason, phase in ((False, GATED["reason"], "vla"), (True, "asm", "approach")):
        c.phase, c.queue = "approach", [1]
        c.stats = {"features": {}, "task_source": "vlm", "vlm_plan_feasible": feasible}
        c._choose()
        assert c.stats["selector_reason"] == reason and c.phase == phase


def test_per_index_and_paired_take_one_value_per_episode_index():
    keys = [(t, e) for t in s.TASKS for e in (4, 9)]
    new, base = np.array([1.0] * 8), np.array([0.0, 1.0] * 4)  # episode 4 better in every task, episode 9 the same
    assert score_vlm_plan.per_index(new - base, keys).tolist() == [[1.0] * 4, [0.0] * 4]
    line, p, mean = score_vlm_plan.paired(new, base, keys)
    assert mean == 0.5 and p == 1.0  # one index with a nonzero sum: the sign-flip test cannot reject
    assert "1 episode indices up / 0 down" in line


def test_planned_tests_need_a_win_before_the_next_counts(capsys):
    st = score_vlm_plan
    keys = [(t, e) for t in s.TASKS for e in range(40)]
    ls = st.LayoutSet(keys, [{}] * 160, np.array([k[0] for k in keys]), np.ones(160), np.zeros(160))
    rows = {st.ASM: np.zeros(160), st.VLM_PLANNER: np.ones(160), st.VLA: np.zeros(160), "oracle": np.ones(160)}
    st.print_tests(rows, ls, planned=True)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("1. vs VLM -> planner") and lines[1].endswith("significant, but the ASM is lower")
    assert lines[2].endswith("does not count (a test before it was not a win)")


def test_score_vlm_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    rng = np.random.default_rng(2)
    _write_set(tmp_path, "iid11", 30, rng)
    _write_set(tmp_path, "iid15", 20, rng)
    _write_set(tmp_path, "iid11", 30, rng, source="rule")
    _write_set(tmp_path, "iid15", 20, rng, source="rule")
    for video in (tmp_path / "vla31_iid15" / CARROT).rglob("*_obj_episode_5_*.mp4"):
        video.unlink()  # episode index 5 loses a task
    monkeypatch.setattr(sys, "argv", ["score_vlm_plan"])
    score_vlm_plan.main()
    out = capsys.readouterr().out
    assert "## iid15, harsh depth (the planned tests)" in out
    assert "note: vlmplan_iid15_harsh: 1 episode indices without all 4 tasks, left out" in out
    assert "76 episodes, 19 episode indices:" in out
    assert "1. vs VLM -> planner (planner-only): " in out and "2. vs VLA-only: " in out
    assert "### The fixed rule (secondary)" in out and "The VLM plan minus the fixed rule (no claims):" in out
    assert "- planner-only: " in out and "- with the ASM: " in out
    assert "note: clean depth is left out: no vlmplan_iid11_none or vlmplan_iid15_none" in out

    # the offline choice is the selector mode's: the gate, then the ASM
    asm = s.train_two_model(
        str(tmp_path / "vlmplan_iid11_harsh"), [str(tmp_path / "vla31_iid11")], score_vlm_plan.CANDIDATES
    )
    test = score_vlm_plan.whole_indices(score_vlm_plan.load("vlmplan", "iid15", "harsh", score_vlm_plan.VLA_RUNS), "")
    use, gated, _, _ = score_vlm_plan.asm_choice(asm, test)
    live = [(s.vlm_plan_gate(r) or asm.decide(r, t))["choice"] for r, t in zip(test.records, test.tasks, strict=True)]
    assert use.tolist() == [c == "planner" for c in live]
    assert gated.sum() == 4 * 3  # episodes 3, 10 and 17 of each task
