"""CPU tests for the VLM-router study: the VLM router's questions (planner/vlm_router.py), the planner client's
first-look questions, and the study's scorer (experiments/score_vlm_router.py)."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner import vlm_router as vr
from examples.SimplerEnv.eval_files.planner.experiments import score_vlm_router

SPOON, CARROT = s.TASKS[0], s.TASKS[1]
TASK = "put carrot on plate"


def test_features_and_questions():
    assert vr.feature("planner", "described") == "vlm_router_planner_yes"
    assert vr.feature("vla", "rates") == "vlm_router_vla_yes_rates"
    q = vr.question(TASK, "planner", "described")
    assert q.startswith(f'The robot is told: "{TASK}"') and "this depth camera is very noisy" in q
    assert q.endswith("Will the motion planner complete the instruction? Answer yes or no.")
    assert "earlier tries" not in q
    q = vr.question(TASK, "vla", "rates", CARROT)
    assert "In 100 earlier tries of this task, the motion planner completed it 51 times and the VLA 63 times." in q
    assert q.endswith("Will the VLA complete the instruction? Answer yes or no.")
    assert set(vr.TRAINING_SUCCESSES) == set(s.TASKS)
    for args, error in [
        (("planner", "rates", None), "SimplerEnv's 4 tasks"),
        (("planner", "rates", "NovelThreeStep-v0"), "SimplerEnv's 4 tasks"),
        (("robot", "described", None), "method must be"),
        (("planner", "tuned", None), "version must be"),
    ]:
        with pytest.raises(ValueError, match=error):
            vr.question(TASK, *args)


def test_choose_and_scores():
    assert vr.choose([0.5, 0.4, 0.6], [0.5, 0.5, 0.5]).tolist() == [True, False, True]  # a tie goes to the planner
    record = {"features": {"vlm_router_planner_yes": 0.7, "vlm_router_vla_yes": 0.4}}
    assert vr.scores(record, "described") == (0.7, 0.4)
    with pytest.raises(ValueError, match="--vlm-router-port"):
        vr.scores(record, "rates")


class FakeVLM:
    """Says yes (0.8) about the VLA and no (0.3) about the planner; keeps the questions."""

    def __init__(self):
        self.questions = []

    def yes_probability(self, images, question):
        assert len(images) == 1  # the first frame only
        self.questions.append(question)
        return {
            "p_yes": 0.8 if question.endswith("Will the VLA complete the instruction? Answer yes or no.") else 0.3,
            "answer_mass": 0.95,
        }


@pytest.fixture
def pc(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner import planner_client

    monkeypatch.setattr(planner_client, "make_planner", lambda name: SimpleNamespace(name=name))  # no MPlib
    return planner_client


def test_client_asks_each_version_and_method(pc):
    c = pc.PlannerClient.__new__(pc.PlannerClient)  # no servers: only what the questions read
    c.vlm_router, c.task_description, c.episode_key = FakeVLM(), TASK, f"{CARROT}/None/3"
    c.stats = {"features": {"grasp_turn_rad": 0.2}}
    c._ask_router(np.zeros((20, 20, 3), np.uint8))
    names = [vr.feature(m, v) for v in vr.VERSIONS for m in vr.METHODS]
    assert c.stats["features"] == {"grasp_turn_rad": 0.2, **{n: 0.8 if "_vla_" in n else 0.3 for n in names}}
    assert c.stats["vlm_router_answer_mass"] == dict.fromkeys(names, 0.95)
    assert sum("earlier tries" in q for q in c.vlm_router.questions) == 2
    assert all(q.startswith(f'The robot is told: "{TASK}"') for q in c.vlm_router.questions)


def test_client_asks_only_in_planner_only_evals(pc):
    with pytest.raises(ValueError, match="planner-only evals, not the planner_then_vla mode"):
        pc.PlannerClient(vla=SimpleNamespace(), target_source="sim", vlm_router_port=10370)
    c = pc.PlannerClient(target_source="sim")
    c.set_episode(f"{CARROT}/None/0")
    c.reset(TASK)
    assert c.vlm_router is None and c.stats["vlm_router"] is False


def _write_planner(root, task, records):
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _write_video(root, task, episode, success):
    folder = root / task / "ckpt" / "scene"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{'success' if success else 'failure'}_obj_episode_{episode}_moved_True.mp4").write_bytes(b"")


def _write_set(root, name, episodes, rng, router=False):
    """The fixed rule's planner-only eval of one layout set (pseed31_{name}_harsh, or with router: router_{name}_harsh,
    with the VLM router's features) and its VLA-only eval: the planner succeeds with a plan and a small turn, the VLA
    with a large one. The described router's P(yes) follows the turn; the rates version always prefers the VLA."""
    for task in s.TASKS:
        records = []
        for e in range(episodes):
            turn, plan_ok = float(rng.uniform(0, 1.5)), bool(rng.random() < 0.8)
            r = {
                "mode": "planner",
                "obj_episode_id": e,
                "success": plan_ok and turn < 0.9,
                "features_version": s.FEATURES_VERSION,
                "features": {"grasp_sam3_score": 0.5, "place_sam3_score": 0.5, "grasp_turn_rad": turn},
                "plans": [{"success": plan_ok, "time": 0.1, "ee_path_length": 0.45, "ee_straight_length": 0.3}],
                "depth_noise": "harsh",
            }
            if router:
                r["features"].update(
                    vlm_router_planner_yes=1 - turn / 1.5,
                    vlm_router_vla_yes=turn / 1.5,
                    vlm_router_planner_yes_rates=0.2,
                    vlm_router_vla_yes_rates=0.6,
                )
                r["vlm_router_answer_mass"] = dict.fromkeys(
                    [vr.feature(m, v) for v in vr.VERSIONS for m in vr.METHODS], 0.9
                )
            records.append(r)
            if not router:
                _write_video(root / f"vla31_{name}", task, e, turn > 0.6)
        _write_planner(root / f"{'router' if router else 'pseed31'}_{name}_harsh", task, records)


def test_score_vlm_router(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    _write_set(tmp_path, "iid11", 30, np.random.default_rng(1))
    _write_set(tmp_path, "iid14", 20, np.random.default_rng(2))
    _write_set(tmp_path, "iid14", 20, np.random.default_rng(2), router=True)  # the same episodes, with the questions
    monkeypatch.setattr(sys, "argv", ["score_vlm_router"])
    score_vlm_router.main()
    out = capsys.readouterr().out
    assert "# VLM-router study: trained on iid11, tested on iid14, harsh depth" in out
    assert "note: Spoon: the questions say {'planner': 54, 'vla': 74} of 100, the training set has" in out
    assert "80 episodes, 20 episode indices:" in out
    for i, version in enumerate(("described", "rates"), 1):
        assert f"{i}. vs {score_vlm_router.ROUTER[version]}: " in out
    assert "Secondary tests (no claims):" in out and "Share of the oracle's gain" in out
    assert "Check: the planner's outcome is the same as in pseed31_iid14_harsh in 80 of 80 episodes" in out
    assert "- vlm_router_planner_yes: 0.900, 0.900" in out
    # the rates version always prefers the VLA here: no episode to the planner
    line = next(ln for ln in out.splitlines() if ln.startswith(f"| {score_vlm_router.ROUTER['rates']} | 0 | 0 |"))
    assert "| 0.0 |" in line

    # the router's choice is the two-model rule on the VLM's answers
    test = score_vlm_router.load("router", "iid14", "harsh", score_vlm_router.VLA_RUNS)
    use, p_plan, _ = score_vlm_router.router_choice(test.records, "described")
    turns = np.array([r["features"]["grasp_turn_rad"] for r in test.records])
    assert use.tolist() == (turns <= 0.75).tolist() and np.allclose(p_plan, 1 - turns / 1.5)

    for f in (tmp_path / "router_iid14_harsh").rglob("planner_stats.jsonl"):
        f.write_text(f.read_text().replace("vlm_router_vla_yes_rates", "x"))
    with pytest.raises(ValueError, match="--vlm-router-port"):
        score_vlm_router.main()
