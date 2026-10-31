"""CPU tests for the planner's corrections of the VLA: planner_client.py's VLA-then-planner mode, the demonstration's
first step (demos.py, planner_demos_to_lerobot.py) and selector_eval.sh's CONTROLLER=vla_then_planner."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import demos
from examples.SimplerEnv.train_files import planner_demos_to_lerobot as tl

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/SimplerEnv/eval_files/planner/experiments/selector_eval.sh"
CARROT = "PutCarrotOnPlateInScene-v0"
KEY = f"{CARROT}/e15/3"
TASK = "put carrot on plate"
SCENE = {"objects": ["bridge_carrot_generated_modified", "bridge_plate_objaverse_larger"], "goals": [[0, 1]]}
IMAGE = np.zeros((48, 64, 3), np.uint8)
HOME = np.array([1.0, 0.0, 0.0, 0.0])  # the hand's orientation at the start (w, x, y, z)
needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="the runner is a bash script"
)


class FakeVLA:
    """The VLA: a small move at each step, with the gripper commands of `grippers` (+1 open, -1 closed) in turn."""

    def __init__(self, grippers=(1.0,)):
        self.grippers, self.steps = grippers, 0

    def reset(self, task_description):
        self.steps = 0

    def step(self, image, task_description=None):
        gripper = self.grippers[min(self.steps, len(self.grippers) - 1)]
        self.steps += 1
        move = {"world_vector": np.array([0.01, 0.0, 0.0]), "rot_axangle": np.zeros(3)}
        return {"from": "vla"}, {**move, "gripper": np.array([gripper]), "terminate_episode": np.zeros(1)}


@pytest.fixture
def pc(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner import planner_client

    monkeypatch.setattr(planner_client, "make_planner", lambda name: SimpleNamespace(name=name))  # no MPlib
    return planner_client


def _client(pc, vla, takeover=(3, 3)):
    c = pc.PlannerClient(vla=vla, target_source="sim", takeover_steps=takeover)
    c.set_episode(KEY)
    c.reset(TASK)
    return c


def _obs(z=0.10, fingers=0.037, quat=HOME):
    """The robot's base at the origin, the hand at (0.3, 0, z) and the controller's target on it."""
    pose = np.r_[0.3, 0.0, z, quat]
    return {
        "agent": {
            "qpos": np.r_[np.zeros(6), fingers, fingers],
            "base_pose": np.r_[np.zeros(3), HOME],
            "controller": {"arm": {"target_pose": pose}},
        },
        "extra": {"tcp_pose": pose},
    }


ENV = SimpleNamespace(unwrapped=SimpleNamespace(agent=SimpleNamespace(robot=SimpleNamespace(get_links=lambda: []))))


def _drive(c, steps, obs, info=None):
    return [c.step(IMAGE, TASK, obs=obs, env=ENV, info=info) for _ in range(steps)]


def test_the_takeover_step_is_drawn_per_episode(pc):
    c = _client(pc, FakeVLA(), takeover=(10, 20))
    drawn = {}
    for e in range(200):
        c.set_episode(f"{CARROT}/e15/{e}")
        c.reset(TASK)
        drawn[e] = c.stats["takeover_step"]
    assert set(drawn.values()) == set(range(10, 21))  # every step from LO to HI
    c.set_episode(f"{CARROT}/e15/7")
    c.reset(TASK)
    assert c.stats["takeover_step"] == drawn[7] and c.stats["takeover"] is None  # the same episode, the same step


def test_the_mode_needs_a_vla_and_the_fixed_rule(pc):
    selector = {"selector": SimpleNamespace(trained_on={}), "robot_mask": "joints"}
    for settings, error in [
        ({"vla": None}, "needs a VLA"),
        (selector, "no selector"),
        ({"takeover_steps": (0, 5)}, "1 <= LO <= HI"),
        ({"takeover_steps": (9, 5)}, "1 <= LO <= HI"),
        ({"takeover_steps": (5,)}, "1 <= LO <= HI"),
        ({"task_source": "vp"}, "fixed rule"),
        ({"plan_check_port": 10512}, "fixed rule"),
    ]:
        kwargs = {"vla": FakeVLA(), "target_source": "sim", "takeover_steps": (10, 20), **settings}
        with pytest.raises(ValueError, match=error):
            pc.PlannerClient(**kwargs)
    assert pc.PlannerClient(vla=FakeVLA(), target_source="sim").demo_info() == {}  # the other modes record it all


def test_the_vla_drives_until_the_takeover_step(pc, monkeypatch):
    vla = FakeVLA()
    c = _client(pc, vla)
    approaches = []
    monkeypatch.setattr(c, "_approach", lambda *args: approaches.append((c.gripper, c.first_look)))
    turned = np.array([0.0, 0.0, 0.0, 1.0])  # the VLA turned the hand about the vertical
    out = _drive(c, 1, _obs()) + _drive(c, 2, _obs(quat=turned), info={"success": False})
    assert [raw for raw, _ in out] == [{"from": "vla"}] * 3 and c.phase == "vla_first" and not approaches
    raw, _ = c.step(IMAGE, TASK, obs=_obs(quat=turned), env=ENV, info={"success": False})
    assert vla.steps == 3 and "from" not in raw  # the planner's step
    # No object in the hand: the whole pick-and-place, from an open hand and with the first look.
    assert approaches == [(pc.GRIPPER_OPEN, True)] and c.stats["takeover"] == "grasp"
    np.testing.assert_array_equal(c.home_orientation, HOME)  # the start's, as in the other modes
    assert c.demo_info() == {"controller": "vla_then_planner", "takeover": "grasp", "first_step": 3}


def test_fingers_closed_on_nothing_grasp_again(pc, monkeypatch):
    c = _client(pc, FakeVLA(grippers=(-1.0,)))
    approaches = []
    monkeypatch.setattr(c, "_approach", lambda *args: approaches.append(c.gripper))
    _drive(c, 4, _obs(fingers=0.015))  # fully closed: nothing between the fingers
    assert approaches == [pc.GRIPPER_OPEN] and c.stats["takeover"] == "grasp" and c.stats["takeover_grip"] == 0.015


def test_the_held_object_is_lifted_and_then_placed(pc, monkeypatch):
    c = _client(pc, FakeVLA(grippers=(1.0, 1.0, -1.0)))
    located = []

    def locate(image, obs, env, which, record=False, view=None):
        located.append(which)
        return {"center": np.array([0.3, 0.0]), "top": 0.06, "bottom": 0.03}

    monkeypatch.setattr(c, "_obstacle_view", lambda obs, qpos: SimpleNamespace())
    monkeypatch.setattr(c, "_locate", locate)
    *_, (_, action) = _drive(c, 4, _obs(fingers=0.02))
    assert c.stats["takeover"] == "holding" and located == ["grasp"] and not c.first_look
    assert c.phase == "lift" and c.gripper == pc.GRIPPER_CLOSED and action["gripper"].tolist() == [-1.0]
    assert 0 < action["world_vector"][2] <= 0.02 + 1e-9 and np.allclose(action["world_vector"][:2], 0)  # up
    np.testing.assert_allclose(c.queue[-1][:3], [0.3, 0.0, 0.18])  # to 8 cm above where the VLA left the hand
    assert c.hold_offset == pytest.approx(0.07)  # the hand 7 cm above the held object's bottom
    assert c.demo_info()["first_step"] == 3

    # A held object that the camera does not find ends the episode.
    c = _client(pc, FakeVLA(grippers=(-1.0,)))
    monkeypatch.setattr(c, "_obstacle_view", lambda obs, qpos: SimpleNamespace())
    monkeypatch.setattr(c, "_locate", lambda *args, **kwargs: None)
    *_, (_, action) = _drive(c, 4, _obs(fingers=0.02))
    assert c.phase == "done" and c.stats["failure"] == "held object not found" and action["terminate_episode"][0] == 1


def test_no_takeover_when_the_task_is_already_done(pc):
    vla = FakeVLA()
    c = _client(pc, vla)
    out = _drive(c, 3, _obs()) + _drive(c, 2, _obs(), info={"success": True})
    assert [raw for raw, _ in out] == [{"from": "vla"}] * 5 and c.phase == "vla"
    assert c.stats["takeover"] == "skipped" and c.demo_info()["first_step"] is None


def _record(root, grippers, first_step, instruction="put carrot on plate"):
    """A recorded episode whose steps' pictures, object maps and labels are numbered by the step."""
    written = {}

    def fake_video(path, frames, fps):
        written["frames"] = [int(f[0, 0, 0]) for f in frames]
        with open(path, "wb") as fh:
            fh.write(b"mp4")

    r = demos.DemoRecorder(str(root), write_video=fake_video)
    r.start(KEY, env_name=CARROT, obj_episode_id=3, layout_seed=15, layout_per_episode=True, **SCENE)
    labels = []
    for t, gripper in enumerate(grippers):
        labels.append(np.r_[0.001 * t, 0.0, 0.0, 0.0, 0.0, 0.0, gripper])
        hand = {"agent": {"base_pose": np.r_[np.zeros(3), HOME]}, "extra": {"tcp_pose": np.r_[0.3, 0.0, 0.1, HOME]}}
        object_map = np.zeros((48, 64), np.uint8)
        object_map[10:14, 5 + t : 9 + t] = 1  # the carrot
        object_map[30:40, 40:60] = 2  # the plate
        r.add(np.full((48, 64, 3), t, np.uint8), hand, demos.action_from_label(labels[-1]), instruction, object_map)
    folder = r.finish(True, first_step=first_step, controller="vla_then_planner")
    return folder, np.array(labels), written


def test_the_recording_is_the_planner_part_only(tmp_path):
    grippers = (1, 1, 0, 0, 0, 1, 1, 1)  # the VLA's 3 steps close the hand; the planner's 5 place and let go
    folder, labels, written = _record(tmp_path / "a", grippers, first_step=3)
    assert written["frames"] == [3, 4, 5, 6, 7]
    with np.load(os.path.join(folder, "episode.npz")) as z:
        np.testing.assert_allclose(z["actions"], labels[3:], atol=1e-6)
        assert z["states"][0, 7] == 0.0  # the state's gripper: the VLA's last command, closed
        assert z["object_map"].shape == (5, 48, 64) and (z["object_map"][0] == 1).any()
    meta = json.loads(open(os.path.join(folder, "meta.json")).read())
    assert (meta["steps"], meta["first_step"], meta["episode_steps"]) == (5, 3, 8)
    assert len(meta["instructions"]) == 5 and meta["controller"] == "vla_then_planner"

    # No takeover: no step is a demonstration, and the training set leaves the episode out.
    folder, _, written = _record(tmp_path / "b", grippers, first_step=None)
    meta = json.loads(open(os.path.join(folder, "meta.json")).read())
    assert (meta["steps"], meta["first_step"], meta["episode_steps"]) == (0, None, 8) and not written
    assert tl.find_episodes([str(tmp_path / "b")], include_failures=True) == []
    assert tl.find_episodes([str(tmp_path)]) == [os.path.normpath(os.path.join(str(tmp_path / "a"), KEY))]
    with pytest.raises(ValueError, match="first step 9 of 8"):
        _record(tmp_path / "c", grippers, first_step=9)


def test_a_recording_that_starts_holding_the_object_starts_after_the_pick_part():
    def actions(*grippers):
        return np.array([[0.0] * 6 + [g] for g in grippers])

    def entries_of(instruction, acts):
        parts = tl.instruction_parts(instruction)
        return tl.vlm_entries(parts, acts, [("carrot", "plate")] * len(parts))

    two_parts = "pick up the carrot and put it on the plate"
    entries = entries_of(two_parts, actions(0, 0, 0, 1, 1))
    assert [(e["frame"], e["subtask"]) for e in entries] == [(0, "put it on the plate"), (3, "put it on the plate")]
    assert entries[0]["target_location"] == "plate" and entries[0]["decision"] == "continue"
    # An open start is as before: the pick part first.
    entries = entries_of(two_parts, actions(1, 1, 0, 0, 1))
    assert [(e["frame"], e["subtask"]) for e in entries] == [
        (0, "pick up the carrot"),
        (2, "put it on the plate"),
        (4, "put it on the plate"),
    ]
    # One part that names the place stays that part.
    entries = entries_of("put the carrot on the plate", actions(0, 0, 1))
    assert [e["subtask"] for e in entries] == ["put the carrot on the plate"] * 2


def test_the_training_set_of_a_correction(tmp_path):
    pytest.importorskip("pyarrow")
    _record(tmp_path / "run", (1, 1, 0, 0, 0, 1, 1, 1), first_step=3)
    out, vp = tmp_path / "data" / "corrections", tmp_path / "vp"
    summary = tl.write_dataset(str(out), tl.find_episodes([str(tmp_path / "run")]), "data", str(vp))
    assert (summary["episodes"], summary["frames"]) == (1, 5)
    source = json.loads((out / "meta" / "sources.jsonl").read_text())
    assert source["first_step"] == 3 and source["episode_key"] == KEY
    with np.load(vp / "corrections" / "chunk-000" / "episode_000000.npz") as z:
        assert int(z["num_frames"]) == 5 and z["target_masks"].shape == (5, 48, 64)
        entries = json.loads(str(z["vlm_results"]))
        assert entries[0]["frame"] == 0 and entries[0]["target_location"] == "plate"  # held: the place from the start
        assert z["location_scores"].tolist() == [1.0] * 5


@needs_bash
def test_the_runner_records_corrections_only_as_asked(tmp_path):
    true = shutil.which("true")
    env = {
        **os.environ,
        **dict.fromkeys(("STARVLA_PYTHON", "SAM3_PYTHON", "SAM31_PYTHON", "SIMPLER_PYTHON"), true),
        "SIMPLERENV_PATH": str(tmp_path),
        "RUNS_DIR": str(tmp_path / "runs"),
        "CONTROLLER": "vla_then_planner",
    }
    for settings, models, error in [
        ({}, "-", 'needs TAKEOVER_STEPS="LO HI"'),
        ({"TAKEOVER_STEPS": "10"}, "-", "needs TAKEOVER_STEPS"),
        ({"TAKEOVER_STEPS": "10 40"}, "two_model", "MODELS must be -"),
        ({"TAKEOVER_STEPS": "10 40", "TASK_SOURCE": "vp"}, "-", "no TASK_SOURCE or PLAN_CHECK"),
        ({"TAKEOVER_STEPS": "10 40", "PLAN_CHECK": "1"}, "-", "no TASK_SOURCE or PLAN_CHECK"),
    ]:
        args = ["bash", str(RUNNER), "0", "10510", "corrections", "iid15", models, "1"]
        result = subprocess.run(args, env={**env, **settings}, capture_output=True, text=True, timeout=60)
        assert result.returncode != 0 and error in result.stderr, (settings, result.stderr)
    assert not (tmp_path / "runs").exists()
