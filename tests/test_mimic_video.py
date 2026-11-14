"""mimic-video in our evaluator: the server's sessions, the eval client and the runners' POLICY=mimic.

Run from the repository root with `pytest tests/test_mimic_video.py`. No GPU, no simulator and no mimic-video: a
made-up policy stands in for mimic-video's inference class, and a made-up simulator for the robot.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import mimic_video_client as mvc
from examples.SimplerEnv.eval_files import mimic_video_server as mvs

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "examples/SimplerEnv/eval_files/planner/experiments"
needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="the runners are bash scripts"
)
IMAGE = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
TASK = "put the spoon on the towel"


class FakePolicy:
    """Stands in for mimic-video's VAMInference: a new action chunk at every 5th step, and what each step got."""

    made = 0

    def __init__(self):
        FakePolicy.made += 1
        self.number = FakePolicy.made
        self.action_buffer = None
        self.seen = []

    def reset(self, task_description):
        self.task_description = task_description
        self.action_buffer = None
        self.index = 0

    def step(self, image, task_description, ee_pose_proprio, gripper_proprio):
        self.seen.append((image, task_description, ee_pose_proprio.p, ee_pose_proprio.q, gripper_proprio))
        if self.action_buffer is None:
            self.action_buffer, self.index = np.full((16, 10), self.number, dtype=np.float32), 0
        self.index += 1
        if self.index >= 5:
            self.action_buffer = None
        return {
            "world_vector": np.array([0.01 * self.number, 0.0, 0.0]),
            "rot_axangle": np.zeros(3, dtype=np.float32),
            "gripper": np.array([1.0], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }


def step_request(**changes):
    request = {"image": IMAGE, "task": TASK, "ee_p": np.zeros(3, np.float32), "ee_q": np.array([1, 0, 0, 0], np.float32)}
    return {**request, "gripper": 1.0, **changes}


def test_each_session_is_an_episode_of_its_own():
    made = {}
    sessions = mvs.Sessions(lambda: made.setdefault(len(made), FakePolicy()))
    assert sessions.predict_action(op="reset", session="a", task=TASK) == {}
    assert sessions.predict_action(op="reset", session="b", task="another task") == {}
    a, b = made[0], made[1]
    assert (a.task_description, b.task_description) == (TASK, "another task")
    answers = [sessions.predict_action(op="step", session="a", **step_request()) for _ in range(6)]
    assert [x["new_chunk"] for x in answers] == [True, False, False, False, False, True]  # one model call in 5 steps
    assert len(a.seen) == 6 and b.seen == []  # the other session got nothing
    answer = sessions.predict_action(op="step", session="b", **step_request(task="another task", gripper=0.25))
    assert answer["new_chunk"] and b.seen[0][1] == "another task" and b.seen[0][4] == 0.25
    assert set(answer) == {*mvs.ACTION_KEYS, "new_chunk", "seconds"}
    assert answer["world_vector"][0] == pytest.approx(0.01 * b.number) and answer["gripper"].dtype == np.float32
    # the policy gets the pose as mimic-video's loop gives it: an object with the position p and the quaternion q
    image, task, p, q, gripper = a.seen[0]
    assert image is IMAGE and task == TASK and p.dtype == np.float32 and list(q) == [1, 0, 0, 0] and gripper == 1.0


def test_a_session_ends_and_the_oldest_goes_when_there_are_too_many():
    sessions = mvs.Sessions(FakePolicy, limit=2)
    for name in "abc":
        sessions.predict_action(op="reset", session=name, task=TASK)
    assert len(sessions) == 2
    with pytest.raises(KeyError, match="reset comes first"):
        sessions.predict_action(op="step", session="a", **step_request())
    sessions.predict_action(op="step", session="c", **step_request())
    assert sessions.predict_action(op="end", session="c") == {} and len(sessions) == 1
    assert sessions.predict_action(op="end", session="c") == {}  # ending twice is no error
    with pytest.raises(ValueError, match="unknown request"):
        sessions.predict_action(op="infer", session="b")
    # a reset of a session that is there starts the episode again with a new policy
    first = sessions._sessions["b"]
    sessions.predict_action(op="reset", session="b", task=TASK)
    assert sessions._sessions["b"] is not first and len(sessions) == 1


def test_the_checkpoints_files_default_to_the_bridge_model(tmp_path):
    # the names are the ones of mimic-video's eval/bridge/eval.sh ("ftcosmos")
    assert mvs.EXPERIMENT == (
        "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
    )
    with pytest.raises(SystemExit, match=r"missing: .*video_backbone.*action_decoder.*bridge\.json"):
        mvs.checkpoint_files(tmp_path, mvs.EXPERIMENT, None, None, None)
    # a short experiment name here: the Bridge model's own makes a path that is too long for a Windows test folder
    names = [
        f"video_backbone/{mvs.VIDEO_MODEL}.pt",
        f"action_decoder/mine_{mvs.ACTION_ITERATION}.pt",
        "dataset_statistics/bridge.json",
    ]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    files = mvs.checkpoint_files(tmp_path, "mine", None, None, None)
    assert [files[k] for k in ("video_model", "action_model", "stats")] == [tmp_path / n for n in names]
    other = tmp_path / "other.pt"
    other.write_text("x")
    assert mvs.checkpoint_files(tmp_path, "no_such_experiment", None, str(other), None)["action_model"] == other


class Server:
    """Stands in for the websocket connection and the server: a request goes to the sessions as the server routes
    it, and an error becomes the server's error answer. With msgpack, each message is also packed and unpacked."""

    def __init__(self, sessions, pack=False):
        self.sessions, self.pack, self.requests = sessions, pack, []

    def predict_action(self, request):
        self.requests.append(request)
        if self.pack:
            from deployment.model_server.tools import msgpack_numpy

            request = msgpack_numpy.unpackb(msgpack_numpy.packb(request))
        try:
            answer = {"status": "ok", "ok": True, "data": self.sessions.predict_action(**request)}
        except Exception as e:
            answer = {"status": "error", "ok": False, "error": {"message": str(e)}}
        if self.pack:
            answer = msgpack_numpy.unpackb(msgpack_numpy.packb(answer))
        return answer


def robot(p=(0.3, 0.0, 0.1), q=(0.0, 0.0, 0.0, 1.0), fingers=(0.030, 0.032), mode=mvc.CONTROL_MODE):
    pose = SimpleNamespace(p=np.array(p, np.float32), q=np.array(q, np.float32))
    controllers = {
        "arm": SimpleNamespace(ee_pose_at_base=pose),
        "gripper": SimpleNamespace(qpos=np.array(fingers, np.float32)),
    }
    agent = SimpleNamespace(control_mode=mode, controllers={mode: SimpleNamespace(controllers=controllers)})
    return SimpleNamespace(unwrapped=SimpleNamespace(agent=agent))


def client_and_policies(pack=False):
    policies = []

    def make():
        policies.append(FakePolicy())
        return policies[-1]

    server = Server(mvs.Sessions(make), pack=pack)
    return mvc.MimicVideoClient(client=server), server, policies


def test_the_client_sends_the_picture_and_the_robots_state_and_gets_the_simulators_action():
    client, server, policies = client_and_policies()
    client.reset(TASK)
    raw, action = client.step(IMAGE, TASK, obs={"unused": 1}, env=robot(), info=None)
    image, task, p, q, gripper = policies[0].seen[0]
    assert image.dtype == np.uint8 and np.array_equal(image, IMAGE) and task == TASK
    assert p.dtype == np.float32 and np.array_equal(p, np.array([0.3, 0.0, 0.1], np.float32))
    assert q.dtype == np.float32 and list(q) == [0.0, 0.0, 0.0, 1.0]
    # the gripper's opening exactly as mimic-video's eval loop computes it
    fingers = np.array([0.030, 0.032], np.float32)
    assert gripper == (fingers.mean() - 0.015) / (0.037 - 0.015) and isinstance(gripper, float)
    assert set(action) == {"world_vector", "rot_axangle", "gripper", "terminate_episode"}
    assert action["gripper"].tolist() == [1.0] and action["terminate_episode"].tolist() == [0.0]
    assert raw["open_gripper"].tolist() == [1.0] and raw["rotation_delta"] is action["rot_axangle"]
    for _ in range(5):
        client.step(IMAGE, TASK, env=robot())
    assert client.chunks == 2 and len(policies[0].seen) == 6
    assert [r["op"] for r in server.requests] == ["reset"] + ["step"] * 6


def test_the_client_starts_a_session_for_each_episode_and_ends_it():
    client, server, policies = client_and_policies()
    client.reset(TASK)
    first = client.session
    client.step(IMAGE, TASK, env=robot())
    client.finish_episode(True, {})
    assert client.session is None and len(server.sessions) == 0
    client.finish_episode(True, {})  # no session: no request
    client.reset(TASK)
    assert client.session != first and len(policies) == 2
    client.reset("another task")  # a reset in an episode ends the session before
    assert len(server.sessions) == 1 and policies[2].task_description == "another task"
    client.step(IMAGE, TASK, env=robot())  # another instruction than the session's: a new session, as a new episode
    assert len(policies) == 4 and policies[3].task_description == TASK and len(policies[3].seen) == 1
    assert len(server.sessions) == 1
    client.step(IMAGE, None, env=robot())  # no instruction: the session's own
    assert len(policies) == 4 and policies[3].seen[1][1] == TASK


def test_the_client_stops_on_what_mimic_video_cannot_use():
    client, server, _ = client_and_policies()
    client.reset(TASK)
    with pytest.raises(ValueError, match="needs the simulator"):
        client.step(IMAGE, TASK)
    with pytest.raises(ValueError, match="control mode"):
        client.step(IMAGE, TASK, env=robot(mode="arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos"))
    server.sessions._sessions.clear()  # the server lost the session (it stopped and started again)
    with pytest.raises(RuntimeError, match="reset comes first"):
        client.step(IMAGE, TASK, env=robot())


def test_the_numbers_keep_their_types_through_the_servers_messages():
    pytest.importorskip("msgpack")
    client, _, policies = client_and_policies(pack=True)
    client.reset(TASK)
    _, action = client.step(IMAGE, TASK, env=robot())
    image, _, p, q, gripper = policies[0].seen[0]
    assert image.dtype == np.uint8 and image.shape == (480, 640, 3) and np.array_equal(image, IMAGE)
    assert p.dtype == np.float32 and q.dtype == np.float32 and isinstance(gripper, float)
    assert action["world_vector"].dtype == np.float64 and action["rot_axangle"].dtype == np.float32
    assert action["gripper"].dtype == np.float32 and action["gripper"].shape == (1,)


def test_the_action_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    client, _, _ = client_and_policies()
    client.reset(TASK)
    raws = [client.step(IMAGE, TASK, env=robot())[0] for _ in range(3)]
    client.visualize_epoch(raws, [IMAGE] * 4, str(tmp_path / "actions.png"))
    assert (tmp_path / "actions.png").stat().st_size > 1000


FAKE_PROGRAM = """#!/bin/bash
# Stands in for every Python of the runners: it writes its arguments to $CALLS.
echo "$*" >> "$CALLS"
"""
RUNNER = """
port_in_use() { return 1; }
wait_port() { echo "wait_port $1" >> "$CALLS"; }
LOG=$RUNS_DIR/status.log
mkdir -p "$RUNS_DIR/run"
fast_defaults
echo "VP_ARGS=[$VP_ARGS] SIM_ARGS=[$SIM_ARGS] POLICY_ARGS=[$POLICY_ARGS] MPS=${MPS:-1}" >> "$CALLS"
echo "CKPT=$(basename "$CKPT")" >> "$CALLS"
SERVER_ENV=(CUDA_VISIBLE_DEVICES=0)
start_vla_servers 10700 "$RUNS_DIR/run" && wait_vla_servers 10700 || exit 3
wait "${SERVER_PIDS[@]}"
sim 0 PutSpoonOnTableClothInScene-v0 0 1 "$RUNS_DIR/run/task" --controller vla --port 10700 $VP_ARGS $SIM_ARGS || exit 4
"""


def run_runner(tmp_path, **settings):
    fake = tmp_path / "python"
    fake.write_text(FAKE_PROGRAM, newline="\n")
    fake.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "RUNS_DIR": str(tmp_path / "runs"),
        "SIMPLERENV_PATH": str(tmp_path),
        "CALLS": str(calls),
        **{name: str(fake) for name in ("STARVLA_PYTHON", "SAM3_PYTHON", "SAM31_PYTHON", "SIMPLER_PYTHON")},
        **settings,
    }
    script = f"source '{EXPERIMENTS}/common.sh' || exit 9\n{RUNNER}\n"
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, cwd=tmp_path)
    return out, calls.read_text().splitlines() if calls.exists() else []


@needs_bash
def test_policy_mimic_starts_mimic_videos_server_alone_and_its_client(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    settings = {"POLICY": "mimic", "MIMIC_PYTHON": str(tmp_path / "python"), "MIMIC_CHECKPOINTS": str(checkpoints)}
    out, calls = run_runner(tmp_path, EPISODE_STEPS="60", POLICY_ARGS="--stop-step 2", **settings)
    assert out.returncode == 0, out.stderr
    assert calls[:2] == [
        "VP_ARGS=[--policy-client mimic_video] SIM_ARGS=[--no-action-plots] POLICY_ARGS=[--stop-step 2] MPS=0",
        "CKPT=mimic-video",
    ]
    server = (
        f"-m examples.SimplerEnv.eval_files.mimic_video_server --checkpoints {checkpoints} --port 10700 --stop-step 2"
    )
    assert sorted(calls[2:4]) == sorted([server, "wait_port 10700"])  # one server, one port
    sim = calls[4]
    assert len(calls) == 5 and sim.startswith("examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py ")
    assert "--ckpt-path mimic-video " in sim and "--max-episode-steps 60 " in sim
    assert sim.endswith("--controller vla --port 10700 --policy-client mimic_video --no-action-plots")
    assert "--use-sam3" not in sim and "--use-vlm" not in sim
    # the settings are needed
    out, _ = run_runner(tmp_path, POLICY="mimic", MIMIC_PYTHON=str(tmp_path / "python"))
    assert out.returncode != 0 and "MIMIC_CHECKPOINTS" in out.stderr
    out, _ = run_runner(tmp_path, POLICY="mimic", MIMIC_CHECKPOINTS=str(checkpoints))
    assert out.returncode != 0 and "MIMIC_PYTHON" in out.stderr


@needs_bash
def test_without_policy_mimic_the_runners_are_as_before(tmp_path):
    out, calls = run_runner(tmp_path)
    assert out.returncode == 0, out.stderr
    settings = " ".join(calls[:2])  # VP-VLA's visual-prompt arguments are on two lines
    assert settings.startswith("VP_ARGS=[--use-sam3 ") and "--use-vlm" in settings
    assert "SIM_ARGS=[--sam3-batch-prompts --no-action-plots] POLICY_ARGS=[--attn sdpa --compile] MPS=1" in settings
    assert [c for c in calls if c.startswith("wait_port")] == ["wait_port 10700", "wait_port 10701", "wait_port 10702"]
    servers = [c for c in calls[2:] if "server" in c and not c.startswith("wait_port")]
    assert len(servers) == 3 and not any("mimic_video" in c for c in calls)
    assert "--max-episode-steps 120 " in calls[-1] and "--policy-client" not in calls[-1]


@needs_bash
def test_the_runners_refuse_what_does_not_go_with_mimic_video(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    fake = tmp_path / "python"
    fake.write_text(FAKE_PROGRAM, newline="\n")
    fake.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "RUNS_DIR": str(tmp_path / "runs"),
        "SIMPLERENV_PATH": str(tmp_path),
        "CALLS": str(tmp_path / "calls.txt"),
        **{name: str(fake) for name in ("STARVLA_PYTHON", "SAM3_PYTHON", "SAM31_PYTHON", "SIMPLER_PYTHON")},
        "POLICY": "mimic",
        "MIMIC_PYTHON": str(fake),
        "MIMIC_CHECKPOINTS": str(checkpoints),
    }
    for runner, arguments, setting, message in [
        ("vla_eval.sh", ["0", "10700", "name", "iid98", "1"], {"HIDDEN": "1"}, "HIDDEN, PERTURB or ACTION_INSTRUCTION"),
        ("vla_eval.sh", ["0", "10700", "name", "iid98", "1"], {"ACTION_INSTRUCTION": "subtask"}, "ACTION_INSTRUCTION"),
        ("selector_eval.sh", ["0", "10700", "name", "iid98", "m_{}.json", "1"], {"PLAN_CHECK": "1"}, "no VLM server"),
        ("selector_eval.sh", ["0", "10700", "name", "iid98", "m_{}.json", "1"], {"TASK_SOURCE": "vlm"}, "no VLM server"),
    ]:
        out = subprocess.run(
            ["bash", str(EXPERIMENTS / runner), *arguments],
            env={**env, **setting},
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert out.returncode == 1 and message in out.stderr, (runner, setting, out.stderr)
    assert not (tmp_path / "calls.txt").exists()  # nothing was started
