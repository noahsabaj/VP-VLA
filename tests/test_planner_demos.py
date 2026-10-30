"""CPU tests for the recorded demonstrations (eval_files/planner/demos.py) and their training set
(train_files/planner_demos_to_lerobot.py)."""

import json
import os
import sys

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import demos
from examples.SimplerEnv.train_files import planner_demos_to_lerobot as tl

SPOON = "PutSpoonOnTableClothInScene-v0"
CARROT = "PutCarrotOnPlateInScene-v0"


def _label(rng):
    return np.concatenate([rng.uniform(-0.03, 0.03, 3), rng.uniform(-0.2, 0.2, 3), [float(rng.integers(0, 2))]])


def test_a_label_gives_back_its_command_and_the_command_its_label():
    rng = np.random.default_rng(0)
    for _ in range(200):
        label = _label(rng)
        command = demos.action_from_label(label)
        np.testing.assert_allclose(demos.label_from_action(command), label, atol=1e-9)
        again = demos.action_from_label(demos.label_from_action(command))
        for key in command:
            np.testing.assert_allclose(again[key], command[key], atol=1e-9)
    # the gripper: +1 open is 1, -1 closed is 0, and back
    assert demos.label_from_action({"world_vector": np.zeros(3), "rot_axangle": np.zeros(3), "gripper": [-1.0]})[6] == 0
    assert demos.action_from_label([0, 0, 0, 0, 0, 0, 1])["gripper"].tolist() == [1.0]
    assert demos.action_from_label([0, 0, 0, 0, 0, 0, 0])["gripper"].tolist() == [-1.0]


def test_the_angles_are_transforms3d_sxyz():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation  # scipy's "xyz" = transforms3d's "sxyz"
    rng = np.random.default_rng(1)
    for _ in range(50):
        rpy = rng.uniform(-1.2, 1.2, 3)
        np.testing.assert_allclose(demos.euler_to_matrix(rpy), rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-12)
        np.testing.assert_allclose(
            demos.matrix_to_axangle(demos.euler_to_matrix(rpy)), rotation.from_euler("xyz", rpy).as_rotvec(), atol=1e-9
        )
        np.testing.assert_allclose(demos.matrix_to_euler(demos.euler_to_matrix(rpy)), rpy, atol=1e-9)
    big = demos.axangle_to_matrix([0.0, 0.0, np.pi - 1e-9])  # near a half turn the quaternion route stays exact
    np.testing.assert_allclose(demos.axangle_to_matrix(demos.matrix_to_axangle(big)), big, atol=1e-7)


def _obs(tcp_xyz, yaw=0.0):
    half = yaw / 2
    return {
        "agent": {"base_pose": np.array([1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0])},
        "extra": {
            "tcp_pose": np.array(
                [1.0 + tcp_xyz[0], 2.0 + tcp_xyz[1], 0.5 + tcp_xyz[2], np.cos(half), 0, 0, np.sin(half)]
            )
        },
    }


def test_the_state_is_the_hand_in_the_base_frame():
    s = demos.state_from_obs(_obs([0.3, -0.1, 0.05], yaw=0.4), 1.0)
    np.testing.assert_allclose(s, [0.3, -0.1, 0.05, 0.0, 0.0, 0.4, 0.0, 1.0], atol=1e-9)


def _record(root, key, steps, success, rng, instruction="put the spoon on the towel"):
    written = {}

    def fake_video(path, frames, fps):
        written["frames"], written["fps"] = len(frames), fps
        with open(path, "wb") as fh:
            fh.write(b"mp4")

    r = demos.DemoRecorder(str(root), write_video=fake_video)
    task, _, episode = key.split("/")
    r.start(key, env_name=task, obj_episode_id=int(episode), layout_seed=15, layout_per_episode=True)
    labels = [_label(rng) for _ in range(steps)]
    for t, label in enumerate(labels):
        r.add(np.full((48, 64, 3), t, np.uint8), _obs([0.3, 0.0, 0.1]), demos.action_from_label(label), instruction)
    return r.finish(success, episode_stats={"moved": True}), labels, written


def test_the_recorder_writes_an_episode(tmp_path):
    folder, labels, written = _record(tmp_path, f"{SPOON}/iid15/3", 5, True, np.random.default_rng(2))
    assert os.path.normpath(folder) == os.path.join(str(tmp_path), SPOON, "iid15", "3")
    assert written == {"frames": 5, "fps": 5}
    with np.load(os.path.join(folder, "episode.npz")) as z:
        np.testing.assert_allclose(z["actions"], np.array(labels), atol=1e-6)
        # the state's gripper is the command before the step: open at the start
        assert z["states"].shape == (5, 8) and z["states"][0, 7] == 1.0
        np.testing.assert_allclose(z["states"][1:, 7], np.array(labels)[:-1, 6])
    meta = json.loads(open(os.path.join(folder, "meta.json")).read())
    assert meta["success"] and meta["steps"] == 5 and (meta["height"], meta["width"]) == (48, 64)
    assert meta["obj_episode_id"] == 3 and meta["episode_stats"] == {"moved": True}
    assert meta["instructions"] == ["put the spoon on the towel"] * 5


def test_the_training_set(tmp_path):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(3)
    runs = tmp_path / "run"
    _record(runs / "a", f"{SPOON}/iid15/0", 4, True, rng)
    _record(runs / "a", f"{SPOON}/iid15/1", 3, False, rng)  # a failure: left out
    _record(runs / "b", f"{CARROT}/iid15/0", 6, True, rng, instruction="put the carrot on the plate")
    stats_file = tmp_path / "dataset_statistics.json"
    bridge = {
        "mean": [0.0] * 7,
        "std": [1.0] * 7,
        "min": [-1.0] * 7,
        "max": [1.0] * 7,
        "q01": [-0.01] * 7,
        "q99": [0.01] * 7,
        "mask": [True] * 6 + [False],
    }
    state = {k: [0.0] * 8 for k in tl.STAT_NAMES}
    stats_file.write_text(json.dumps({"oxe_bridge": {"action": bridge, "state": state, "num_transitions": 9}}))
    folders = tl.find_episodes([str(runs)])
    assert len(folders) == 2 and len(tl.find_episodes([str(runs)], include_failures=True)) == 3
    out = tmp_path / "planner_demos"
    summary = tl.write_dataset(str(out), folders, str(stats_file))
    assert (summary["episodes"], summary["frames"], summary["tasks"]) == (2, 10, 2)
    meta = out / "meta"
    info = json.loads((meta / "info.json").read_text())
    assert info["total_frames"] == 10 and info["features"][tl.VIDEO_KEY]["shape"] == [48, 64, 3]
    assert json.loads((meta / "modality.json").read_text())["action"]["gripper"] == {"start": 6, "end": 7}
    assert json.loads((meta / "stats_gr00t.json").read_text())["action"]["q99"] == [0.01] * 7
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines()]
    assert [e["length"] for e in episodes] == [4, 6] and episodes[1]["tasks"] == ["put the carrot on the plate"]
    tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [
        {"task_index": 0, "task": "put the spoon on the towel"},
        {"task_index": 1, "task": "put the carrot on the plate"},
    ]
    table = pq.read_table(out / "data/chunk-000/episode_000001.parquet").to_pydict()
    assert table["index"] == list(range(4, 10)) and table["task_index"] == [1] * 6
    assert table["timestamp"] == pytest.approx([t / 5 for t in range(6)])
    assert len(table["action"][0]) == 7 and len(table["observation.state"][0]) == 8
    assert (out / f"videos/chunk-000/{tl.VIDEO_KEY}/episode_000000.mp4").read_bytes() == b"mp4"
    # the labels are up to 0.03 m and 0.2 rad, so most are outside the 1st to 99th percentiles of +-0.01
    assert summary["clip"][0].startswith("action.x: ") and "% above +0.0100" in summary["clip"][0]
    with pytest.raises(SystemExit, match="not empty"):
        tl.write_dataset(str(out), folders, str(stats_file))


def test_main(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    _record(tmp_path / "run", f"{SPOON}/iid15/0", 4, True, np.random.default_rng(4))
    out = tmp_path / "set"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "run"), "--out", str(out), "--stats", "data"])
    tl.main()
    assert "1 episodes, 4 steps, 1 instructions" in capsys.readouterr().out
    q99 = json.loads((out / "meta/stats_gr00t.json").read_text())["action"]["q99"]
    assert len(q99) == 7
