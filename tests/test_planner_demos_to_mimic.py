"""CPU tests for the demonstrations as mimic-video's training episodes (train_files/planner_demos_to_mimic.py) and
for the fingers' positions that the recorder keeps for them (eval_files/planner/demos.py)."""

import json
import os

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import demos
from examples.SimplerEnv.train_files import planner_demos_to_mimic as tm

SPOON = "PutSpoonOnTableClothInScene-v0"
CARROT = "PutCarrotOnPlateInScene-v0"
STEP_NS = 200_000_000  # 5 steps a second
STEP_KEYS = ("workspace_rgb", "eef_pose_lowdim", "gripper_state_lowdim", "gripper_action_lowdim")


def _obs(xyz, yaw=0.0, fingers=None):
    half = yaw / 2
    agent = {"base_pose": np.array([1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0])}
    if fingers is not None:
        agent["qpos"] = np.r_[np.zeros(6), fingers]
    tcp = np.array([1.0 + xyz[0], 2.0 + xyz[1], 0.5 + xyz[2], np.cos(half), 0, 0, np.sin(half)])
    return {"agent": agent, "extra": {"tcp_pose": tcp}}


def _record(root, key, grippers, success=True, fingers=None, instruction="put the spoon on the towel", first_step=0):
    """A demonstration of len(grippers) steps: the hand goes 1 cm in x and turns 0.1 rad at each step."""
    r = demos.DemoRecorder(str(root), write_video=lambda path, frames, fps: open(path, "wb").close())
    r.start(key, env_name=key.split("/")[0])
    for t, g in enumerate(grippers):
        r.add(
            np.full((tm.HEIGHT, tm.WIDTH, 3), t, np.uint8),
            _obs([0.3 + 0.01 * t, 0.0, 0.1], yaw=0.1 * t, fingers=None if fingers is None else fingers[t]),
            demos.action_from_label([0.01, 0, 0, 0, 0, 0.1, g]),
            instruction,
        )
    return r.finish(success, first_step=first_step)


def _frames(folder):
    """What read_frames gives for a recorded folder (the test's recorder writes an empty frames.mp4)."""
    with np.load(os.path.join(os.path.dirname(folder), "episode.npz")) as z:
        steps = len(z["actions"])
    return np.stack([np.full((tm.HEIGHT, tm.WIDTH, 3), t, np.uint8) for t in range(steps)])


def _encoder(calls):
    def make():
        calls.append("made")

        def encode(text):
            calls.append(text)
            return np.full((1, 512, 1024), len(text), np.float32)

        return encode

    return make


def test_the_recorder_keeps_the_fingers_positions(tmp_path):
    fingers = [[0.037, 0.037], [0.030, 0.032], [0.015, 0.017]]
    folder = _record(tmp_path, f"{SPOON}/iid15/0", [1, 0, 0], fingers=fingers)
    with np.load(os.path.join(folder, "episode.npz")) as z:
        assert z["fingers"].dtype == np.float32 and z["fingers"].shape == (3, 2)
        np.testing.assert_allclose(z["fingers"], fingers, atol=1e-7)
    # the planner's part of an episode: the fingers of its steps only
    folder = _record(tmp_path, f"{SPOON}/iid15/1", [1, 0, 0], fingers=fingers, first_step=1)
    with np.load(os.path.join(folder, "episode.npz")) as z:
        np.testing.assert_allclose(z["fingers"], fingers[1:], atol=1e-7)
    # an observation without the joints' positions: no fingers in the file
    folder = _record(tmp_path, f"{SPOON}/iid15/2", [1, 0, 0])
    with np.load(os.path.join(folder, "episode.npz")) as z:
        assert "fingers" not in z.files


def test_the_poses_are_the_hand_in_the_base_frame():
    states = np.array([[0.3, -0.1, 0.05, 0.2, -0.3, 0.4, 0.0, 1.0], [0.31, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    poses = tm.pose_matrices(states)
    assert poses.shape == (2, 4, 4) and poses.dtype == np.float64
    np.testing.assert_allclose(poses[0, :3, 3], [0.3, -0.1, 0.05])
    np.testing.assert_allclose(poses[0, :3, :3], demos.euler_to_matrix([0.2, -0.3, 0.4]))
    np.testing.assert_allclose(poses[1], [[1, 0, 0, 0.31], [0, 1, 0, 0.0], [0, 0, 1, 0.1], [0, 0, 0, 1]])
    np.testing.assert_array_equal(poses[:, 3], [[0, 0, 0, 1]] * 2)


def test_the_gripper_state_is_measured_when_the_demonstration_has_the_fingers():
    states = np.zeros((4, 8))
    states[:, 7] = [1, 1, 0, 0]
    opening, source = tm.gripper_state({"states": states})
    assert source == "command" and opening.tolist() == [1, 1, 0, 0]
    # mimic-video's eval loop: the fingers' mean position, 0.015 m closed to 0.037 m open, as 0 to 1
    fingers = np.array([[0.037, 0.037], [0.030, 0.022], [0.015, 0.015], [0.010, 0.040]])
    opening, source = tm.gripper_state({"states": states, "fingers": fingers})
    assert source == "measured"
    np.testing.assert_allclose(opening, [1.0, 0.5, 0.0, (0.025 - 0.015) / 0.022], atol=1e-12)
    assert tm.gripper_state({"states": states, "fingers": np.full((4, 2), 0.05)})[0].tolist() == [1.0] * 4
    assert tm.gripper_state({"states": states, "fingers": np.zeros((4, 2))})[0].tolist() == [0.0] * 4


def test_an_episode_has_the_keys_of_mimic_videos_bridge_data(tmp_path):
    folder = _record(tmp_path, f"{SPOON}/iid15/0", [1, 0, 0, 1], fingers=[[0.037] * 2, [0.036] * 2, [0.02] * 2] * 2)
    with np.load(os.path.join(folder, "episode.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    frames = _frames(os.path.join(folder, "frames.mp4"))
    data, source = tm.episode_tensors(frames, arrays, "put the spoon on the towel", np.ones((1, 512, 1024)))
    assert source == "measured"
    assert sorted(data) == sorted(
        f"{name}{end}"
        for name in (*STEP_KEYS, "language_instruction", "language_embedding")
        for end in ("", "_timestamps")
    )
    assert data["workspace_rgb"].dtype == np.uint8 and data["workspace_rgb"].shape == (4, 480, 640, 3)
    assert data["eef_pose_lowdim"].shape == (4, 4, 4)
    np.testing.assert_allclose(data["eef_pose_lowdim"][2, :3, 3], [0.32, 0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(data["eef_pose_lowdim"][2, :3, :3], demos.euler_to_matrix([0, 0, 0.2]), atol=1e-6)
    for name in STEP_KEYS:
        times = data[f"{name}_timestamps"]
        assert times.dtype == np.float64 and times.tolist() == [0, STEP_NS, 2 * STEP_NS, 3 * STEP_NS]
    # the gripper action is the command in force (open at the start), so one step later it is the step's own command
    assert data["gripper_action_lowdim"].tolist() == [1, 1, 0, 0]
    np.testing.assert_array_equal(data["gripper_action_lowdim"][1:], arrays["actions"][:-1, 6])
    np.testing.assert_allclose(data["gripper_state_lowdim"], [1, 21 / 22, 5 / 22, 1], atol=1e-6)
    assert bytes(data["language_instruction"][0]).decode() == "put the spoon on the towel"
    assert data["language_instruction"].shape == (1, 26) and data["language_instruction"].dtype == np.uint8
    assert data["language_embedding"].dtype == np.float32 and data["language_embedding"].shape == (1, 512, 1024)
    for name in ("language_instruction", "language_embedding"):
        assert data[f"{name}_timestamps"].dtype == np.uint64 and data[f"{name}_timestamps"].tolist() == [0]
    # without an embedding (mimic-video's precompute_t5.py can add it later) the episode has no embedding keys
    assert "language_embedding" not in tm.episode_tensors(frames, arrays, "put the spoon on the towel")[0]


def test_an_episode_that_does_not_fit_stops_the_run():
    arrays = {"actions": np.ones((3, 7)), "states": np.ones((3, 8))}
    frames = np.zeros((3, 480, 640, 3), np.uint8)
    tm.episode_tensors(frames, arrays, "put the spoon on the towel")
    with pytest.raises(ValueError, match="the pictures are"):
        tm.episode_tensors(frames[:2], arrays, "put the spoon on the towel")
    with pytest.raises(ValueError, match="the pictures are"):
        tm.episode_tensors(np.zeros((3, 48, 64, 3), np.uint8), arrays, "put the spoon on the towel")
    with pytest.raises(ValueError, match="2 states for 3 actions"):
        tm.episode_tensors(frames, {**arrays, "states": np.ones((2, 8))}, "put the spoon on the towel")
    with pytest.raises(ValueError, match="no instruction"):
        tm.episode_tensors(frames, arrays, " ")
    with pytest.raises(ValueError, match="not 1 x 512 x 1024"):
        tm.episode_tensors(frames, arrays, "put the spoon on the towel", np.ones((512, 1024)))
    closed = np.ones((3, 8))
    closed[1, 7] = 0  # the state says closed after an open command
    with pytest.raises(ValueError, match="one step later"):
        tm.episode_tensors(frames, {**arrays, "states": closed}, "put the spoon on the towel")


def test_only_whole_successful_episodes_with_one_instruction_are_taken():
    meta = {"success": True, "first_step": 0, "steps": 40, "episode_steps": 40, "episode_key": "a/b/0"}
    assert tm.usable(meta) is None
    assert tm.usable({**meta, "success": False}) == "not successful"
    assert tm.usable({**meta, "first_step": 12, "steps": 28}) == "not a whole episode"
    assert tm.usable({**meta, "first_step": None, "steps": 0}) == "not a whole episode"
    assert tm.usable({**meta, "steps": 2, "episode_steps": 2}) == "fewer than 3 steps"
    assert tm.episode_instruction({**meta, "instructions": ["put it there"] * 40}) == "put it there"
    with pytest.raises(ValueError, match="a/b/0: 2 instructions"):
        tm.episode_instruction({**meta, "instructions": ["put it there"] * 39 + ["now the other one"]})


def test_the_embeddings_file_keeps_each_instruction_once(tmp_path):
    path = str(tmp_path / "t5.npz")
    assert tm.load_embeddings(path) == {}
    one, two = np.full((1, 512, 1024), 1.5), np.full((1, 512, 1024), -2.0)
    tm.save_embeddings(path, {"put the spoon on the towel": one, "put carrot on plate": two})
    loaded = tm.load_embeddings(path)
    assert sorted(loaded) == ["put carrot on plate", "put the spoon on the towel"]
    assert loaded["put carrot on plate"].dtype == np.float32
    np.testing.assert_array_equal(loaded["put the spoon on the towel"], one)
    np.testing.assert_array_equal(loaded["put carrot on plate"], two)


def test_the_demonstrations_become_episode_files(tmp_path):
    root, out, saved, calls = tmp_path / "demos", str(tmp_path / "out"), [], []

    def save(data, path):
        saved.append(path)

    _record(root, f"{SPOON}/iid15/0", [1, 0, 0, 1], fingers=[[0.03, 0.03]] * 4)
    _record(root, f"{SPOON}/iid15/1", [1, 1, 0, 0])
    _record(root, f"{SPOON}/iid15/2", [1, 0, 0, 1], success=False)
    _record(root, f"{CARROT}/iid15/0", [1, 0, 0, 0], instruction="put carrot on plate")
    _record(root, f"{CARROT}/iid15/1", [1, 0, 0, 0], instruction="put carrot on plate", first_step=2)
    folders = tm.find_episodes([str(root), str(root / SPOON)])
    assert len(folders) == 5 and folders == sorted(folders)
    embeddings = str(tmp_path / "t5.npz")
    report = tm.convert(folders, out, embeddings, make_encoder=_encoder(calls), frames_of=_frames, save=save)
    assert report["written"] == [
        f"{CARROT}__iid15__0.safetensors",
        f"{SPOON}__iid15__0.safetensors",
        f"{SPOON}__iid15__1.safetensors",
    ]
    assert report["left_out"] == {"not successful": 1, "not a whole episode": 1}
    assert report["gripper_state"] == {"command": 2, "measured": 1}
    assert report["instructions"] == {"put carrot on plate": 1, "put the spoon on the towel": 2}
    paths = sorted(os.path.basename(path) for path in saved)
    assert paths == report["written"] and all(os.path.dirname(path) == out for path in saved)
    # one encoder for the run and one encoding for each different instruction
    assert calls == ["made", "put carrot on plate", "put the spoon on the towel"]
    assert sorted(tm.load_embeddings(embeddings)) == ["put carrot on plate", "put the spoon on the towel"]
    # a second run has every embedding: it does not load the encoder; --limit takes the first usable episodes
    again = tm.convert(folders, out, embeddings, 2, make_encoder=_encoder(calls), frames_of=_frames, save=save)
    assert len(calls) == 3 and again["written"] == report["written"][:2]
    assert again["left_out"] == report["left_out"]


def test_the_saved_episode_has_its_instructions_embedding(tmp_path):
    root, saved = tmp_path / "demos", []
    _record(root, f"{CARROT}/iid15/0", [1, 0, 0, 0], instruction="put carrot on plate")
    tm.convert(
        tm.find_episodes([str(root)]),
        str(tmp_path / "out"),
        str(tmp_path / "t5.npz"),
        make_encoder=_encoder([]),
        frames_of=_frames,
        save=lambda data, path: saved.append(data),
    )
    assert saved[0]["language_embedding"].shape == (1, 512, 1024)
    assert float(saved[0]["language_embedding"][0, 0, 0]) == len("put carrot on plate")
    assert saved[0]["workspace_rgb"][3, 0, 0].tolist() == [3, 3, 3]


def test_the_embeddings_file_must_be_outside_the_episodes_folder(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="--embeddings must not be in --out"):
        tm.main([str(tmp_path), "--out", str(out), "--embeddings", str(out / "t5.npz")])


def test_the_files_are_written_with_safetensors(tmp_path):
    st = pytest.importorskip("safetensors.numpy")
    root, out = tmp_path / "demos", tmp_path / "out"
    _record(root, f"{SPOON}/iid15/0", [1, 0, 0, 1], fingers=[[0.03, 0.03]] * 4)
    tm.convert(
        tm.find_episodes([str(root)]), str(out), str(tmp_path / "t5.npz"), make_encoder=_encoder([]), frames_of=_frames
    )
    data = st.load_file(str(out / f"{SPOON}__iid15__0.safetensors"))
    assert data["eef_pose_lowdim"].shape == (4, 4, 4) and data["language_embedding_timestamps"].dtype == np.uint64
    assert json.loads(json.dumps(data["gripper_action_lowdim"].tolist())) == [1, 1, 0, 0]
