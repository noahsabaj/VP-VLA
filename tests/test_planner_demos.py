"""CPU tests for the recorded demonstrations (eval_files/planner/demos.py) and their training set
(train_files/planner_demos_to_lerobot.py)."""

import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import demos
from examples.SimplerEnv.train_files import planner_demos_to_lerobot as tl
from examples.SimplerEnv.train_files.rewordings import FAMILIES, REWORDINGS, rewording

SPOON = "PutSpoonOnTableClothInScene-v0"
CARROT = "PutCarrotOnPlateInScene-v0"
TWO_STEP = "TwoStepSpoonAndCarrot-v0"
SCENE = {"objects": ["bridge_carrot_generated_modified", "bridge_plate_objaverse_larger"], "goals": [[0, 1]]}
# The two-step task's scene: the spoon goes on the towel, then the carrot on the plate.
KITCHEN = {
    "objects": [
        "bridge_spoon_generated_modified",
        "table_cloth_generated_shorter",
        "bridge_carrot_generated_modified",
        "bridge_plate_objaverse_larger",
    ],
    "goals": [[0, 1], [2, 3]],
}


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


class _Actor:
    def __init__(self, actor_id, name, xyz=(0.0, 0.0, 0.0)):
        self.id, self.name, self.pose = actor_id, name, SimpleNamespace(p=np.array(xyz))


def test_the_scene_objects_and_their_map():
    carrot, plate = _Actor(7, SCENE["objects"][0], (-0.1, 0.05, 0.9)), _Actor(9, SCENE["objects"][1])
    u = SimpleNamespace(episode_objs=[carrot, plate], episode_source_obj=carrot, episode_target_obj=plate)
    env = SimpleNamespace(unwrapped=u)
    assert demos.scene_objects(env) == SCENE
    seg = np.zeros((2, 3, 4), np.int32)
    seg[..., 1] = [[7, 7, 9], [3, 9, 0]]  # 3: a robot link
    obs = {"image": {"3rd_view_camera": {"Segmentation": seg}}}
    assert demos.scene_object_map(obs, env).tolist() == [[1, 1, 2], [0, 2, 0]]
    xyz = demos.scene_object_xyz(env)
    assert xyz.dtype == np.float32 and xyz.shape == (2, 3)
    np.testing.assert_allclose(xyz, [[-0.1, 0.05, 0.9], [0.0, 0.0, 0.0]], atol=1e-6)
    u.task = SimpleNamespace(goals=((0, 1, False), (1, 0, True)))  # a harder scene: its task's goals, in order
    assert demos.scene_objects(env)["goals"] == [[0, 1], [1, 0]]


def _object_map(t, hidden=()):
    m = np.zeros((48, 64), np.uint8)
    if t not in hidden:
        m[10:14, 5 + 2 * t : 9 + 2 * t] = 1  # the object to move
    m[30:40, 40:60] = 2  # its place
    return m


def _record(
    root,
    key,
    steps,
    success,
    rng,
    instruction="put the spoon on the towel",
    scene=None,
    maps=None,
    grippers=None,
    xyz=None,
):
    written = {}

    def fake_video(path, frames, fps):
        written["frames"], written["fps"] = len(frames), fps
        with open(path, "wb") as fh:
            fh.write(b"mp4")

    r = demos.DemoRecorder(str(root), write_video=fake_video)
    task, _, episode = key.split("/")
    r.start(key, env_name=task, obj_episode_id=int(episode), layout_seed=15, layout_per_episode=True, **(scene or {}))
    labels = [_label(rng) for _ in range(steps)]
    for t, label in enumerate(labels):
        if grippers is not None:
            label[6] = grippers[t]
        r.add(
            np.full((48, 64, 3), t, np.uint8),
            _obs([0.3, 0.0, 0.1]),
            demos.action_from_label(label),
            instruction,
            object_map=None if maps is None else maps(t),
            object_xyz=None if xyz is None else xyz(t),
        )
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


def test_the_recorder_keeps_the_object_maps_and_positions(tmp_path):
    rng = np.random.default_rng(5)

    def xyz(t):
        return [[0.01 * t, 0.0, 0.9], [0.2, 0.1, 0.88]]

    folder, _, _ = _record(tmp_path, f"{CARROT}/iid15/0", 4, True, rng, scene=SCENE, maps=_object_map, xyz=xyz)
    with np.load(os.path.join(folder, "episode.npz")) as z:
        assert z["object_map"].dtype == np.uint8 and z["object_map"].shape == (4, 48, 64)
        np.testing.assert_array_equal(z["object_map"][3], _object_map(3))
        assert z["object_xyz"].dtype == np.float32 and z["object_xyz"].shape == (4, 2, 3)
        np.testing.assert_allclose(z["object_xyz"][3], xyz(3), atol=1e-6)
    meta = json.loads(open(os.path.join(folder, "meta.json")).read())
    assert (meta["objects"], meta["goals"]) == (SCENE["objects"], [[0, 1]])
    with pytest.raises(ValueError, match="1 object maps for 2 steps"):
        _record(tmp_path, f"{CARROT}/iid15/1", 2, True, rng, maps=lambda t: _object_map(t) if t == 0 else None)
    with pytest.raises(ValueError, match="1 object xyz for 2 steps"):
        _record(tmp_path, f"{CARROT}/iid15/2", 2, True, rng, xyz=lambda t: xyz(t) if t == 1 else None)


def _grippers(*opens):
    return np.array([[0, 0, 0, 0, 0, 0, g] for g in opens], float)


def _entries(instruction, actions, names=("carrot", "plate")):
    parts = tl.instruction_parts(instruction)
    return tl.vlm_entries(parts, actions, [names] * len(parts))


def test_the_subtasks_follow_the_gripper():
    one = _entries("put carrot on plate", _grippers(1, 1, 0, 0, 1, 1))
    assert [(e["frame"], e["decision"], e["gripper_state_changed"]) for e in one] == [
        (0, "continue", False),
        (2, "continue", True),
        (4, "continue", True),
    ]
    assert {(e["subtask"], e["target_object"], e["target_location"]) for e in one} == {
        ("put carrot on plate", "carrot", "plate")
    }
    # a pick ends at the grasp, and only the place part names the place
    two = _entries("Pick up the carrot and put it on the plate", _grippers(1, 1, 0, 0, 1))
    assert [(e["frame"], e["subtask_idx"], e["subtask"], e["decision"], e["target_location"]) for e in two] == [
        (0, 0, "pick up the carrot", "continue", None),
        (2, 1, "put it on the plate", "proceed", "plate"),
        (4, 1, "put it on the plate", "continue", "plate"),
    ]
    assert tl.object_word("bridge_plate_objaverse_larger_2") == "plate"  # a scene's second plate
    assert tl.object_word("baked_green_cube_3cm") == "green block" and tl.object_word("new_model") == "new_model"


def test_every_training_rewording_names_the_place():
    for task, families in REWORDINGS.items():
        for family, options in families.items():
            for option in options:
                entries = _entries(option, _grippers(1, 1, 0, 0, 1, 1), ("object", "place"))
                assert entries[-1]["target_location"] == "place", (task, family, option)


def test_the_two_step_rewordings_have_one_part_per_goal_in_order():
    for option in [o for options in REWORDINGS[TWO_STEP].values() for o in options]:
        parts = tl.instruction_parts(option)
        assert len(parts) == 2 and all(map(tl.is_placing, parts)), option
        carrot = [re.search(r"\b(carrot|root)\b", p) is not None for p in parts]
        assert carrot == [False, True], option  # the spoon's goal first, as the task's goals


def _kitchen_map(t):
    m = np.zeros((48, 64), np.uint8)
    for k, (row, col) in enumerate([(2, 2), (2, 20), (30, 2), (30, 20)]):  # spoon, towel, carrot, plate
        m[row : row + 4, col : col + 4] = k + 1
    return m


def _moves(steps, *carried):
    """object_xyz for the kitchen's 4 objects: object k moves 0.2 m in x during steps [a, b) for each (k, a, b)."""

    def xyz(t):
        out = np.zeros((4, 3))
        out[:, 2] = 0.9
        for k, a, b in carried:
            out[k, 0] = 0.2 * min(max(t - a + 1, 0), b - a) / (b - a)
        return out

    assert all(b <= steps for _, _, b in carried)
    return xyz


def _two_step_prompts(tmp_path, grippers, xyz, instruction="put the spoon on the towel and the carrot on the plate"):
    runs, vp = tmp_path / "run", tmp_path / "vp"
    steps = len(grippers)
    _record(
        runs,
        f"{TWO_STEP}/iid100/0",
        steps,
        True,
        np.random.default_rng(10),
        instruction,
        KITCHEN,
        _kitchen_map,
        grippers,
        xyz,
    )
    tl.write_dataset(str(tmp_path / "data" / "harder_demos"), tl.find_episodes([str(runs)]), "data", str(vp))
    with np.load(vp / "harder_demos" / "chunk-000" / "episode_000000.npz") as z:
        return json.loads(str(z["vlm_results"])), {k: z[k] for k in ("target_masks", "location_masks")}


def _marked(masks):
    """The object number that each step's mask marks (0: none)."""
    return [int(np.max(m * _kitchen_map(0))) for m in masks]


def test_the_visual_prompts_of_two_goals(tmp_path):
    pytest.importorskip("pyarrow")
    # grasp the spoon at 2, put it on the towel at 5; grasp the carrot at 7, put it on the plate at 10
    grippers = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    entries, masks = _two_step_prompts(tmp_path / "a", grippers, _moves(12, (0, 2, 5), (2, 7, 10)))
    assert [(e["frame"], e["subtask_idx"], e["decision"]) for e in entries] == [
        (0, 0, "continue"),
        (2, 0, "continue"),
        (5, 1, "proceed"),
        (7, 1, "continue"),
        (10, 1, "continue"),
    ]
    assert [(e["subtask"], e["target_object"], e["target_location"]) for e in entries[1:3]] == [
        ("put the spoon on the towel", "spoon", "towel"),
        ("the carrot on the plate", "carrot", "plate"),
    ]
    assert _marked(masks["target_masks"]) == [1] * 5 + [3] * 7  # the spoon until it is on the towel, then the carrot
    assert _marked(masks["location_masks"]) == [2] * 5 + [4] * 7
    # done the other way round: the carrot's part and prompts first
    entries, masks = _two_step_prompts(tmp_path / "b", grippers, _moves(12, (2, 2, 5), (0, 7, 10)))
    assert [e["subtask"] for e in entries] == ["the carrot on the plate"] * 2 + ["put the spoon on the towel"] * 3
    assert _marked(masks["target_masks"]) == [3] * 5 + [1] * 7 and _marked(masks["location_masks"]) == [4] * 5 + [2] * 7


def test_a_failed_grasp_does_not_end_its_goal(tmp_path):
    pytest.importorskip("pyarrow")
    # the spoon slips at 4 (grasped at 2), is grasped again at 8 and put on the towel at 15; the carrot at 18 and 25
    grippers = [1, 1, 0, 0, 1, 1, 1, 1] + [0] * 7 + [1, 1, 1] + [0] * 7 + [1] * 5
    entries, masks = _two_step_prompts(tmp_path, grippers, _moves(30, (0, 9, 15), (2, 19, 25)))
    assert [(e["frame"], e["subtask_idx"]) for e in entries] == [
        (0, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (15, 1),
        (18, 1),
        (25, 1),
    ]
    assert _marked(masks["target_masks"]) == [1] * 15 + [3] * 15


def test_the_visual_prompts_need_object_maps_and_a_part_per_goal(tmp_path):
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(7)
    _record(tmp_path / "a", f"{CARROT}/iid15/0", 3, True, rng, scene=SCENE)
    with pytest.raises(SystemExit, match="without object maps"):
        tl.write_dataset(str(tmp_path / "set_a"), tl.find_episodes([str(tmp_path / "a")]), "data", str(tmp_path / "v"))
    moves = _moves(4, (0, 0, 1), (2, 2, 3))
    _record(tmp_path / "b", f"{TWO_STEP}/iid100/0", 4, True, rng, scene=KITCHEN, maps=_kitchen_map, xyz=moves)
    with pytest.raises(SystemExit, match='2 goals; "put the spoon on the towel" needs one part for each'):
        tl.write_dataset(str(tmp_path / "set_b"), tl.find_episodes([str(tmp_path / "b")]), "data", str(tmp_path / "w"))
    two_parts = "put the spoon on the towel and the carrot on the plate"
    _record(tmp_path / "c", f"{TWO_STEP}/iid100/0", 4, True, rng, two_parts, KITCHEN, _kitchen_map)
    with pytest.raises(SystemExit, match="2 goals, recorded without the objects' positions"):
        tl.write_dataset(str(tmp_path / "set_c"), tl.find_episodes([str(tmp_path / "c")]), "data", str(tmp_path / "x"))


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


def test_the_visual_prompts(tmp_path):
    pytest.importorskip("pyarrow")
    runs = tmp_path / "run"

    def maps(t):  # the object is out of sight at step 3
        return _object_map(t, hidden=(3,))

    _record(
        runs,
        f"{CARROT}/iid15/0",
        6,
        True,
        np.random.default_rng(6),
        instruction="put carrot on plate",
        scene=SCENE,
        maps=maps,
        grippers=[1, 1, 0, 0, 0, 1],
    )
    vp = tmp_path / "vp"
    summary = tl.write_dataset(str(tmp_path / "data" / "planner_demos"), tl.find_episodes([str(runs)]), "data", str(vp))
    assert summary["visual_prompts"] == [f"visual prompts, {CARROT}: crosshair 83.3% of 6 steps, box 100.0%"]
    with np.load(vp / "planner_demos" / "chunk-000" / "episode_000000.npz") as z:  # read as the training reads it
        assert (int(z["episode_index"]), int(z["num_frames"]), float(z["fps"])) == (0, 6, 5.0)
        assert str(z["task_description"]) == "put carrot on plate"
        entries = json.loads(str(z["vlm_results"]))
        assert [e["frame"] for e in entries] == [0, 2, 5]
        assert {(e["target_object"], e["target_location"]) for e in entries} == {("carrot", "plate")}
        assert z["target_masks"].shape == (6, 48, 64) and z["target_masks"].dtype == np.uint8
        assert z["target_boxes"][1].tolist() == [7, 10, 11, 14]  # x1, y1, x2, y2 around the object's pixels
        assert z["target_scores"].tolist() == [1, 1, 1, 0, 1, 1]  # no crosshair while the object is out of sight
        assert z["location_boxes"][3].tolist() == [40, 30, 60, 40] and z["location_scores"].all()
        np.testing.assert_array_equal(z["location_masks"][0], _object_map(0) == 2)
    with pytest.raises(SystemExit, match="not empty"):
        tl.write_dataset(str(tmp_path / "other" / "planner_demos"), tl.find_episodes([str(runs)]), "data", str(vp))


def test_the_rewordings(tmp_path):
    pytest.importorskip("pyarrow")
    runs = tmp_path / "run"
    key = f"{CARROT}/iid15/0"
    _record(
        runs,
        key,
        4,
        True,
        np.random.default_rng(8),
        instruction="put carrot on plate",
        scene=SCENE,
        maps=_object_map,
        grippers=[1, 0, 0, 1],
    )
    out, vp = tmp_path / "data" / "planner_demos", tmp_path / "vp"
    summary = tl.write_dataset(str(out), tl.find_episodes([str(runs)]), "data", str(vp), reword=True)
    assert (summary["recorded"], summary["episodes"], summary["frames"]) == (1, 5, 20)
    expected = ["put carrot on plate"] + [rewording(CARROT, f, key) for f in FAMILIES]
    tasks = [json.loads(line)["task"] for line in (out / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == expected
    sources = [json.loads(line) for line in (out / "meta" / "sources.jsonl").read_text().splitlines()]
    assert [s["rewording"] for s in sources] == [None, *FAMILIES] and len({s["source"] for s in sources}) == 1
    for episode, instruction in enumerate(expected):
        with np.load(vp / "planner_demos" / "chunk-000" / f"episode_{episode:06d}.npz") as z:
            entries = json.loads(str(z["vlm_results"]))
            assert str(z["task_description"]) == instruction
            assert entries[-1]["target_object"] == "carrot" and entries[-1]["target_location"] == "plate"
    # a task without training rewordings: a new test scene (T5)
    _record(tmp_path / "novel", "NovelSpongeOnPlate-v0/iid15/0", 2, True, np.random.default_rng(9))
    with pytest.raises(SystemExit, match="no training rewordings for NovelSpongeOnPlate-v0"):
        tl.write_dataset(str(tmp_path / "set_n"), tl.find_episodes([str(tmp_path / "novel")]), "data", reword=True)


def test_main(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    _record(tmp_path / "run", f"{SPOON}/iid15/0", 4, True, np.random.default_rng(4))
    out = tmp_path / "set"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "run"), "--out", str(out), "--stats", "data"])
    tl.main()
    assert "1 episodes of 1 recorded, 4 steps, 1 instructions" in capsys.readouterr().out
    q99 = json.loads((out / "meta/stats_gr00t.json").read_text())["action"]["q99"]
    assert len(q99) == 7
