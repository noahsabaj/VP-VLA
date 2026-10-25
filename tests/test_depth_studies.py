"""CPU tests for the two depth studies: the difficulty sweep's levels and the missing-depth scenes (planner/geometry.py),
and their scorers (experiments/score_sweep.py and experiments/score_missing_depth.py)."""

import json
import sys
import zlib
from collections import Counter

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import score_missing_depth as md
from examples.SimplerEnv.eval_files.planner.experiments import score_sweep as sw

EGGPLANT = "PutEggplantInBasketScene-v0"
KEY = "PutSpoonOnTableClothInScene-v0/iid15/3"
TRAIN_LEVELS = [(0.0, "none"), (1.0, "harsh")]


def _geometry():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner import geometry

    return geometry


def _record(episode, success, level, *, plan_ok=True, turn=0.5, sam=0.5, feasible=True):
    """A planner-only record with the VLM plan (feasible: one move; else no first look at all)."""
    r = {
        "mode": "planner",
        "obj_episode_id": episode,
        "success": success,
        "features_version": s.FEATURES_VERSION,
        "features": {"grasp_sam3_score": sam, "place_sam3_score": sam, "grasp_turn_rad": turn, "grasp_reach_m": 0.1},
        "plans": [{"success": plan_ok, "time": 0.1, "ee_path_length": 0.45, "ee_straight_length": 0.3}],
        "depth_noise": level,
        "task_source": "vlm",
        "layout_seed": 11,
        "layout_per_episode": True,
        "vlm_plan": {"feasible": feasible, "steps": [{"object": "spoon", "target": "towel"}], "error": None},
        "vlm_plan_feasible": feasible,
    }
    if not feasible:
        r.update(success=False, features={}, plans=[])
        r["vlm_plan"]["steps"] = []
    return r


def _write_planner(root, task, records):
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _write_video(root, task, episode, success):
    folder = root / task / "ckpt" / "scene"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{'success' if success else 'failure'}_obj_episode_{episode}_moved_True.mp4").write_bytes(b"")


def _draws(episodes, rng):
    """Each episode's latent draws, the same at every level: (plan draw, turn, SAM score)."""
    return {(t, e): (rng.uniform(), 1.5 * rng.uniform(), rng.uniform()) for t in s.TASKS for e in range(episodes)}


def _write_vla(root, name, draws):
    for (t, e), (_, turn, _) in draws.items():
        _write_video(root / f"vla31_{name}", t, e, turn < 0.75)


def _write_sweep(root, name, draws, levels):
    """Planner-only evals with the VLM plan at each level: the planner finds a plan less often at a higher scale, and
    succeeds with a plan and a good SAM score; every 7th episode's VLM plan is not pick-and-place."""
    for scale, level in levels:
        for t in s.TASKS:
            records = []
            for (task, e), (u, turn, sam) in draws.items():
                if task == t:
                    feasible, plan_ok = e % 7 != 3, u > 0.15 + 0.35 * scale
                    records.append(
                        _record(e, plan_ok and sam > 0.3, level, plan_ok=plan_ok, turn=turn, sam=sam, feasible=feasible)
                    )
            _write_planner(root / f"vlmplan_{name}_{level}", t, records)


def test_sweep_levels_scale_harsh():
    g = _geometry()
    assert [lvl for scale, lvl in sw.SWEEP if scale not in (0.0, 1.0)] == [g.sweep_level(x) for x in g.SWEEP_SCALES]
    assert {lvl for _, lvl in sw.SWEEP} == {n for n in g.DepthNoise.LEVELS if n != "realistic"}
    half, double = g.DepthNoise.LEVELS["harsh_x0.5"], g.DepthNoise.LEVELS["harsh_x2"]
    assert half == pytest.approx(dict(k=0.006, drop=0.04, edge_drop=0.45, edge_jump=0.01, pos_mm=5.0, rot_deg=0.5))
    assert double["edge_drop"] == 1.0 and double["drop"] == pytest.approx(0.16) and double["edge_jump"] == 0.01


def test_sweep_levels_draw_the_same_errors_in_the_first_picture():
    g = _geometry()
    depth = np.full((24, 24), 0.6)
    depth[8:16, 8:16] = 0.45  # a box: its edges can drop out
    out = {}
    for scale, level in sw.SWEEP[1:]:
        noise = g.DepthNoise(level, seed=0)
        noise.new_episode(KEY)
        out[scale] = (noise.apply(depth), noise.pose_error)
    kept = np.ones_like(depth, bool)
    for scale in sorted(out):  # the dropped pixels of a smaller scale are dropped at every larger one
        noisy, pose = out[scale]
        assert (noisy[~kept] == 0).all()
        kept &= noisy > 0
    noisy1, pose1 = out[1.0]
    for scale, (noisy, pose) in out.items():
        both = (noisy > 0) & (noisy1 > 0)
        np.testing.assert_allclose(noisy[both] - depth[both], scale * (noisy1[both] - depth[both]), atol=1e-12)
        np.testing.assert_allclose(pose[:3, 3], scale * pose1[:3, 3], atol=1e-12)


def test_missing_depth_draws():
    g = _geometry()
    with pytest.raises(ValueError):
        g.MissingDepth("glass", 0)
    off = g.MissingDepth("none", 0)
    assert not off.new_episode(KEY) and not off.missing
    m = g.MissingDepth("target", 0)
    own = np.random.default_rng([0, zlib.crc32(KEY.encode()), g.MissingDepth.STREAM]).random()
    assert own != np.random.default_rng([0, zlib.crc32(KEY.encode())]).random()  # not the depth noise's stream
    assert m.new_episode(KEY) == (own < 0.5) == m.new_episode(KEY)  # the same episode, the same draw
    draws = [m.new_episode(f"PutCarrotOnPlateInScene-v0/iid15/{e}") for e in range(2000)]
    assert 0.46 < np.mean(draws) < 0.54
    assert not any(m.new_episode(f"{EGGPLANT}/iid15/{e}") for e in range(200))  # its basket is part of the sink


def test_camera_view_takes_out_the_depth_of_the_missing_actors():
    g = _geometry()
    depth, seg = np.full((12, 12), 0.5), np.zeros((12, 12, 2), np.int32)
    seg[4:8, 4:8, 1] = 7
    obs = {
        "image": {g.CAMERA: {"rgb": np.zeros((12, 12, 3), np.uint8), "depth": depth[..., None], "Segmentation": seg}},
        "camera_param": {g.CAMERA: {"intrinsic_cv": [[10, 0, 6], [0, 10, 6], [0, 0, 1]], "extrinsic_cv": np.eye(4)}},
        "agent": {"base_pose": [0, 0, 0, 1, 0, 0, 0]},
    }
    view = g.CameraView(obs, no_depth_ids=[7])
    assert (view.depth[4:8, 4:8] == 0).all() and (view.depth[seg[..., 1] != 7] == 0.5).all()
    assert len(view.points()) == 144 - 16
    views = []
    for ids in ([], [7]):
        noise = g.DepthNoise("harsh", seed=0)
        noise.new_episode(KEY)
        views.append(g.CameraView(obs, noise=noise, no_depth_ids=ids))
    outside = seg[..., 1] != 7
    np.testing.assert_array_equal(views[0].depth[outside], views[1].depth[outside])  # the noise draws the same
    assert (views[1].depth[~outside] == 0).all()


def test_score_sweep(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    rng = np.random.default_rng(3)
    train, test = _draws(30, rng), _draws(20, rng)
    _write_vla(tmp_path, "iid11", train)
    _write_vla(tmp_path, "iid15", test)
    _write_sweep(tmp_path, "iid11", train, TRAIN_LEVELS)
    _write_sweep(tmp_path, "iid15", test, sw.SWEEP[:-1])  # harsh_x2 has not run
    monkeypatch.setattr(sys, "argv", ["score_sweep"])
    sw.main()
    out = capsys.readouterr().out
    assert "note: no eval of iid15 at harsh_x2" in out and "1. not run: it needs every level" in out

    _write_sweep(tmp_path, "iid15", test, sw.SWEEP[-1:])
    sw.main()
    out = capsys.readouterr().out
    assert "80 episodes at each level, 20 episode indices:" in out
    levels = [lvl for _, lvl in sw.SWEEP]
    first = [line[2:].split(" | ")[0] for line in out.splitlines() if line[2:].split(" | ")[0] in levels]
    assert first[:7] == levels  # the first table: one row per level, in order
    assert "1. the ASM vs the better of VLM -> planner and VLA-only at each level: " in out
    assert "What makes the ASM better at each level" in out and "The sweep's ASM against an ASM trained on one" in out

    # the planned test's value of an episode index: its sum over the levels and the 4 tasks
    asm = sw.train("iid11")
    ls = sw.common({lvl: sw.load("vlmplan", "iid15", lvl, sw.VLA_RUNS) for _, lvl in sw.SWEEP})
    keys = ls["none"].keys
    out = {lvl: sw.outcomes(asm, ls[lvl]) for _, lvl in sw.SWEEP}
    d = sw.index_values([out[lvl]["asm"] - out[lvl]["better"] for _, lvl in sw.SWEEP], keys)
    assert d.shape == (20, 7 * 4)
    # one row per episode index, one column per level and task: level i puts i + 1 on the carrot task of index 7
    probe = [np.array([(i + 1.0) * (t == s.TASKS[1] and e == 7) for t, e in keys]) for i in range(7)]
    d = sw.index_values(probe, keys)
    row = sorted({e for _, e in keys}).index(7)
    assert d[row].tolist() == [(j // 4 + 1.0) * (j % 4 == 1) for j in range(28)] and d.sum() == 28


def test_pooled_asm_trains_on_both_levels_as_one(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    train = _draws(30, np.random.default_rng(4))
    _write_vla(tmp_path, "iid11", train)
    _write_sweep(tmp_path, "iid11", train, TRAIN_LEVELS)
    trained, fit = {}, s.fit_two_model

    def spy(records, *args, **kwargs):
        trained["levels"] = Counter(r["depth_noise"] for r in records)
        return fit(records, *args, **kwargs)

    monkeypatch.setattr(s, "fit_two_model", spy)
    asm = sw.train("iid11")
    assert asm.trained_on["depth_noise"] is None and asm.trained_on["pooled"]["depth_noise"] == ["none", "harsh"]
    assert asm.trained_on["task_source"] == "vlm"
    ungated = sum(e % 7 != 3 for _, e in train)  # the episodes of each level that the gate does not give the VLA
    assert trained["levels"] == {"none": ungated, "harsh": ungated}


def _write_missing(root, name, draws, prefix):
    """Planner-only evals with the VLM plan and clean depth. With prefix miss, the place target of about half of the
    episodes gives no depth (none in the eggplant task): the planner then fails after the grasp."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for t in s.TASKS:
        records = []
        for (task, e), (u, turn, sam) in draws.items():
            if task != t:
                continue
            missing = prefix == "miss" and t != EGGPLANT and bool(rng.random() < 0.5)
            feasible, plan_ok = e % 7 != 3, u > 0.1
            success = plan_ok and sam > 0.05 and not missing
            r = _record(e, success, "none", plan_ok=plan_ok, turn=turn, sam=sam, feasible=feasible)
            if feasible:
                r["features"].update(grasp_valid_depth_frac=0.95, place_valid_depth_frac=0.0 if missing else 0.95)
                if not success:
                    r["failure"] = "place target not found" if missing else "grasp missed"
            r.update(missing_depth="target" if prefix == "miss" else "none", target_depth_missing=missing)
            records.append(r)
        _write_planner(root / f"{prefix}_{name}_none", t, records)


def test_score_missing_depth(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    rng = np.random.default_rng(5)
    for name, episodes in (("iid11", 40), ("iid15", 20)):
        draws = _draws(episodes, rng)
        _write_vla(tmp_path, name, draws)
        _write_missing(tmp_path, name, draws, "miss")
        _write_missing(tmp_path, name, draws, "vlmplan")
    monkeypatch.setattr(sys, "argv", ["score_missing_depth"])
    md.main()
    out = capsys.readouterr().out
    assert "-> plan_ok, plan_path_ratio, grasp_valid_depth_frac, place_valid_depth_frac" in out
    assert "80 episodes, 20 episode indices; the place target gave no depth in" in out
    assert "1. vs VLM -> planner (planner-only): " in out and "2. vs VLA-only: " in out
    assert f"- vs {md.FALLBACK} (secondary): " in out and f"- vs {md.CLEAN_ASM} (secondary): " in out
    assert "| target without depth | place target not found |" in out
    assert "The cost of the missing depth to VLM -> planner (secondary): vlmplan_iid15_none on the same 80" in out
    assert "### What makes the ASM better" in out

    # the ASM sends the episodes whose target has no depth to the VLA; the first plan cannot see them
    asm = s.train_two_model(str(tmp_path / "miss_iid11_none"), [str(tmp_path / "vla31_iid11")], md.CANDIDATES)
    test = md.whole_indices(md.load("miss", "iid15", "none", md.VLA_RUNS), "")
    use, gated, _, _ = md.asm_choice(asm, test)
    missing = md.missing(test)
    assert missing.sum() > 10 and use[missing].mean() < 0.25 < use[~missing & ~gated].mean()
    plan_ok = np.array([s.row_features(r, ["plan_ok"])[0] for r in test.records])
    assert plan_ok[missing & ~gated].mean() > 0.8
