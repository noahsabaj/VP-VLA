"""CPU tests for the marriage study's training sets (train_files/select_demos.py), on small pools with the study's
rules."""

import json
import os
import sys

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import demos
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.train_files import select_demos as sd

SPOON, CARROT, HARD = "PutSpoonOnTableClothInScene-v0", "PutCarrotOnPlateInScene-v0", "ClutteredSpoonOnTowel-v0"
JOB = "{task}:{start}:{end}"
# The study's pools at a small size: the planner's first run only (6 episodes of each task); the harder scenes' pool
# grows 2 at a time up to 8; VLA-0's from 4, 2 at a time (one chunk per episode), its harder scene's up to 6; the
# corrections from 2, 2 at a time.
POOLS = {
    "planner": sd.Pool("demos_planner_iid100", "the planner", (SPOON, CARROT), 100, 6, 0, 6, JOB),
    "planner_harder": sd.Pool("demos_vp_harder_iid100", "the planner, marks", (HARD,), 100, 4, 2, 8, JOB),
    "vla": sd.Pool("demos_vla_iid100", "VLA-0", (SPOON, CARROT), 100, 4, 2, None, JOB, chunk=1),
    "vla_harder": sd.Pool("demos_vla_harder_iid100", "VLA-0", (HARD,), 100, 4, 2, 6, JOB, chunk=1),
    "corrections": sd.Pool(
        "demos_corrections_iid101",
        "VLA-PLT, then the planner",
        (SPOON, CARROT),
        101,
        2,
        2,
        None,
        "none:" + JOB,
        corrections=True,
    ),
}
CURVE = {"planner_demos_450": 1, "planner_demos_1125": 2}
N, TOP = 3, 2


def _episode(runs, pool, task, episode, success, chunk="", layout_seed=None, **meta):
    """A recorded episode as the selection reads it (meta.json and episode.npz), in its pool's run folder (the runners
    write it into TASK[/CHUNK]/demos/TASK/LAYOUTS/EPISODE; shorter here, for Windows' path limit)."""
    seed = POOLS[pool].layout_seed if layout_seed is None else layout_seed
    folder = os.path.join(runs, POOLS[pool].run, task, chunk, str(episode))
    os.makedirs(folder)
    np.savez(os.path.join(folder, "episode.npz"), actions=np.zeros((1, 7), np.float32))
    meta = {
        "env_name": task,
        "obj_episode_id": episode,
        "layout_seed": seed,
        "layout_per_episode": True,
        "success": success,
        "steps": 1,
        **meta,
    }
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return os.path.normpath(folder)


def _pool(runs, pool, task, outcomes, start=0, **meta):
    """Episodes start, start + 1, ... of a task, each a success or not."""
    return [_episode(runs, pool, task, start + i, bool(ok), **meta) for i, ok in enumerate(outcomes)]


def _corrections(runs, task, takeovers, start=0):
    """Corrections: (takeover, success) for each episode; the episodes without a takeover have no steps."""
    return [
        _episode(runs, "corrections", task, start + i, ok, takeover=takeover, steps=int(takeover in sd.TOOK_OVER))
        for i, (takeover, ok) in enumerate(takeovers)
    ]


def _select(runs, asm=True):
    return sd.select(str(runs), str(runs / "energy_model" / "asm_none.json") if asm else None, POOLS, N, TOP, CURVE)


def _line(report, task_or_run, start=""):
    """The report's first line that starts with `start` and names `task_or_run`."""
    return next(line for line in report if task_or_run in line and line.strip().startswith(start))


def test_episode_lists():
    assert sd.episode_list([3, 7, 8, 9, 12]) == "3, 7-9, 12" and sd.episode_list([]) == ""


def test_a_pool_has_each_episode_once_and_its_own_layouts(tmp_path):
    assert sd.read_pool(str(tmp_path), POOLS["vla"]) is None  # not run yet
    _pool(tmp_path, "vla", SPOON, [1, 0])
    _episode(tmp_path, "vla", SPOON, 4, True, chunk="eps_4_5")
    with pytest.raises(SystemExit, match=f"no recording of {SPOON} episode 2-3"):
        sd.read_pool(str(tmp_path), POOLS["vla"])
    _pool(tmp_path, "vla", SPOON, [1, 1], start=2, chunk="eps_2_4")
    assert sd.read_pool(str(tmp_path), POOLS["vla"])[SPOON].recorded == 5
    _episode(tmp_path, "vla", SPOON, 4, True, chunk="eps_4_6")
    with pytest.raises(SystemExit, match=f"{SPOON} episode 4 is also in"):
        sd.read_pool(str(tmp_path), POOLS["vla"])
    _episode(tmp_path, "vla_harder", HARD, 0, True, layout_seed=15)
    with pytest.raises(SystemExit, match=r"layout seed 15, .* has iid100"):
        sd.read_pool(str(tmp_path), POOLS["vla_harder"])
    _episode(tmp_path, "planner_harder", CARROT, 0, True)
    with pytest.raises(SystemExit, match=f"{CARROT} is not one of the tasks of demos_vp_harder_iid100"):
        sd.read_pool(str(tmp_path), POOLS["planner_harder"])


def test_the_usable_episodes(tmp_path):
    # the pool's episodes from its last one on are not used
    folders = _pool(tmp_path, "planner_harder", HARD, [1, 0, 0, 0, 0, 0, 1, 0, 1, 1])
    episodes = sd.read_pool(str(tmp_path), POOLS["planner_harder"])[HARD]
    assert episodes == sd.Episodes(10, [(0, folders[0]), (6, folders[6])])
    report = sd.pool_report(POOLS["planner_harder"], {HARD: episodes}, {HARD: 3})
    assert report[1] == (
        f"  {HARD}: 10 episodes (episodes from 8 on are not used), 2 successes (25.0%); needs 3: done, short: keeps 2"
    )
    # a correction: the planner took over, holding the object or to grasp it, and the episode succeeded
    folders = _corrections(
        tmp_path, SPOON, [("holding", True), ("skipped", True), ("grasp", True), ("grasp", False), (None, True)]
    )
    episodes = sd.read_pool(str(tmp_path), POOLS["corrections"])[SPOON]
    assert episodes.usable == [(0, folders[0]), (2, folders[2])]


def test_the_next_chunks():
    vla, vla_harder, planner_harder = POOLS["vla"], POOLS["vla_harder"], POOLS["planner_harder"]

    def chunks(pool, recorded, usable, need):
        episodes = sd.Episodes(recorded, [(e, "") for e in range(usable)])
        return sd.is_done(pool, episodes, need), sd.next_chunks(pool, "T", episodes, need)

    assert chunks(vla, 3, 3, 3) == (False, [])  # its first run is not complete
    assert chunks(vla, 4, 2, 3) == (False, ["T:4:5", "T:5:6"])
    assert chunks(vla, 4, 3, 3) == (True, [])
    assert chunks(vla_harder, 5, 1, 3) == (False, ["T:5:6"])  # up to its last episode
    assert chunks(vla_harder, 6, 1, 3) == (True, [])  # done, short
    assert chunks(planner_harder, 4, 1, 3) == (False, ["T:4:6"])  # one chunk for each task
    assert chunks(POOLS["planner"], 6, 0, 3) == (True, [])  # the planner's pool has its first run only
    assert chunks(POOLS["corrections"], 2, 1, 3) == (False, ["none:T:2:4"])


def _asm(runs):
    """VLA-PLT's ASM: P(planner succeeds) falls with the plan's detour, P(VLA succeeds) is 1/2 everywhere."""

    def model(feature, weight):
        m = s.SuccessModel([feature], tasks=[SPOON, CARROT])
        m.model = s.LogisticRegression.from_json(
            {"mean": [0.0] * 3, "std": [1.0] * 3, "w": [weight, 0.0, 0.0], "b": 0.0, "threshold": 0.5}
        )
        return m

    asm = s.TwoModelASM(model("plan_path_ratio", -1.0), model("grasp_reach_m", 0.0), {"depth_noise": "none"})
    os.makedirs(runs / "energy_model")
    with open(runs / "energy_model" / "asm_none.json", "w") as fh:
        json.dump(asm.to_json(), fh)


def _planner_records(runs, task, outcomes, detours):
    """The planner's record of each episode of its pool (planner_stats.jsonl): the outcome and the first look."""
    with open(runs / "demos_planner_iid100" / task / "planner_stats.jsonl", "w") as fh:
        for episode, (ok, detour) in enumerate(zip(outcomes, detours, strict=True)):
            plan = {"success": True, "ee_path_length": detour, "ee_straight_length": 1.0, "time": 0.1}
            fh.write(
                json.dumps(
                    {
                        "mode": "planner",
                        "obj_episode_id": episode,
                        "success": bool(ok),
                        "features_version": 2,
                        "features": {"grasp_reach_m": 0.3},
                        "plans": [plan],
                        "depth_noise": "none",
                    }
                )
                + "\n"
            )


def test_the_sets(tmp_path):
    runs = tmp_path / "runs"
    _asm(runs)
    # the planner's pools: the carrot has 2 successes in its 6 episodes, and the harder scene needs a second chunk
    spoon = _pool(runs, "planner", SPOON, [1, 0, 1, 1, 1, 0])
    carrot = _pool(runs, "planner", CARROT, [0, 1, 1, 0, 0, 0])
    _planner_records(runs, SPOON, [1, 0, 1, 1, 1, 0], [1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    _planner_records(runs, CARROT, [0, 1, 1, 0, 0, 0], [1.0] * 6)
    hard = _pool(runs, "planner_harder", HARD, [1, 0, 0, 0])
    sel = _select(runs)
    assert _line(sel.report, CARROT) == f"  {CARROT}: 6 episodes, 2 successes (33.3%); needs 3: done, short: keeps 2"
    assert _line(sel.report, HARD).endswith(f"needs 3: needs more: {HARD}:4:6")
    assert _line(sel.report, "demos_vla_iid100").endswith(": not run yet")
    assert sel.sets == dict.fromkeys(sd.SETS) and sel.waiting["planner_demos"] == "waiting for demos_vp_harder_iid100"
    assert sel.waiting["vla_successes_reworded"] == (
        "waiting for demos_vp_harder_iid100, demos_vla_harder_iid100, demos_vla_iid100"
    )

    # the harder scene's second chunk; VLA-0's first runs, before the harder scene's gap is known
    hard += _pool(runs, "planner_harder", HARD, [1, 1], start=4, chunk="eps_4_6")
    vla_spoon = _pool(runs, "vla", SPOON, [1, 1, 1, 0])
    vla_carrot = _pool(runs, "vla", CARROT, [1, 0, 1, 1])
    _pool(runs, "vla_harder", HARD, [0, 0, 0, 0])
    sel = _select(runs)
    assert _line(sel.report, SPOON, f"{SPOON}: 4 episodes") == (
        f"  {SPOON}: 4 episodes, 3 successes (75.0%); needs 3 (and a share of the harder scenes' gap, when it is "
        "known): done"
    )
    assert _line(sel.report, HARD, f"{HARD}: 4 episodes, 0").endswith(f"needs 3: needs more: {HARD}:4:5 {HARD}:5:6")
    vla_p = [spoon[0], spoon[2], spoon[3], carrot[1], carrot[2], hard[0], hard[4], hard[5]]
    assert sel.sets["planner_demos"] == sel.sets["planner_demos_reworded"] == vla_p
    assert sel.sets["planner_demos_450"] == [spoon[0], carrot[1], hard[0]]
    assert sel.sets["planner_demos_1125"] == [spoon[0], spoon[2], carrot[1], carrot[2], hard[0], hard[4]]
    # VLA-PLT: the spoon's best 2 of its 4 successes (three share the shortest detour: the lower episodes win), and one
    # drawn from the others; the carrot's 2 successes are all it has; the harder scene's are VLA-PL's
    rest = [1, 3]  # the spoon's successes 2 and 4, as indices of its successes 0, 2, 3, 4
    drawn = int(np.random.default_rng(0).choice(rest, 1, replace=False)[0])
    chosen = sorted([0, 2, drawn])  # 0 and 2: episodes 0 and 3
    assert sel.sets["targeted_demos_reworded"] == [[spoon[0], spoon[2], spoon[3], spoon[4]][i] for i in chosen] + [
        carrot[1],
        carrot[2],
        hard[0],
        hard[4],
        hard[5],
    ]
    score = 0.5 / (1 + np.exp([1.0, 2.0, 1.0, 1.0]))  # P(planner) x (1 - P(VLA)) of the spoon's 4 successes
    assert _line(sel.report, SPOON, f"{SPOON}: mean score") == (
        f"  {SPOON}: mean score {score[[0, 2]].mean():.3f} of the best 2, {score[drawn]:.3f} of the 1 drawn, "
        f"{score.mean():.3f} of all 4 successes"
    )
    assert _line(sel.report, CARROT, f"{CARROT}: all") == f"  {CARROT}: all 2 successes"
    assert sel.waiting["targeted_corrections_reworded"] == "waiting for demos_corrections_iid101"
    assert sel.waiting["vla_successes_reworded"] == "waiting for demos_vla_harder_iid100"
    assert (
        _line(sel.report, "planner_demos_reworded", "planner_demos_reworded")
        == "  planner_demos_reworded (VLA-PL): ready, 8 recorded episodes, 40 in the set"
    )

    # the harder scene's VLA-0 pool ends at its last episode with no success: a gap of 3, which the spoon fills with 2
    # more (the first task takes the remainder) and the carrot with 1
    _pool(runs, "vla_harder", HARD, [0, 0], start=4, chunk="eps_4_6")
    corrections = _corrections(runs, SPOON, [("holding", True), ("skipped", True)])
    sel = _select(runs)
    assert _line(sel.report, SPOON, f"{SPOON}: 4 episodes, 3 successes").endswith(
        f"needs 5: needs more: {SPOON}:4:5 {SPOON}:5:6"
    )
    assert _line(sel.report, CARROT, f"{CARROT}: 4 episodes, 3 successes").endswith("needs 3: done")
    assert _line(sel.report, HARD, f"{HARD}: 6 episodes, 0").endswith("needs 3: done, short: keeps 0")
    assert _line(sel.report, SPOON, f"{SPOON}: 2 episodes, 1 successful").endswith(
        f"needs 3: needs more: none:{SPOON}:2:4"
    )
    assert _line(sel.report, CARROT, f"{CARROT}: 0 episodes").endswith("its first run has 0 of 2 episodes")
    assert sel.waiting["vla_successes_reworded"] == "waiting for demos_vla_iid100"

    vla_spoon += _pool(runs, "vla", SPOON, [1, 1], start=4, chunk="eps_4_6")
    corrections += _corrections(runs, SPOON, [("grasp", True), ("holding", True)], start=2)
    carrot_corrections = _corrections(
        runs, CARROT, [("grasp", True), ("holding", False), ("holding", True), ("grasp", True)]
    )
    sel = _select(runs)
    assert sel.sets["vla_successes_reworded"] == vla_spoon[:3] + vla_spoon[4:] + [vla_carrot[0], *vla_carrot[2:]]
    assert len(sel.sets["vla_successes_reworded"]) == len(vla_p)  # as many as VLA-PL
    assert sel.sets["targeted_corrections_reworded"] == sel.sets["targeted_demos_reworded"] + [
        corrections[0],
        corrections[2],
        corrections[3],
        carrot_corrections[0],
        *carrot_corrections[2:],
    ]
    assert sel.waiting == {}


def test_the_targeting_checks_the_planner_records(tmp_path):
    runs = tmp_path / "runs"
    _asm(runs)
    _pool(runs, "planner", SPOON, [1, 1, 1, 1, 0, 0])
    _pool(runs, "planner", CARROT, [1, 1, 1, 1, 0, 0])
    _pool(runs, "planner_harder", HARD, [1, 1, 1, 1])
    _planner_records(runs, CARROT, [1, 1, 1, 1, 0, 0], [1.0] * 6)
    _planner_records(runs, SPOON, [1, 1, 1, 0, 0, 0], [1.0] * 6)
    with pytest.raises(SystemExit, match=f"{SPOON} episode 3 succeeded in the recording but not in the planner record"):
        _select(runs)
    _planner_records(runs, SPOON, [1, 1, 1], [1.0] * 3)
    with pytest.raises(SystemExit, match=f"no planner record of {SPOON} episode 3"):
        _select(runs)
    # without the ASM, VLA-PLT waits for it and the others are ready
    os.remove(runs / "energy_model" / "asm_none.json")
    sel = _select(runs)
    assert sel.sets["targeted_demos_reworded"] is None and sel.sets["planner_demos"] is not None
    assert sel.waiting["targeted_demos_reworded"].startswith("waiting for the ASM (")


def test_main(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(tmp_path), "--write", "planner_demos"])
    with pytest.raises(SystemExit):
        sd.main()  # --write needs the set's folders
    assert "--write needs --data-root, --stats and --visual-prompts" in capsys.readouterr().err
    args = [
        "x",
        "--runs",
        str(tmp_path),
        "--write",
        "planner_demos",
        "--data-root",
        str(tmp_path / "data"),
        "--stats",
        "data",
        "--visual-prompts",
        str(tmp_path / "vp"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(
        SystemExit, match="planner_demos is not ready: waiting for demos_planner_iid100, demos_vp_harder_iid100"
    ):
        sd.main()
    assert (
        "demos_planner_iid100 (the planner, fixed rule; iid100; episodes 0 to 699): not run yet"
        in capsys.readouterr().out
    )
    # a ready set is written with the converter, its visual prompts in VP_DIR/SET
    recorder = demos.DemoRecorder(str(tmp_path / "rec"), write_video=lambda path, frames, fps: open(path, "wb").close())
    recorder.start(
        f"{SPOON}/e100/0",
        env_name=SPOON,
        obj_episode_id=0,
        layout_seed=100,
        layout_per_episode=True,
        objects=["bridge_spoon_generated_modified", "table_cloth_generated_shorter"],
        goals=[[0, 1]],
    )
    obs = {"agent": {"base_pose": [0, 0, 0, 1, 0, 0, 0]}, "extra": {"tcp_pose": [0.3, 0, 0.1, 1, 0, 0, 0]}}
    for t in range(3):
        recorder.add(
            np.zeros((8, 8, 3), np.uint8),
            obs,
            demos.action_from_label([0.01, 0, 0, 0, 0, 0, t < 2]),
            "put the spoon on the towel",
            object_map=np.ones((8, 8), np.uint8),
        )
    folder = recorder.finish(True)
    ready = sd.Selection({**dict.fromkeys(sd.SETS), "planner_demos": [folder]}, {}, ["the report"])
    monkeypatch.setattr(sd, "select", lambda runs, asm: ready)
    sd.main()
    out = capsys.readouterr().out
    assert out.startswith("the report\n") and "planner_demos: 1 episodes of 1 recorded, 3 steps, 1 instructions" in out
    assert (tmp_path / "data" / "planner_demos" / "meta" / "info.json").exists()
    assert (tmp_path / "vp" / "planner_demos" / "chunk-000" / "episode_000000.npz").exists()
