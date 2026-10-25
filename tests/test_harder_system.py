"""CPU tests for the harder-task system study: a pooled ASM with its own task columns (planner/selector.py) and the
study's scorer (experiments/score_harder_system.py)."""

import json
import shutil
import sys

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import score_harder_system as hsys

HARDER = list(hs.TASKS)


def _record(episode, success, *, plan_ok=True, turn=0.5, feasible=True, layout_seed=11, source="vlm"):
    r = {
        "mode": "planner",
        "obj_episode_id": episode,
        "success": success,
        "features_version": s.FEATURES_VERSION,
        "features": {"grasp_sam3_score": 0.5, "place_sam3_score": 0.5, "grasp_turn_rad": turn, "grasp_reach_m": 0.1},
        "plans": [{"success": plan_ok, "time": 0.1, "ee_path_length": 0.45, "ee_straight_length": 0.3}],
        "depth_noise": "harsh",
        "task_source": source,
        "layout_seed": layout_seed,
        "layout_per_episode": layout_seed is not None,
    }
    if source == "vlm":
        steps = [{"object": "spoon", "target": "towel"}] if feasible else []
        r.update(vlm_plan={"feasible": feasible, "steps": steps, "error": None}, vlm_plan_feasible=feasible)
        if not feasible:
            r.update(success=False, features={}, plans=[])
    return r


def _write_planner(root, task, records):
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _write_video(root, task, episode, success):
    folder = root / task  # short paths: Windows allows 260 characters
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{'success' if success else 'failure'}_obj_episode_{episode}.mp4").write_bytes(b"")


def _write_eval(root, planner, vla, tasks, episodes, rng, *, layout_seed, source="vlm", hard=False):
    """A planner-only eval and its VLA-only eval (vla None: none): the planner succeeds with a plan, the VLA with a
    small turn; on the harder tasks the planner fails more. Every 7th VLM plan is not pick-and-place."""
    for t in tasks:
        records = []
        for e in range(episodes):
            turn, plan_ok = float(rng.uniform(0, 1.5)), bool(rng.random() < (0.5 if hard else 0.8))
            feasible = source == "rule" or e % 7 != 3
            records.append(
                _record(
                    e, plan_ok, plan_ok=plan_ok, turn=turn, feasible=feasible, layout_seed=layout_seed, source=source
                )
            )
            if vla is not None:
                _write_video(root / vla, t, e, turn < 0.75)
        _write_planner(root / planner, t, records)


def _write_study(root, rng, groups=hsys.GROUPS):
    _write_eval(root, "vlmplan_iid11_harsh", "vla31_iid11", s.TASKS, 30, rng, layout_seed=11)
    _write_eval(root, hsys.HARDER_TRAIN[0], hsys.HARDER_TRAIN[1], HARDER, 20, rng, layout_seed=None, hard=True)
    _write_eval(root, "pseed31_iid11_harsh", None, s.TASKS, 30, rng, layout_seed=11, source="rule")
    for g in groups:
        tasks, n = (HARDER, 16) if g == "harder" else (s.TASKS, 8)
        _write_eval(
            root,
            hsys.group_run("vlmplan", g, "iid13"),
            hsys.vla_run(g, "iid13"),
            tasks,
            n,
            rng,
            layout_seed=13,
            hard=g == "harder",
        )
        _write_eval(root, hsys.group_run("pseed31", g, "iid13"), None, tasks, n, rng, layout_seed=13, source="rule")


def test_run_names():
    assert hsys.GROUPS == ["own", *INSTRUCTION_SETS, "harder"]
    assert hsys.group_run("vlmplan", "synonyms", "iid13") == "vlmplan_words_synonyms_iid13_harsh"
    assert hsys.group_run("pseed31", "harder", "iid13") == "pseed31_harder_iid13_harsh"
    assert hsys.vla_run("own", "iid13") == "vla_own_iid13"
    assert hsys.TASK_COLUMNS == [*s.TASKS, *HARDER]


def test_merged_settings():
    parts = [
        {"depth_noise": "harsh", "task_source": "vlm", "layout_seed": 11},
        {"depth_noise": "harsh", "task_source": "vlm", "layout_seed": None, "layout_per_episode": False},
    ]
    merged = s.merged_settings(parts)
    assert merged["depth_noise"] == "harsh" and merged["task_source"] == "vlm"
    assert merged["layout_seed"] is None and merged["layout_per_episode"] is None
    assert merged["pooled"] == {"layout_seed": [11, None], "layout_per_episode": [False]}
    assert s.merged_settings(parts[:1]) == parts[0]  # one part: nothing pooled
    with pytest.raises(SystemExit, match="different task_source"):
        s.merged_settings([parts[0], {**parts[0], "task_source": "rule"}])
    # a part written before missing_depth existed had none
    assert s.merged_settings([{**parts[0], "missing_depth": "none"}, parts[0]])["missing_depth"] == "none"
    with pytest.raises(SystemExit, match="different missing_depth"):
        s.merged_settings([{**parts[0], "missing_depth": "target"}, parts[0]])


def test_layout_groups_hold_out_a_layout_in_every_set_and_level():
    def rec(e, seed, per_episode=True):
        return {"obj_episode_id": e, "layout_seed": seed, "layout_per_episode": per_episode}

    g = s.layout_groups([rec(3, 11), rec(3, 11), rec(4, 11), rec(3, None, False), rec(3, 13)])
    assert g[0] == g[1] and len(set(g.tolist())) == 4  # the same layout at two depth levels: one group
    # the harder scenes' fixed layouts: episodes 2k and 2k + 1 have the same xy config, in every task of a scene
    fixed = [{**rec(e, None, False), "task_folder": t} for t in HARDER[:2] for e in range(4)]
    assert s.layout_groups(fixed).tolist() == [0, 0, 1, 1] * 2
    own = [{**rec(e, None, False), "task_folder": s.TASKS[0]} for e in range(2)]  # SimplerEnv's own: by episode
    assert s.layout_groups(own).tolist() == [0, 1]


def test_train_pooled_with_more_task_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    _write_study(tmp_path, np.random.default_rng(0), groups=[])
    parts = hsys.training_parts("iid11")
    asm = s.train_pooled(parts, hsys.CANDIDATES, task_columns=hsys.TASK_COLUMNS)
    assert asm.planner.tasks == asm.vla.tasks == hsys.TASK_COLUMNS
    assert asm.trained_on["pooled"] == {"layout_seed": [11, None], "layout_per_episode": [True, False]}
    copy = s.TwoModelASM.from_json(json.loads(json.dumps(asm.to_json())))
    r = _record(0, True)
    assert copy.decide(r, HARDER[2]) == asm.decide(r, HARDER[2])
    with pytest.raises(ValueError, match="no task column for .*" + HARDER[0]):
        s.train_pooled(parts)  # the default columns: SimplerEnv's 4 tasks


def test_score_harder_system(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(hsys, "DESIGN_EPISODES", {g: 16 if g == "harder" else 8 for g in hsys.GROUPS})
    _write_study(tmp_path, np.random.default_rng(1), groups=hsys.GROUPS[:-1])  # the harder scenes' test has not run
    monkeypatch.setattr(sys, "argv", ["score_harder_system"])
    hsys.main()
    out = capsys.readouterr().out
    assert "note: no vlmplan_harder_iid13_harsh, vla_harder_iid13" in out
    assert "not run: they need the planner-only and the VLA-only evals of every instruction group" in out

    _write_study(tmp_path, np.random.default_rng(1))
    hsys.main()
    out = capsys.readouterr().out
    assert "# Harder-task system study: trained on vlmplan_iid11_harsh and vlmplan_harder_paper_harsh" in out
    assert f"{5 * 4 * 8 + 5 * 16} episodes, 16 episode indices:" in out
    assert "| Setup | Own | synonyms | attributes | order | reasoning | Harder scenes | All |" in out
    for i, base in enumerate(hsys.SEQUENCE, 1):
        assert f"{i}. vs {base}: " in out
    assert "Retrieval router: k = " in out and "- vs rule A: " in out
    assert "| Two-step |" in out and "### What makes the ASM better" in out
    assert "### The fixed rule (secondary): pseed31_*_iid13_harsh, 240 episodes" in out
    assert "(SimplerEnv's tasks, 160 episodes)" in out

    # an episode that one eval did not run takes its whole episode index out: 20 + 5 episodes at index 2
    next((tmp_path / "vla_words_order_iid13" / s.TASKS[1]).glob("*_obj_episode_2.mp4")).unlink()
    hsys.main()
    out = capsys.readouterr().out
    assert "note: 1 episode indices without every task of every group in both evals, left out" in out
    assert "215 episodes, 15 episode indices:" in out and "1. vs VLM -> planner (planner-only): " in out
    # a VLA-only eval that did not run: no planned tests
    shutil.rmtree(tmp_path / "vla_words_order_iid13")
    hsys.main()
    out = capsys.readouterr().out
    assert "note: no vla_words_order_iid13" in out and "not run: they need the planner-only and the VLA-only" in out


def test_tests_take_one_value_per_episode_index():
    keys = [("own", s.TASKS[0], 0), ("synonyms", s.TASKS[0], 0), ("harder", HARDER[0], 0), ("harder", HARDER[0], 1)]
    ls = hsys.LayoutSet(keys, [{}] * 4, np.array([k[1] for k in keys]), np.zeros(4), np.zeros(4))
    new, base = np.array([1.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])
    line, p, mean = hsys.paired(new, base, hsys.units(ls))
    assert mean == pytest.approx(0.25) and p == 1.0  # index 0: +1, index 1: 0
    assert "1 episode indices up / 0 down" in line
