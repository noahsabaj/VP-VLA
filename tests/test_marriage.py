"""CPU tests for the marriage study's scorer (experiments/score_marriage.py): run names, units, the planned tests in
their fixed order (H3's intersection-union test, H4's two systems) and evals that are not there."""

import json
import shutil
import sys

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files import novel_scenes as ns
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import score_marriage as sm

HARDER = list(hs.TASKS)
NOVEL = list(ns.TASKS)
SETS = list(INSTRUCTION_SETS)


def _record(episode, success, *, source, level, layout_seed, turn=0.5, feasible=True):
    r = {
        "mode": "planner",
        "obj_episode_id": episode,
        "success": bool(success),
        "features_version": s.FEATURES_VERSION,
        "features": {"grasp_turn_rad": turn, "grasp_reach_m": 0.1},
        "plans": [{"success": True, "time": 0.1, "ee_path_length": 0.45, "ee_straight_length": 0.3}],
        "depth_noise": level,
        "task_source": source,
        "layout_seed": layout_seed,
        "layout_per_episode": layout_seed is not None,
    }
    if source == "vp":
        r["vlm_plan_feasible"] = feasible
    return r


def _planner(root, name, tasks, episodes, success, *, source="rule", level="none", layout_seed=16):
    """A planner-only eval: success(task, episode) gives each outcome; with the VLM marks every 5th episode has no
    target (the gate gives it to the VLA)."""
    for t in tasks:
        folder = root / name / t  # short paths: Windows allows 260 characters
        folder.mkdir(parents=True, exist_ok=True)
        records = [
            _record(
                e,
                success(t, e),
                source=source,
                level=level,
                layout_seed=layout_seed,
                turn=0.2 * (e % 7),
                feasible=source != "vp" or e % 5 != 4,
            )
            for e in range(episodes)
        ]
        (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _vla(root, name, tasks, episodes, success):
    for t in tasks:
        folder = root / name / t
        folder.mkdir(parents=True, exist_ok=True)
        for e in range(episodes):
            (folder / f"{'success' if success(t, e) else 'failure'}_obj_episode_{e}.mp4").write_bytes(b"")


def yes(t, e):
    return True


def no(t, e):
    return False


def quarter(t, e):
    return e % 4 == 0


def _write_study(root, n1=8, n2=3, n3=4, n5=2):
    """Every eval of the planned tests: VLA-P, VLA-PL and VLA-PLT solve every test episode, VLA-0, VLA-S and every
    option without training none; on the ASMs' training sets the planner with the VLM marks fails most and VLA-PL
    succeeds most, so the married system's ASM picks VLA-PL."""
    tasks = list(s.TASKS)
    for prefix, ok in (("vlap", yes), ("vlapl", yes), ("vlaplt", yes), ("vlas", no), (None, no)):
        _vla(root, sm.vla_name(prefix, "own", "iid16"), tasks, n1, ok)
        for g in SETS:
            _vla(root, sm.vla_name(prefix, g, "paper"), tasks, n2, ok)
    for g in SETS:
        _vla(root, sm.vla_name("vlasub", g, "paper"), tasks, n2, no)
        _planner(root, sm.planner_name(sm.RULE, g, "paper", "none"), tasks, n2, no, layout_seed=None)
        _planner(root, sm.planner_name(sm.MARKS, g, "paper", "none"), tasks, n2, no, source="vp", layout_seed=None)
    _vla(root, sm.vla_name(None, "harder", "iid17"), HARDER, n3, no)
    _vla(root, sm.vla_name("vlapl", "harder", "iid17"), HARDER, n3, yes)
    _planner(root, sm.planner_name(sm.RULE, "harder", "iid17", "none"), HARDER, n3, no, layout_seed=17)
    _planner(root, sm.planner_name(sm.MARKS, "harder", "iid17", "none"), HARDER, n3, no, source="vp", layout_seed=17)
    _vla(root, sm.vla_name(None, "novel", "iid18"), NOVEL, n5, no)
    _vla(root, sm.vla_name("vlapl", "novel", "iid18"), NOVEL, n5, yes)
    _planner(root, sm.planner_name(sm.RULE, "novel", "iid18", "none"), NOVEL, n5, no, layout_seed=18)
    _planner(root, sm.planner_name(sm.MARKS, "novel", "iid18", "none"), NOVEL, n5, no, source="vp", layout_seed=18)
    for level in ("none", "harsh"):
        _planner(root, sm.planner_name(sm.RULE, "own", "iid16", level), tasks, n1, no, level=level)
        _planner(root, sm.planner_name(sm.MARKS, "own", "iid16", level), tasks, n1, no, source="vp", level=level)
    # the ASMs' training sets: iid11, and the harder scenes' fixed layouts
    for level in ("none", "harsh"):
        _planner(root, f"pseed31_iid11_{level}", tasks, 20, lambda t, e: e % 2, level=level, layout_seed=11)
        _planner(root, f"vp_iid11_{level}", tasks, 20, quarter, source="vp", level=level, layout_seed=11)
    _vla(root, "vla31_iid11", tasks, 20, lambda t, e: e % 3 == 0)
    _vla(root, "vlapl_iid11", tasks, 20, lambda t, e: e % 10 != 0)
    _planner(root, "pseed31_harder_paper_none", HARDER, 8, lambda t, e: e % 2, layout_seed=None)
    _planner(root, "vp_harder_paper_none", HARDER, 8, quarter, source="vp", layout_seed=None)
    _vla(root, "vla_harder_paper", HARDER, 8, lambda t, e: e % 3 == 0)
    _vla(root, "vlapl_harder_paper", HARDER, 8, lambda t, e: e % 10 != 0)


@pytest.fixture
def small(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "DESIGN", {"T1": 8, "T2": 3, "T3": 4, "T5": 2})
    monkeypatch.setattr(sm, "CONTROL_EPISODES", 2)
    monkeypatch.setattr(sys, "argv", ["score_marriage"])
    return tmp_path


def test_run_names():
    assert sm.vla_name(None, "own", "iid16") == "vla31_iid16"
    assert sm.vla_name(None, "synonyms", "paper") == "vla_words_synonyms_paper"  # the reworded study's
    assert sm.vla_name(None, "harder", "iid17") == "vla_harder_iid17"
    assert sm.vla_name(None, "wrong", "iid16") == "vla31_wrong_iid16"
    assert sm.vla_name("vlapl", "own", "iid11") == "vlapl_iid11"
    assert sm.vla_name("vlapl", "order", "paper") == "vlapl_words_order_paper"
    assert sm.vla_name("vlapl", "harder", "paper") == "vlapl_harder_paper"
    assert sm.vla_name("vlapl", "blank", "iid16") == "vlapl_blank_iid16"
    assert sm.vla_name(None, "novel", "iid18") == "vla_novel_iid18"
    assert sm.vla_name("vlapl", "novel", "iid18") == "vlapl_novel_iid18"
    assert sm.planner_name(sm.RULE, "novel", "iid18", "none") == "pseed31_novel_iid18_none"
    assert sm.planner_name(sm.MARKS, "novel", "iid18", "none") == "vp_novel_iid18_none"
    assert sm.planner_name(sm.RULE, "own", "iid16", "harsh") == "pseed31_iid16_harsh"
    assert sm.planner_name(sm.RULE, "order", "paper", "none") == "words_order_paper"  # the reworded study's
    assert sm.planner_name(sm.RULE, "harder", "paper", "none") == "pseed31_harder_paper_none"
    assert sm.planner_name(sm.MARKS, "own", "iid11", "none") == "vp_iid11_none"
    assert sm.planner_name(sm.MARKS, "order", "paper", "none") == "vp_words_order_paper_none"
    assert sm.planner_name(sm.MARKS, "harder", "iid17", "none") == "vp_harder_iid17_none"
    with pytest.raises(ValueError, match="reworded-instruction study's"):
        sm.planner_name(sm.RULE, "order", "paper", "harsh")
    assert sm.SETS == ["synonyms", "attributes", "order", "reasoning"]


def test_units_follow_the_shared_layouts():
    t1, t2, t3, t5 = (sm.design(p) for p in sm.PARTS)
    assert len(t1) == 100 and {len(ks) for ks in t1.values()} == {4}  # an episode index: its 4 tasks
    assert len(t2) == 96 and {len(ks) for ks in t2.values()} == {4}  # a layout: its 4 reworded sets
    assert len(t3) == 48 and {len(ks) for ks in t3.values()} == {5}  # an episode index: its 5 scenes
    assert len(t5) == 8 * 48 and {len(ks) for ks in t5.values()} == {1}  # an episode: each task has its own layouts
    task = s.TASKS[1]
    assert sm.unit(("T2", "order", task, 3)) == sm.unit(("T2", "synonyms", task, 3)) == f"T2/{task}/3"
    assert sm.unit(("T1", "own", s.TASKS[0], 3)) == sm.unit(("T1", "own", task, 3)) == "T1/3"
    assert sm.unit(("T3", "harder", HARDER[0], 3)) == "T3/3"
    assert sm.unit(("T5", "novel", NOVEL[0], 3)) == f"T5/{NOVEL[0]}/3" != sm.unit(("T5", "novel", NOVEL[1], 3))


def test_complete_and_paired(monkeypatch):
    monkeypatch.setattr(sm, "DESIGN", {"T1": 2, "T2": 2, "T3": 2, "T5": 2})
    keys = [k for ks in sm.design("T2").values() for k in ks]
    new = dict.fromkeys(keys, 1.0)
    base = dict.fromkeys(keys, 0.0)
    assert sm.complete([new, base], ["T2"]) == keys
    del base[("T2", "order", s.TASKS[0], 1)]  # its layout leaves the test, in all 4 sets
    kept = sm.complete([new, base], ["T2"])
    assert len(kept) == len(keys) - 4 and not any(k[2] == s.TASKS[0] and k[3] == 1 for k in kept)
    # one value per unit: a layout up by 2 of its 4 sets, another unchanged
    ks = sm.design("T2")[f"T2/{s.TASKS[2]}/0"] + sm.design("T2")[f"T2/{s.TASKS[2]}/1"]
    a = dict(zip(ks, [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], strict=True))
    b = dict(zip(ks, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], strict=True))
    line, p, mean = sm.paired(a, b, ks)
    assert mean == pytest.approx(0.25) and p == 1.0
    assert "8 episodes, 2 units: 1 up, 0 down" in line


def test_planned_tests_in_order(small, capsys):
    _write_study(small)
    sm.main()
    out = capsys.readouterr().out
    assert "| VLA-PL | 100.0 (32) | 100.0 (12) |" in out and "| VLA-S | 0.0 (32) |" in out
    assert "H1, does the planner teach the VLA? VLA-P minus VLA-0 on T1: +100.0 points" in out
    assert "32 episodes, 8 units: 8 up, 0 down), exact p = 0.0078: a win" in out
    assert "H2, is it the planner's data? VLA-PL minus VLA-S on T1 and T2: +100.0 points" in out
    assert "80 episodes, 20 units: 20 up, 0 down)" in out
    for name in (sm.PLANNERS[sm.RULE], "VLA-0", sm.SUBTASK, sm.PLANNERS[sm.MARKS]):
        assert f"  - vs {name}: +100.0 points" in out
    assert "  intersection-union: p = 0.00049 (the largest of the four): a win" in out
    # H4: the married system's ASM gives VLA-PL the episodes (and the VLM marks' gate gives it more); the current
    # system's methods fail every test episode
    assert "the current system (the fixed rule, VLA-0, the ASM), on T1 (harsh depth), T2 and T3: +100.0 points" in out
    assert "100 episodes, 24 units: 24 up, 0 down)" in out
    # H5: the same on T5, with the ASMs without task columns
    h5 = "H5, does the married system beat the current one on scenes that no model trained on?"
    assert f"{h5} The married system minus the current system, on T5: +100.0 points" in out
    assert "16 episodes, 16 units: 16 up, 0 down), exact p = 3.1e-05: a win" in out
    assert "H6, does choosing where to collect help? VLA-PLT minus VLA-PL on T1 and T2: +0.0 points" in out
    assert "exact p = 1: not a win" in out
    assert "- VLA-PLTD minus VLA-PLT, T1 and T2: not run (it needs vlapltd_iid16, vlapltd_words_synonyms_paper" in out
    assert "### H4's systems on T2 (clean depth; 48 episodes)" in out
    assert "### H5's systems on T5 (clean depth; 16 episodes)" in out
    assert "| System | Row | new object | new pair | reasoning | long | All |" in out
    assert "- VLA-0, blank: not run" in out


def test_a_test_without_its_evals_stops_the_sequence(small, capsys):
    _write_study(small)
    shutil.rmtree(small / "vlas_words_order_paper")
    sm.main()
    out = capsys.readouterr().out
    assert "H1, does the planner teach the VLA?" in out and ": a win" in out
    h2 = "H2, is it the planner's data? VLA-PL minus VLA-S on T1 and T2"
    assert f"{h2}: not run (it needs vlas_words_order_paper)" in out
    assert out.count(": does not count (a test before it was not a win or did not run)") == 4  # H3 to H6


def test_h3_needs_a_win_against_each_option(small, capsys):
    _write_study(small)
    for g in SETS:  # the planner with the VLM marks solves every reworded episode, as VLA-PL does
        shutil.rmtree(small / f"vp_words_{g}_paper_none")
        _planner(small, f"vp_words_{g}_paper_none", list(s.TASKS), 3, yes, source="vp", layout_seed=None)
    sm.main()
    out = capsys.readouterr().out
    assert f"  - vs {sm.PLANNERS[sm.MARKS]}: " in out
    assert "  intersection-union: p = 1 (the largest of the four): not a win" in out
    assert out.count(": does not count (a test before it was not a win or did not run)") == 3  # H4, H5, H6


def test_systems_gate_and_train(small):
    _write_study(small)
    study = sm.Study("iid16", "iid17", "iid11")
    trained, missing = sm.train_asm(study, sm.MARRIED, "none")
    assert not missing and trained.asm.planner.tasks == sm.TASK_COLUMNS  # with the harder scenes' columns
    # the gated episodes (every 5th has no target) are not training data: 16 of 20 per task in iid11
    assert trained.rates[s.TASKS[0]] == (pytest.approx(4 / 16), pytest.approx(14 / 16))
    harsh, _ = sm.train_asm(study, sm.CURRENT, "harsh")
    assert harsh.asm.planner.tasks == list(s.TASKS)
    scored, missing = sm.score_system(study, sm.MARRIED, sm.SYSTEM_TESTS["H4"])
    assert not missing
    gated = ("T1", "own", s.TASKS[0], 4)  # no target: the VLA, whatever the ASM says
    assert gated not in scored.probs and scored.rows[sm.ASM][gated] == scored.rows[sm.VLA_ONLY][gated] == 1.0
    assert all(v == 1.0 for v in scored.rows[sm.ASM].values())
    _, missing = sm.score_system(sm.Study("iid16", "iid17", "iid14"), sm.CURRENT, sm.SYSTEM_TESTS["H4"])
    assert "pseed31_iid14_harsh" in missing and "vla31_iid14" in missing


def test_new_tasks_take_an_asm_without_task_columns(small):
    _write_study(small)
    study = sm.Study("iid16", "iid17", "iid11")
    trained, missing = sm.train_asm(study, sm.MARRIED, "none", task_free=True)
    assert not missing and trained.asm.planner.tasks == trained.asm.vla.tasks == [sm.ANY]
    # all the clean-depth ASM's training episodes, as one task: 16 of 20 per task in iid11, 7 of 8 per harder scene
    assert trained.rates == {sm.ANY: (pytest.approx(21 / 99), pytest.approx(86 / 99))}
    model = sm.train_asm(study, sm.MARRIED, "none")[0].asm.planner  # with task columns, it cannot score T5
    with pytest.raises(ValueError, match="no task column for"):
        model.columns(np.zeros((1, len(model.features))), [NOVEL[0]])
    scored, missing = sm.score_system(study, sm.MARRIED, sm.SYSTEM_TESTS["H5"])
    assert not missing and {k[0] for k in scored.rows[sm.ASM]} == {"T5"}
    assert all(v == 1.0 for v in scored.rows[sm.ASM].values())  # P(VLA) is the higher: VLA-PL solves them
    _, missing = sm.score_system(study, sm.CURRENT, sm.SYSTEM_TESTS["H5"])
    assert not missing
    shutil.rmtree(small / "vp_novel_iid18_none")
    _, missing = sm.score_system(sm.Study("iid16", "iid17", "iid11"), sm.MARRIED, sm.SYSTEM_TESTS["H5"])
    assert missing == ["vp_novel_iid18_none"]


def test_controls_and_planner_only(small, capsys):
    _write_study(small)
    _vla(small, "vlapl_blank_iid16", list(s.TASKS), 2, no)
    np.testing.assert_equal(
        sm.Study("iid16", "iid17", "iid11").control("vlapl", "blank"),
        {("T1", "own", t, e): 0.0 for t in s.TASKS for e in range(2)},
    )
    sm.main()
    out = capsys.readouterr().out
    assert "- VLA-PL, blank: 0.0 against 100.0 with the own instruction (8 episodes; 0 / 8, p = 0.0078)" in out
    assert "| planner (VLM marks), harsh depth | 0.0 (32) |" in out
