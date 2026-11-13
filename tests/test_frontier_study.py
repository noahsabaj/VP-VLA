"""CPU tests for the frontier-model study's scorer (experiments/score_frontier.py) on made-up evals: the run names,
the two families of planned tests in their fixed order, the test sets that the agent ran, evals that are not there
and the secondary results."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files import novel_scenes as ns
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS
from examples.SimplerEnv.eval_files.planner import frontier
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner.experiments import frontier_answers as fa
from examples.SimplerEnv.eval_files.planner.experiments import score_frontier as sf
from examples.SimplerEnv.eval_files.planner.experiments import score_marriage as sm

README = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments/README.md"
TASKS, HARDER, NOVEL, SETS = list(s.TASKS), list(hs.TASKS), list(ns.TASKS), list(INSTRUCTION_SETS)
N2, N3, N5 = 3, 4, 2  # the made-up design's episodes of each task in T2 (in each set), T3 and T5
WHOLE = "84 episodes, 32 units"  # T2: 4 sets x 4 tasks x 3, in 12 units; T3: 5 x 4, in 4; T5: 8 x 2, in 16


def yes(t, e):
    return True


def no(t, e):
    return False


def _record(episode, success, *, source, level, layout_seed, saved=False, refused=False):
    r = {
        "mode": "planner",
        "obj_episode_id": episode,
        "success": bool(success),
        "features_version": s.FEATURES_VERSION,
        "features": {"grasp_turn_rad": 0.2 * (episode % 7), "grasp_reach_m": 0.1},
        "plans": [{"success": True, "time": 0.1, "ee_path_length": 0.45, "ee_straight_length": 0.3}],
        "depth_noise": level,
        "task_source": source,
        "layout_seed": layout_seed,
        "layout_per_episode": layout_seed is not None,
    }
    if source == "vp":
        r["vlm_plan_feasible"] = episode % 5 != 4  # every 5th episode has no target: the gate gives it to the VLA
    if source == "vlm":
        r["vlm_plan_feasible"] = not refused
        r["vlm_plan_text"] = "" if refused else f"the plan of episode {episode}"
        # the object's point lies in its mask, the target's does not
        r["vlm_plan_sam"] = [] if refused else [{"which": "grasp", "covers_point": True},
                                                 {"which": "place", "covers_point": False}]  # fmt: skip
        if saved:
            r["vlm_plan_answer"] = {"file": "a.jsonl", "model": fa.MODEL, "error": "refusal" if refused else None}
    return r


def _planner(root, name, tasks, episodes, success, *, source="vp", level="none", layout_seed=16, **kwargs):
    for t in tasks:
        folder = root / name / t  # short paths: Windows allows 260 characters
        folder.mkdir(parents=True, exist_ok=True)
        records = [
            _record(e, success(t, e), source=source, level=level, layout_seed=layout_seed, **kwargs)
            for e in range(episodes)
        ]
        (folder / "planner_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def _vla(root, name, tasks, episodes, success):
    for t in tasks:
        folder = root / name / t
        folder.mkdir(parents=True, exist_ok=True)
        for e in range(episodes):
            (folder / f"{'success' if success(t, e) else 'failure'}_obj_episode_{e}.mp4").write_bytes(b"")


def _agent(root, name, tasks, episodes, success, end=lambda t, e: "done"):
    for t in tasks:
        folder = root / name / t
        folder.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "obj_episode_id": e,
                "success": bool(success(t, e)),
                "turns": 10 + e,
                "end": end(t, e),
                "refusal": False,
                "error": None,
                "usage": {"input_tokens": 9000, "output_tokens": 1000},
                "cost_usd": 0.5,
                "model_seconds": 60.0,
            }
            for e in range(episodes)
        ]
        (folder / frontier.AGENT_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _sets(root, write, name, *args, **kwargs):
    """One row's evals of T2, T3 and T5: name(group, layouts) gives each eval's name."""
    for g in SETS:
        write(root, name(g, "paper"), TASKS, N2, *args, **{**kwargs, **({"layout_seed": None} if kwargs else {})})
    write(root, name("harder", "iid17"), HARDER, N3, *args, **{**kwargs, **({"layout_seed": 17} if kwargs else {})})
    write(root, name("novel", "iid18"), NOVEL, N5, *args, **{**kwargs, **({"layout_seed": 18} if kwargs else {})})


def _plan(root, row, success, **kwargs):
    _sets(root, _planner, lambda g, layouts: sf.plan_name(sf.PLAN_PREFIX[row], g, layouts, "none"), success,
          source="vlm", saved=row == sf.FPLAN, **kwargs)  # fmt: skip


def _write_study(root, fplan=yes, qplan=no, marks=no, vlapl=no, agent=no):
    """Every eval of the planned tests. On the ASM's training sets the planner with the VLM's marks fails most and
    VLA-PL succeeds most, so the married system gives its episodes to VLA-PL."""
    _plan(root, sf.FPLAN, fplan)
    _plan(root, sf.QPLAN, qplan)
    _sets(root, _planner, lambda g, layouts: sm.planner_name(sm.MARKS, g, layouts, "none"), marks, source="vp")
    _sets(root, _vla, lambda g, layouts: sm.vla_name("vlapl", g, layouts), vlapl)
    if agent is not None:
        _sets(root, _agent, sf.agent_name, agent)
    # the married system's ASM: iid11, and the harder scenes' fixed layouts
    _planner(root, "vp_iid11_none", TASKS, 20, lambda t, e: e % 4 == 0, layout_seed=11)
    _vla(root, "vlapl_iid11", TASKS, 20, lambda t, e: e % 10 != 0)
    _planner(root, "vp_harder_paper_none", HARDER, 8, lambda t, e: e % 4 == 0, layout_seed=None)
    _vla(root, "vlapl_harder_paper", HARDER, 8, lambda t, e: e % 10 != 0)


@pytest.fixture
def small(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "DESIGN", {"T1": 8, "T2": N2, "T3": N3, "T5": N5})
    monkeypatch.setattr(sys, "argv", ["score_frontier"])
    return tmp_path


def _out(capsys):
    sf.main()
    return capsys.readouterr().out


def _line(out, start):
    return next(line for line in out.splitlines() if line.startswith(start))


def test_run_names_and_the_plan(small, monkeypatch):
    assert sf.plan_name("fplan", "own", "iid16", "none") == "fplan_iid16_none"
    assert sf.plan_name("fplan", "own", "iid16", "harsh") == "fplan_iid16_harsh"
    assert sf.plan_name("fplan", "order", "paper", "none") == "fplan_words_order_paper_none"
    assert sf.plan_name("fplan", "harder", "iid17", "none") == "fplan_harder_iid17_none"
    assert sf.plan_name("vlmplan", "novel", "iid18", "none") == "vlmplan_novel_iid18_none"
    assert sf.plan_name("vlmplan", "synonyms", "paper", "none") == "vlmplan_words_synonyms_paper_none"
    assert sf.agent_name("reasoning", "paper") == "fagent_words_reasoning_paper"
    assert sf.agent_name("harder", "iid17") == "fagent_harder_iid17"
    assert sf.agent_name("novel", "iid18") == "fagent_novel_iid18"
    # the plan: two families at 0.025, T2 + T3 + T5 with the marriage study's design, the agent's sets in its order
    assert sf.ALPHA == 0.025 and sf.TEST_PARTS == ("T2", "T3", "T5") and sf.AGENT_ORDER == ("T5", "T2", "T3")
    monkeypatch.undo()  # the real design
    monkeypatch.setenv("RUNS_DIR", str(small))
    assert [sm.DESIGN[p] * len(sm.groups(p)) * len(sm.group_tasks(sm.groups(p)[0])) for p in sf.TEST_PARTS] == [
        384,
        240,
        384,
    ]
    assert fa.MODEL == "claude-fable-5-2" and fa.EFFORT == "high"
    fr = sf.Frontier()
    assert [[name for name, *_ in tests] for _, tests in sf.families(fr).values()] == [
        ["F1", "F2", "F3"],
        ["A1", "A2", "A3"],
    ]
    text = README.read_text(encoding="utf-8")
    block = text[text.index("## Evals (GPU)") : text.index("## Result files (CPU)")]
    for name in ("fplan_iid16_none", "fplan_iid16_harsh", "fplan_harder_iid17_none", "fplan_novel_iid18_none",
                 "vlmplan_iid16_none", "vlmplan_harder_iid17_none", "vlmplan_novel_iid18_none"):  # fmt: skip
        assert f" {name} " in block, name  # the README's commands write the folders that the scorer reads
    for name in ("fplan_words_${set}_paper_none", "vlmplan_words_${set}_paper_none"):
        assert f" {name} " in block, name


def test_the_plan_arm_wins_and_the_agent_loses(small, capsys):
    _write_study(small)
    out = _out(capsys)
    assert not any(line.startswith("Evals that are not there") for line in out.splitlines())
    assert "| frontier plan | - | 100.0 (48) | 100.0 (20) | 100.0 (16) | 100.0 (84) |" in out
    assert "| married system | - | 0.0 (48) | 0.0 (20) | 0.0 (16) | 0.0 (84) |" in out
    assert "### The plan arm: T2 + T3 + T5" in out and "### The agent arm: T2 + T3 + T5" in out
    up = f"+100.0 points (95% interval +100.0 to +100.0; {WHOLE}: 32 up, 0 down), exact p = 4.7e-10"
    assert f"  - vs Qwen3-VL's plan: {up}" in out and f"  - vs the VLM's marks: {up}" in out
    assert "  intersection-union: p = 4.7e-10 (the largest of the 2): a win" in out
    assert _line(out, "F2, ").endswith(f"The frontier plan minus VLA-PL: {up}: a win")
    assert _line(out, "F3, ").endswith(f"The frontier plan minus the married system: {up}: a win")
    down = f"-100.0 points (95% interval -100.0 to -100.0; {WHOLE}: 0 up, 32 down), exact p = 4.7e-10"
    assert _line(out, "A1, ").endswith(
        f"The frontier agent minus the frontier plan: {down}: a loss (significant, with the frontier arm lower)"
    )
    # the agent against VLA-PL: no difference, and it does not count after A1
    for name in ("A2", "A3"):
        assert _line(out, f"{name}, ").endswith(
            "exact p = 1: does not count (a test before it in its family was not a win or did not run)"
        )


def test_f1_needs_a_win_against_each_of_the_two(small, capsys):
    _write_study(small, marks=yes)  # the VLM's marks solve every episode, as the frontier plan does
    out = _out(capsys)
    assert "  intersection-union: p = 1 (the largest of the 2): not a win (not significant)" in out
    for name in ("F2", "F3"):
        assert "does not count (a test before it in its family" in _line(out, f"{name}, ")
    # the families are judged apart: the agent's family still runs (the agent, at 0, is below the frontier plan)
    assert "a loss (significant, with the frontier arm lower)" in _line(out, "A1, ")


def test_a_frontier_plan_that_is_lower_than_each_is_a_loss(small, capsys):
    _write_study(small, fplan=no, qplan=yes, marks=yes, agent=yes)
    out = _out(capsys)
    assert "(the largest of the 2): a loss (significant, with the frontier arm lower)" in out
    assert _line(out, "A1, ").endswith("32 up, 0 down), exact p = 4.7e-10: a win")
    assert _line(out, "A2, ").endswith("32 up, 0 down), exact p = 4.7e-10: a win")  # VLA-PL solves none
    assert _line(out, "A3, ").endswith(": a win")


def test_significant_at_005_only_is_not_a_win(small, capsys):
    # the frontier plan is higher in 5 units (T3's 4 episode indices and one T5 episode) and equal elsewhere:
    # p = 2 / 2^5 = 0.0625; in 6 units: 0.03125, below 0.05 but not below the family's 0.025
    first = NOVEL[0]
    _write_study(small, fplan=lambda t, e: t in HARDER or (t == first and e < 2), qplan=no, marks=no)
    out = _out(capsys)
    assert "6 up, 0 down), exact p = 0.031" in out
    assert "  intersection-union: p = 0.031 (the largest of the 2): not a win (not significant)" in out


def test_the_agents_tests_take_the_sets_that_ran(small, capsys):
    _write_study(small, agent=None)
    out = _out(capsys)
    assert "### The agent arm: no test set ran" in out
    assert _line(out, "A1, ").endswith("not run (it needs a test set that the agent ran)")
    assert "The frontier agent: no eval yet" in out
    # T5 whole, and two of T2's four reworded sets: the tests take T5
    _agent(small, "fagent_novel_iid18", NOVEL, N5, no)
    for g in SETS[:2]:
        _agent(small, sf.agent_name(g, "paper"), TASKS, N2, no)
    out = _out(capsys)
    assert "### The agent arm: T5" in out
    assert "16 episodes, 16 units: 0 up, 16 down), exact p = 3.1e-05: a loss" in _line(out, "A1, ")
    assert _line(out, "note: T2 is left out of the agent arm's tests").endswith(
        "it has fagent_words_synonyms_paper, fagent_words_attributes_paper but not fagent_words_order_paper, "
        "fagent_words_reasoning_paper"
    )
    assert "The frontier agent (16 episodes of T5)" in out


def test_a_test_without_its_evals_is_not_run(small, capsys):
    _write_study(small)
    shutil.rmtree(small / "fplan_novel_iid18_none")
    shutil.rmtree(small / "vlapl_harder_iid17")
    out = _out(capsys)
    assert "Evals that are not there: fplan_novel_iid18_none, vlapl_harder_iid17" in out
    assert _line(out, "F1, ").endswith("not run (it needs fplan_novel_iid18_none)")
    assert _line(out, "F2, ").endswith("not run (it needs fplan_novel_iid18_none, vlapl_harder_iid17)")
    assert _line(out, "A2, ").endswith("not run (it needs vlapl_harder_iid17)")
    # an eval with a part of its episodes: only the units that every eval has in full
    _plan(small, sf.FPLAN, yes)
    _vla(small, "vlapl_harder_iid17", HARDER, N3, no)
    (small / "fplan_novel_iid18_none" / NOVEL[0] / "planner_stats.jsonl").write_text(
        json.dumps(_record(0, True, source="vlm", level="none", layout_seed=18, saved=True)) + "\n"
    )
    out = _out(capsys)
    assert "83 episodes, 31 units: 31 up, 0 down)" in _line(out, "F2, ")


def test_a_record_of_another_eval_stops_the_load(small):
    _write_study(small)
    fr = sf.Frontier()
    assert not fr.plan(sf.FPLAN, sf.TEST_PARTS)[1] and len(fr.row(sf.FPLAN, sf.TEST_PARTS)[0]) == 84
    # Qwen3-VL's eval under the frontier plan's name: its records have no saved answer
    shutil.rmtree(small / "fplan_harder_iid17_none")
    _planner(small, "fplan_harder_iid17_none", HARDER, N3, yes, source="vlm", layout_seed=17)
    with pytest.raises(SystemExit, match=r"fplan_harder_iid17_none: .* is not an episode of the frontier plan"):
        sf.Frontier().plan(sf.FPLAN, ["T3"])
    # the frontier plan's eval under Qwen3-VL's name, and an eval with harsh depth
    shutil.rmtree(small / "vlmplan_novel_iid18_none")
    _planner(small, "vlmplan_novel_iid18_none", NOVEL, N5, yes, source="vlm", layout_seed=18, saved=True)
    with pytest.raises(SystemExit, match="is not an episode of Qwen3-VL's plan"):
        sf.Frontier().plan(sf.QPLAN, ["T5"])
    shutil.rmtree(small / "vlmplan_novel_iid18_none")
    _planner(small, "vlmplan_novel_iid18_none", NOVEL, N5, yes, source="vlm", layout_seed=18, level="harsh")
    with pytest.raises(SystemExit, match="at none depth"):
        sf.Frontier().plan(sf.QPLAN, ["T5"])
    # an agent record with an end that the plan does not have, and a second record for an episode
    file = small / "fagent_novel_iid18" / NOVEL[0] / frontier.AGENT_FILE
    rows = file.read_text().splitlines()
    file.write_text("\n".join([*rows, rows[0]]) + "\n")
    with pytest.raises(SystemExit, match="a second record for"):
        sf.Frontier().agent_records(["T5"])
    file.write_text(json.dumps({**json.loads(rows[0]), "end": "timeout"}) + "\n")
    with pytest.raises(SystemExit, match="is not one of"):
        sf.Frontier().agent_records(["T5"])


def test_the_married_system_follows_the_marriage_studys_rules(small):
    _write_study(small, vlapl=yes)
    fr = sf.Frontier()
    married, missing = fr.row(sf.MARRIED, sf.TEST_PARTS)
    assert not missing and len(married) == 84 and all(v == 1.0 for v in married.values())  # it gives them to VLA-PL
    for rule in sf.MARRIED_RULES:  # H4's rule on T2 and T3, H5's (an ASM without task columns) on T5
        scored, _ = sm.score_system(fr.study, sm.MARRIED, rule)
        assert {k: v for k, v in married.items() if k[0] in rule} == scored.rows[sm.ASM]
    assert fr.row(sf.MARRIED, ["T1", "T5"])[0].keys() == {k for k in married if k[0] == "T5"}  # no T1 for this row
    shutil.rmtree(small / "vlapl_iid11")
    assert sf.Frontier().row(sf.MARRIED, sf.TEST_PARTS) == ({}, ["vlapl_iid11"])


def test_secondary_results(small, capsys):
    _write_study(small, agent=lambda t, e: e % 2 == 0)
    for g in SETS:  # the agent gives up in every reworded episode
        shutil.rmtree(small / sf.agent_name(g, "paper"))
        _agent(small, sf.agent_name(g, "paper"), TASKS, N2, no, end=lambda t, e: "give_up")
    # T1: the frontier plan with clean and with harsh depth (the same answers), with one refusal in each task
    for level, ok in (("none", yes), ("harsh", lambda t, e: e % 2 == 0)):
        name = sf.plan_name("fplan", "own", "iid16", level)
        _planner(small, name, TASKS, 7, ok, source="vlm", level=level, saved=True)
        for t in TASKS:
            with open(small / name / t / "planner_stats.jsonl", "a") as f:
                f.write(json.dumps(_record(7, False, source="vlm", level=level, layout_seed=16, saved=True,
                                           refused=True)) + "\n")  # fmt: skip
    answers = small / "fanswers"
    answers.mkdir()
    row = {"key": "k", "instruction": "i", "text": "{}", "error": None, "request": {"model": fa.MODEL},
           "usage": {"input_tokens": 1000, "output_tokens": 500}, "seconds": 20.0}  # fmt: skip
    (answers / "novel_iid18.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps({**row, "key": "k2", "text": "", "error": "refusal"}) + "\n"
    )
    (answers / "harder_iid17.jsonl").write_text(json.dumps({**row, "error": "error: Outage", "usage": {}}) + "\n")
    out = _out(capsys)
    assert "| frontier plan | 87.5 (32) | 100.0 (48) | 100.0 (20) | 100.0 (16) | 100.0 (84) |" in out
    assert "| frontier agent | - | 0.0 (48) | 50.0 (20) | 50.0 (16) | 21.4 (84) |" in out
    assert "| Row | " + " | ".join(SETS) + " |" in out
    assert "T5 by task:\n| Row | " + " | ".join(ns.SHORT[t] for t in NOVEL) + " |" in out
    assert "- clean depth: 87.5 (32) % (episodes)" in out and "- harsh depth: 50.0 (32) % (episodes)" in out
    assert "- harsh minus clean: -37.5 points" in out
    assert "- episodes whose answer is not the same at the two depth levels: 0 of 32" in out
    # the plans: 4 refusals of T1's 32 episodes; 2 names in each other episode, the object's point in its mask
    assert "| frontier plan | T1 | 32 | 12.5 (4) | 4, 0 | 56 | 50.0 (28) |" in out
    assert "| frontier plan | T2 | 48 | 0.0 (0) | 0, 0 | 96 | 50.0 (48) |" in out
    assert "| Qwen3-VL's plan | T5 | 16 | 0.0 (0) | 0, 0 | 32 | 50.0 (16) |" in out
    assert f"note: the frontier plan's saved answers came from: {fa.MODEL} (116 episodes)" in out
    # the agent: turns 10 + e; done in T3 and T5 (36 episodes, half of them solved), give_up in T2 (48)
    assert "The frontier agent (84 episodes of T2 + T3 + T5): 11.0 turns per episode (median 11, most 13)" in out
    assert "| done | 42.9 (36) | 50.0 |" in out and "| give_up | 57.1 (48) | 0.0 |" in out
    assert "| turn limit | 0.0 (0) | - |" in out
    assert "- tokens per episode: input_tokens 9,000, output_tokens 1,000" in out
    assert "- cost: 42.00 USD (0.500 per episode, 84 episodes); the model's time per episode: 60 s" in out
    # the plan arm's requests, from the answers files
    assert "| harder_iid17.jsonl | 1 | 0 | 1 | 0 | 0 | 0.00 | 20.0 |" in out
    assert "| novel_iid18.jsonl | 2 | 1 | 0 | 1,000 | 500 | 0.06 | 20.0 |" in out
    assert "| all | 3 | 1 | 1 | 667 | 333 | 0.06 | 20.0 |" in out
