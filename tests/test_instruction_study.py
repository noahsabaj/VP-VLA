"""CPU tests for the instruction study's scorer (experiments/score_instruction.py) on made-up run folders: the
planned tests I1 to I3 with Holm's method, the secondary results, and evals that are not there."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from examples.SimplerEnv.eval_files import choice_scenes as cs
from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files import model_time as mt
from examples.SimplerEnv.eval_files.planner.experiments import score_instruction as si

README = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments/README.md"
GOAL, OTHER, NEITHER = "goal", "other", "neither"  # where a task's object ends: its target, the other one, neither
SPOON_NEW, CARROT_NEW = cs.NEW


def follows(task, e):
    """Does what the instruction says."""
    return GOAL


def habit(task, e):
    """Does the same thing whatever the instruction says: each object on the target that it was trained with."""
    return GOAL if task in cs.TRAINED else OTHER


def never(task, e):
    return NEITHER


def first_fail(n, task=SPOON_NEW):
    """Does what the instruction says, but in the first n episodes of one task its object ends on neither target."""
    return lambda t, e: NEITHER if t == task and e < n else GOAL


def _eval(root, name, ends, *, episodes=48, seed=19, chunk=None, stats=True, skip=()):
    """An eval's episode records as the evaluator writes them (model_time.cost_record with the episode stats), one
    file per task; chunk: in a folder of the task, as a VLA-only eval's chunks. skip: (task, episode) not run."""
    for task, t in cs.TASKS.items():
        folder = root / name / task / (chunk or "")  # short paths: Windows allows 260 characters
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for e in range(episodes):
            if (task, e) in skip:
                continue
            end = ends(task, e)
            info = {"env_name": task, "obj_episode_id": e, "layout_seed": seed, "layout_per_episode": True}
            if stats:
                info["episode_stats"] = {
                    hs.pair_name(t, t.goals[0]): end == GOAL,
                    hs.pair_name(t, t.logged[0]): end == OTHER,
                    "moved_other_obj": False,
                }
            lines.append(json.dumps(mt.cost_record(f"{task}/e{seed}/{e}", end == GOAL, 120, 30.0, mt.Meter(), **info)))
        (folder / "episode_costs.jsonl").write_text("".join(line + "\n" for line in lines))


def _study(root, vla0=never, vlapl=habit, rule=follows, marks=follows, seed=19, **kwargs):
    """The study's four evals on the layout set of a layout seed: by default VLA-0 solves nothing, VLA-PL only the
    trained pairings, and both planners everything."""
    _eval(root, f"vla_choice_iid{seed}", vla0, chunk="eps_0_48", seed=seed, **kwargs)
    _eval(root, f"vlapl_choice_iid{seed}", vlapl, chunk="eps_0_48", seed=seed, **kwargs)
    _eval(root, f"pseed31_choice_iid{seed}_none", rule, seed=seed, **kwargs)
    _eval(root, f"vp_choice_iid{seed}_none", marks, seed=seed, **kwargs)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["score_instruction"])
    return tmp_path


def _lines(capsys):
    si.main()
    return capsys.readouterr().out.splitlines()


def _line(lines, start):
    (line,) = [x for x in lines if x.startswith(start)]
    return line


def test_run_names_and_design():
    names = {c: name.format("iid19") for c, name in si.CONTROLLERS.items()}
    assert names == {
        "VLA-0": "vla_choice_iid19",
        "VLA-PL": "vlapl_choice_iid19",
        "planner (fixed rule)": "pseed31_choice_iid19_none",
        "planner (VLM marks)": "vp_choice_iid19_none",
    }
    assert si.EPISODES == 48 and si.TASKS == list(cs.TASKS)
    assert [name for name, *_ in si.PLANNED] == ["I1", "I2", "I3"]
    text = README.read_text(encoding="utf-8")
    block = text[text.index("## Evals (GPU)") : text.index("## Result files (CPU)")]
    for name in names.values():  # the README's commands write the folders that the scorer reads
        assert f" {name} " in block, name


def test_a_tied_vla_and_a_planner_that_follows(runs_dir, capsys):
    _study(runs_dir)
    lines = _lines(capsys)
    assert lines[0].startswith("# Instruction study: the choice scenes on iid19, episodes 0 to 47 of each task")
    assert not any(x.startswith("Evals that are not there") for x in lines)
    assert "| VLA-PL | 100.0 (48) | 100.0 (48) | 0.0 (48) | 0.0 (48) | 100.0 (96) | 0.0 (96) | 50.0 (192) |" in lines
    assert _line(lines, "| planner (VLM marks) | 100.0 (48)").endswith("| 100.0 (96) | 100.0 (96) | 100.0 (192) |")
    whole = "+100.0 points (95% interval +100.0 to +100.0; +2.00 per episode index; 48 episode indices: 48 up, 0 down)"
    for name in ("I1", "I2", "I3"):  # 48 indices, all up: p = 2 / 2^48
        assert _line(lines, f"- {name}, ").endswith(f": {whole}, exact p = 7.1e-15: shown")
    # the following score: a controller that does the same thing whatever the instruction says has 0
    assert "- VLA-PL: 0.0% (0 of 96 pairs)" in lines and "- VLA-0: 0.0% (0 of 96 pairs)" in lines
    assert "- planner (fixed rule): 100.0% (96 of 96 pairs)" in lines
    assert "- planner (VLM marks): 100.0% (96 of 96 pairs)" in lines
    # on the new pairings VLA-PL's object always ends on its trained target, the others' never
    assert "| VLA-PL | 100.0 (48) | 100.0 (48) | 100.0 (96) |" in lines
    assert "| VLA-0 | 0.0 (48) | 0.0 (48) | 0.0 (96) |" in lines
    assert "| planner (VLM marks) | 0.0 (48) | 0.0 (48) | 0.0 (96) |" in lines
    same = "+0.0 points (95% interval +0.0 to +0.0; +0.00 per episode index; 48 episode indices: 0 up, 0 down)"
    for label in ("all four tasks", "the trained pairings", "the new pairings"):
        assert f"- the planner (VLM marks) minus the planner (fixed rule), {label}: {same}, exact p = 1" in lines
    assert _line(lines, "- VLA-0's drop").endswith(f": {same}, exact p = 1")  # nothing solved: no drop
    assert not any(x.startswith("note:") for x in lines)


def test_a_vla_that_follows_is_not_tied(runs_dir, capsys):
    _study(runs_dir, vlapl=follows)  # VLA-PL solves everything: its drop is 0
    lines = _lines(capsys)
    none = "+0.0 points (95% interval +0.0 to +0.0; +0.00 per episode index; 48 episode indices: 0 up, 0 down)"
    assert _line(lines, "- I1, ").endswith(f": {none}, exact p = 1: not shown (not significant)")
    assert _line(lines, "- I2, ").endswith(f": {none}, exact p = 1: not shown (not significant)")
    assert _line(lines, "- I3, ").endswith(f": {none}, exact p = 1: not shown (not significant)")
    assert "- VLA-PL: 100.0% (96 of 96 pairs)" in lines


def test_a_difference_in_the_other_direction_is_said_plainly(runs_dir, capsys):
    # VLA-PL solves only the new pairings, and the planner reading the VLM's marks only the trained ones
    _study(runs_dir, vlapl=lambda t, e: GOAL if t in cs.NEW else NEITHER, marks=habit)
    lines = _lines(capsys)
    down = "-100.0 points (95% interval -100.0 to -100.0; -2.00 per episode index; 48 episode indices: 0 up, 48 down)"
    for name in ("I1", "I2", "I3"):
        assert _line(lines, f"- {name}, ").endswith(
            f": {down}, exact p = 7.1e-15: not shown: significant in the other direction"
        )


def test_holm_judges_the_three_tests_together(runs_dir, capsys):
    # I1: VLA-PL fails a new pairing in 6 episode indices, p = 2 / 2^6 = 0.031: below 0.05, but not below 0.05 / 3
    _study(runs_dir, vla0=first_fail(6), vlapl=first_fail(6), marks=first_fail(6))
    lines = _lines(capsys)
    small = "+6.2 points (95% interval +2.1 to +11.5; +0.12 per episode index; 48 episode indices: 6 up, 0 down)"
    assert _line(lines, "- I1, ").endswith(f": {small}, exact p = 0.031: not shown (not significant)")
    shutil.rmtree(runs_dir / "vlapl_choice_iid19")
    shutil.rmtree(runs_dir / "vp_choice_iid19_none")
    # I1 in 8 indices (p = 0.0078, below 0.05 / 3), then I3 in 6 (p = 0.031, not below 0.05 / 2): only I1 is shown
    _eval(runs_dir, "vlapl_choice_iid19", first_fail(8), chunk="eps_0_48")
    _eval(runs_dir, "vp_choice_iid19_none", first_fail(2))
    lines = _lines(capsys)
    assert _line(lines, "- I1, ").endswith("48 episode indices: 8 up, 0 down), exact p = 0.0078: shown")
    assert _line(lines, "- I3, ").endswith(
        "48 episode indices: 6 up, 0 down), exact p = 0.031: not shown (not significant)"
    )


def test_missing_evals(runs_dir, capsys):
    # I1 in 7 indices (p = 0.016, below 0.05 / 3) and I3 in 6 (p = 0.031); I2's other eval is not there
    _study(runs_dir, vlapl=first_fail(7), marks=first_fail(1))
    shutil.rmtree(runs_dir / "vla_choice_iid19")
    lines = _lines(capsys)
    assert "Evals that are not there: vla_choice_iid19" in lines
    assert not any(x.startswith("| VLA-0 |") or x.startswith("- VLA-0:") for x in lines)
    assert _line(lines, "- I1, ").endswith("exact p = 0.016: shown")
    assert _line(lines, "- I2, ").endswith(": not run (it needs vla_choice_iid19); it counts as p = 1 in Holm's method")
    # I2 stays in the family, so I3's bar is 0.05 / 2, not 0.05
    assert _line(lines, "- I3, ").endswith("exact p = 0.031: not shown (not significant)")
    assert _line(lines, "- VLA-0's drop").endswith(": not run (it needs vla_choice_iid19)")
    assert "- VLA-PL: 92.7% (89 of 96 pairs)" in lines  # the spoon's two tasks in 41 indices, the carrot's in 48

    for name in si.CONTROLLERS.values():  # no eval at all: the scorer still runs
        shutil.rmtree(runs_dir / name.format("iid19"), ignore_errors=True)
    lines = _lines(capsys)
    assert _line(lines, "Evals that are not there: ").count("choice_iid19") == 4
    counted = "; it counts as p = 1 in Holm's method"
    assert _line(lines, "- I1, ").endswith(f": not run (it needs vlapl_choice_iid19){counted}")
    assert _line(lines, "- I3, ").endswith(f": not run (it needs vp_choice_iid19_none, vlapl_choice_iid19){counted}")


def test_units_are_the_episode_indices_that_every_eval_has_in_full(runs_dir, capsys):
    # VLA-PL did not run one task of index 3; the marks' eval has only the first 40 episodes of each task
    _study(runs_dir)
    shutil.rmtree(runs_dir / "vlapl_choice_iid19")
    shutil.rmtree(runs_dir / "vp_choice_iid19_none")
    _eval(runs_dir, "vlapl_choice_iid19", habit, chunk="eps_0_48", skip={(SPOON_NEW, 3)})
    _eval(runs_dir, "vp_choice_iid19_none", follows, episodes=40)
    study = si.Study("iid19")
    vlapl, marks, rule = (study.arms[c] for c in (si.VLAPL, si.MARKS, si.RULE))
    assert si.units([rule]) == list(range(48))
    assert si.units([vlapl]) == [e for e in range(48) if e != 3]
    assert si.units([marks, vlapl]) == [e for e in range(40) if e != 3]
    lines = _lines(capsys)
    assert "47 episode indices: 47 up, 0 down" in _line(lines, "- I1, ")
    assert "39 episode indices: 39 up, 0 down" in _line(lines, "- I3, ")
    # the success rates take every episode of the design that the eval ran
    assert "| VLA-PL | 100.0 (48) | 100.0 (48) | 0.0 (47) | 0.0 (48) | 100.0 (96) | 0.0 (95) | 50.3 (191) |" in lines
    assert "- VLA-PL: 0.0% (0 of 95 pairs)" in lines


def test_the_values_of_a_unit(runs_dir):
    # VLA-PL: index 0 all four tasks, index 1 only the trained ones, index 2 only the new ones, then nothing
    def ends(t, e):
        return GOAL if e == 0 or (e == 1 and t in cs.TRAINED) or (e == 2 and t in cs.NEW) else OTHER

    _study(runs_dir, vlapl=ends, episodes=4)
    arm = si.Study("iid19").arms[si.VLAPL]
    assert [si.drop(arm, e) for e in range(4)] == [0, 2, -2, 0]
    assert [si.successes(arm, cs.NEW, e) for e in range(4)] == [2, 0, 2, 0]
    assert si.following(arm) == (2, 8)  # both objects at index 0, none after it
    assert si.on_trained_target(arm, SPOON_NEW) == (2, 4, 0) == si.on_trained_target(arm, CARROT_NEW)
    assert [si.logged_name(t) for t in cs.NEW] == ["spoon_on_towel", "carrot_on_plate"]
    # the design's episodes only: a made-up design of 2 episodes leaves the others out
    with pytest.MonkeyPatch.context() as m:
        m.setattr(si, "EPISODES", 2)
        assert sorted(e for _, e in si.Study("iid19").arms[si.VLAPL]) == [0] * 4 + [1] * 4


def test_the_planned_tests_values():
    # one unit: VLA-PL solves the trained pairings and one new one, VLA-0 one trained one, the marks' planner all
    def arm(solved):
        return {(t, 0): {"success": t in solved} for t in cs.TASKS}

    vlapl, vla0, marks = arm([*cs.TRAINED, SPOON_NEW]), arm([cs.TRAINED[0]]), arm(cs.TASKS)
    values = {name: value for name, *_, value in si.PLANNED}
    assert values["I1"]([vlapl], 0) == 1  # 2 trained - 1 new
    assert values["I2"]([vlapl, vla0], 0) == 0  # 1 - (1 trained - 0 new)
    assert values["I3"]([marks, vlapl], 0) == 1  # 2 new - 1 new
    assert [c for *_, c, _ in si.PLANNED] == [(si.VLAPL,), (si.VLAPL, si.VLA0), (si.MARKS, si.VLAPL)]
    assert si.verdict(True, 0.5) == "shown" and si.verdict(False, 0.5) == "not shown (not significant)"
    assert si.verdict(True, -0.5) == "not shown: significant in the other direction"
    r = si.paired([2, 2, 0, -1], 2)  # mean 0.75 per index: 37.5 points of the two episodes on each side
    assert r.line.startswith("+37.5 points (95% interval ") and "4 episode indices: 2 up, 1 down" in r.line
    assert r.p == 0.5 and r.mean == 0.75  # sums of +-2, +-2, +-1 at least 3 in size: 4 of 8


def test_records_that_do_not_fit_stop_the_load(runs_dir):
    _study(runs_dir, seed=98, episodes=4)  # the smoke tests' layout set
    assert len(si.Study("iid98").arms[si.VLAPL]) == 16
    _eval(runs_dir, "vla_choice_iid19", never, seed=98, episodes=4)  # in a folder of the study's layout set
    with pytest.raises(SystemExit, match="is not of the layout set iid19"):
        si.Study("iid19")
    _eval(runs_dir, "vla_choice_paper", never, seed=98, episodes=4)
    with pytest.raises(SystemExit, match="--layouts iidN"):
        si.Study("paper")
    # the same episode in two chunks of a task, and in a folder of another task
    _eval(runs_dir, "vlapl_choice_iid98", habit, episodes=2, seed=98, chunk="eps_0_2")
    with pytest.raises(SystemExit, match="a second record for ChoiceCarrotOnPlate-v0 episode 0"):
        si.Study("iid98")
    shutil.rmtree(runs_dir / "vlapl_choice_iid98")
    _eval(runs_dir, "vlapl_choice_iid98", habit, episodes=2, seed=98)
    again = runs_dir / "vlapl_choice_iid98" / "again"
    shutil.copytree(runs_dir / "vlapl_choice_iid98" / SPOON_NEW, again)
    with pytest.raises(SystemExit, match="vlapl_choice_iid98: a second record for ChoiceSpoonOnPlate-v0 episode 0"):
        si.Study("iid98")


def test_records_without_the_stats_and_records_that_disagree(runs_dir, capsys):
    _study(runs_dir, stats=False)  # as an eval from before the evaluator wrote the stats
    lines = _lines(capsys)
    assert _line(lines, "- I1, ").endswith("exact p = 7.1e-15: shown")  # the tests need the outcomes only
    assert "| VLA-PL | - | - | - |" in lines
    assert _line(lines, "note: vlapl_choice_iid19: 96 records of the new pairings have no logged pair")
    # a record whose success is not its goal's stat
    path = runs_dir / "vp_choice_iid19_none" / SPOON_NEW / "episode_costs.jsonl"
    shutil.rmtree(runs_dir / "vp_choice_iid19_none")
    _eval(runs_dir, "vp_choice_iid19_none", follows)
    records = [json.loads(x) for x in path.read_text().splitlines()]
    records[5]["episode_stats"]["spoon_on_plate"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    lines = _lines(capsys)
    assert _line(lines, "note: vp_choice_iid19_none: ").endswith(
        "the success is not the goal's episode stat in 1 records (ChoiceSpoonOnPlate-v0 episode 5)"
    )
