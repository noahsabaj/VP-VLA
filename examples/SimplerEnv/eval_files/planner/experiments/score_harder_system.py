"""The harder-task system study (README.md, "Harder-task system study", has the plan that this scorer follows).

The VLM-plan study's system on harder tasks, with harsh depth: SimplerEnv's 4 tasks with their own instructions and
with the 4 reworded sets (instruction_sets.py), and the 5 harder scenes (harder_scenes.py). One ASM with a task
column for each of the 9 tasks, trained on the VLM-plan study's training set and the harder scenes' fixed layouts
together (selector.train_pooled), with P(planner)'s features chosen from the VLM-plan study's CANDIDATES; a reworded
instruction keeps its task's column. Each choice is scored offline, as in the VLM-plan study.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_harder_system [--train iid11] [--test iid13]

Runs, the planner-only ones with harsh depth: training vlmplan_{train}_harsh with vla31_{train}, and
vlmplan_harder_paper_harsh with vla_harder_paper; test vlmplan_{group}_{test}_harsh (group: own, words_SET, harder)
with the plan-check v2 study's VLA-only evals vla_{group}_{test}. Secondary: the fixed rule's
pseed31_{group}_{test}_harsh, and its ASM trained on pseed31_{train}_harsh (SimplerEnv's 4 tasks only). In the test
set an episode index is one layout in every instruction set, so the tests and intervals take one value per episode
index: its sum over every group and task.
"""

import argparse
import os
from collections import Counter
from typing import Dict, List

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs

from .. import selector as s
from . import mechanism, runs
from .runs import SHORT
from .score_plan_check_v2 import GROUPS, group_tasks, vla_run
from .score_two_model import LayoutSet, print_models
from .score_vlm_plan import (ASM, CANDIDATES, GATE, KNN, PER_TASK, RULE_A, RULE_ASM, RULE_C, RULE_PLANNER, VLA, VLA_RUNS,
                             VLM_PLANNER, asm_choice, baselines, ungated, verdict)
from .stats import cluster_bootstrap_interval, sign_flip, unit_sums

LEVEL = "harsh"
TASK_COLUMNS = [*s.TASKS, *hs.TASKS]
NAMES = {**SHORT, **hs.SHORT}
HARDER_TRAIN = ("vlmplan_harder_paper_harsh", "vla_harder_paper")  # the harder scenes' fixed layouts
SEQUENCE = (VLM_PLANNER, VLA, PER_TASK)  # planned tests 1 to 3, in this order
SECONDARY = (RULE_A, RULE_C, KNN)
GROUP_NAMES = {"own": "Own", "harder": "Harder scenes"}


def group_run(prefix: str, group: str, layouts: str) -> str:
    """A planner-only eval of one instruction group: {prefix}_{own | words_SET | harder}_{layouts}_harsh."""
    return f"{prefix}_{group if group in ('own', 'harder') else 'words_' + group}_{layouts}_{LEVEL}"


def concat(parts: List[LayoutSet]) -> LayoutSet:
    return LayoutSet([k for p in parts for k in p.keys], [r for p in parts for r in p.records],
                     *[np.concatenate([getattr(p, f) for p in parts]) for f in ("tasks", "y_plan", "y_vla")])


def load_part(label: str, planner_root: str, vla_roots: List[str], tasks: List[str]) -> LayoutSet:
    """The episodes both evals ran, keyed (label, task, episode)."""
    planner, vla, keys = s.matched_layouts(planner_root, vla_roots, tasks)
    records = [planner[k] for k in keys]
    return LayoutSet([(label, *k) for k in keys], records, np.array([k[0] for k in keys]),
                     np.array([float(r["success"]) for r in records]), np.array([vla[k] for k in keys]))


def training_parts(train: str) -> List[s.Part]:
    return [s.Part(runs.path(f"vlmplan_{train}_{LEVEL}"), runs.paths([VLA_RUNS.format(set=train)])),
            s.Part(runs.path(HARDER_TRAIN[0]), runs.paths([HARDER_TRAIN[1]]), list(hs.TASKS))]


def load_test(prefix: str, test: str) -> Dict[str, LayoutSet]:
    """Each group's episodes that both its planner-only eval and its VLA-only eval ran; a group without its
    planner-only eval is left out."""
    out = {}
    for g in GROUPS:
        root = runs.path(group_run(prefix, g, test))
        if os.path.isdir(root):
            out[g] = load_part(g, root, runs.paths([vla_run(g, test)]), group_tasks(g))
    return out


def units(ls: LayoutSet) -> np.ndarray:
    return np.array([k[-1] for k in ls.keys])


def paired(new: np.ndarray, base: np.ndarray, u: np.ndarray) -> tuple:
    """new - base with one value per episode index (its sum): (a line with the mean difference per episode, its 95%
    bootstrap interval over the indices and the indices up and down; the exact sign-flip test's p; the mean)."""
    d = new - base
    per = unit_sums(d, u)
    lo, hi = cluster_bootstrap_interval(d, u)
    p = sign_flip(per, 1)
    return (f"{100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
            f"{int((per > 0).sum())} episode indices up / {int((per < 0).sum())} down, exact p = {p:.2g}",
            p, float(d.mean()))


def print_rows(rows: Dict[str, np.ndarray], picks: Dict[str, int], ls: LayoutSet) -> None:
    groups = [g for g in GROUPS if any(k[0] == g for k in ls.keys)]
    of = np.array([k[0] for k in ls.keys])
    print("| Setup | " + " | ".join(GROUP_NAMES.get(g, g) for g in groups) + " | All | 95% interval | VLA picks |")
    print("|---" * (len(groups) + 4) + "|")
    for name, v in rows.items():
        lo, hi = cluster_bootstrap_interval(v, units(ls))
        print(f"| {name} | " + " | ".join(f"{100 * v[of == g].mean():.1f}" for g in groups)
              + f" | {100 * v.mean():.1f} | {100 * lo:.1f} to {100 * hi:.1f} | {picks.get(name, '')} |")


def print_tasks(rows: Dict[str, np.ndarray], ls: LayoutSet) -> None:
    """The main setups per task, every instruction group of the task together (secondary)."""
    tasks = [t for t in TASK_COLUMNS if (ls.tasks == t).any()]
    print("\nPer task (secondary; SimplerEnv's tasks with all their instruction groups):")
    print("| Setup | " + " | ".join(NAMES[t] for t in tasks) + " |\n" + "|---" * (len(tasks) + 1) + "|")
    for name in (VLM_PLANNER, VLA, ASM, "oracle"):
        print(f"| {name} | " + " | ".join(f"{100 * rows[name][ls.tasks == t].mean():.1f}" for t in tasks) + " |")


def print_tests(rows: Dict[str, np.ndarray], ls: LayoutSet, complete: bool) -> None:
    """The planned tests in their fixed order (a test counts only when every test before it was a win), then the
    secondary baselines."""
    print("Planned tests (each two-sided; one value per episode index):")
    if not complete:
        print("not run: they need every instruction group")
    counts, u = complete, units(ls)
    for i, base in enumerate(SEQUENCE, 1):
        line, p, mean = paired(rows[ASM], rows[base], u)
        print(f"{i}. vs {base}: {line}" + (verdict(p, mean, counts) if complete else ""))
        counts = counts and p < 0.05 and mean > 0
    for base in SECONDARY:
        print(f"- vs {base} (secondary): {paired(rows[ASM], rows[base], u)[0]}")


def print_plans(ls: LayoutSet) -> None:
    """What the VLM planned on each first frame, per task (all instruction groups of the task together)."""
    print("\nThe VLM plans:\n| Task | Episodes | Pick-and-place plans | With more than one move | "
          "First move (most common) |\n|---|---|---|---|---|")
    for t in [t for t in TASK_COLUMNS if (ls.tasks == t).any()]:
        rs = [r for r, k in zip(ls.records, ls.tasks, strict=True) if k == t]
        ok = [r for r in rs if (r.get("vlm_plan") or {}).get("feasible")]
        moves = Counter(f"{r['vlm_plan']['steps'][0]['object']} -> {r['vlm_plan']['steps'][0]['target']}" for r in ok)
        print(f"| {NAMES[t]} | {len(rs)} | {len(ok)} | {sum(len(r['vlm_plan']['steps']) > 1 for r in ok)} | "
              + ", ".join(f"{m} ({n})" for m, n in moves.most_common(2)) + " |")


def print_rule(test: LayoutSet, rows: Dict[str, np.ndarray], train: str, name: str) -> None:
    """The fixed rule against the VLM plan (secondary), on the episodes that both planner-only evals ran: rule ->
    planner in every group, and the fixed rule's ASM (trained on pseed31_{train}_harsh, so SimplerEnv's tasks only)."""
    rule = load_test("pseed31", name)
    root = runs.path(f"pseed31_{train}_{LEVEL}")
    if not rule or not os.path.isdir(root):
        print(f"\nnote: the fixed-rule rows are left out: no pseed31_{train}_{LEVEL} or pseed31_*_{name}_{LEVEL}")
        return
    rule = concat(list(rule.values()))
    pos, both = {k: i for i, k in enumerate(rule.keys)}, set(rule.keys)
    idx = np.array([i for i, k in enumerate(test.keys) if k in both], int)
    if not len(idx):
        print("\nnote: no episode in both the VLM plan's and the fixed rule's planner-only evals")
        return
    keys = [test.keys[i] for i in idx]
    r = np.array([pos[k] for k in keys], int)
    y_rule, y_vla, tasks = rule.y_plan[r], rule.y_vla[r], rule.tasks[r]
    print(f"\n### The fixed rule (secondary): pseed31_*_{name}_{LEVEL}, {len(keys)} episodes")
    u = np.array([k[-1] for k in keys])
    print(f"- planner-only, the VLM plan minus the fixed rule: {paired(rows[VLM_PLANNER][idx], y_rule, u)[0]} "
          f"(the fixed rule {100 * y_rule.mean():.1f}, the VLM plan {100 * rows[VLM_PLANNER][idx].mean():.1f})")
    base = np.isin(tasks, s.TASKS)
    if base.any():
        asm = s.train_two_model(root, runs.paths([VLA_RUNS.format(set=train)]))
        p_plan, p_vla = asm.probs([rule.records[i] for i in r[base]], tasks[base])
        v = np.where(p_plan >= p_vla, y_rule[base], y_vla[base])
        print(f"- {RULE_ASM} (SimplerEnv's tasks, {int(base.sum())} episodes): {100 * v.mean():.1f}; "
              f"{ASM} minus it: {paired(rows[ASM][idx][base], v, u[base])[0]}")
    print(f"- {RULE_PLANNER} per group: " + ", ".join(
        f"{GROUP_NAMES.get(g, g)} {100 * y_rule[[k[0] == g for k in keys]].mean():.1f}"
        for g in GROUPS if any(k[0] == g for k in keys)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="iid11", help="the training set of SimplerEnv's tasks")
    ap.add_argument("--test", default="iid13", help="the test set of the planned tests")
    a = ap.parse_args()
    print(f"# Harder-task system study: trained on vlmplan_{a.train}_{LEVEL} and {HARDER_TRAIN[0]}, tested on "
          f"{a.test} ({LEVEL} depth)")
    parts = training_parts(a.train)
    asm = s.train_pooled(parts, CANDIDATES, task_columns=TASK_COLUMNS)
    print("\n".join(s.describe(asm)))
    train = concat([load_part(os.path.basename(p.planner_root), p.planner_root, p.vla_roots, p.tasks or s.TASKS)
                    for p in parts])
    groups = load_test("vlmplan", a.test)
    missing = [group_run("vlmplan", g, a.test) for g in GROUPS if g not in groups]
    if missing:
        print(f"note: no {', '.join(missing)}")
    if not groups:
        return
    test = concat(list(groups.values()))
    use, gated, p_plan, p_vla = asm_choice(asm, test)
    rows = {VLM_PLANNER: test.y_plan, VLA: test.y_vla, GATE: np.where(gated, test.y_vla, test.y_plan),
            ASM: np.where(use, test.y_plan, test.y_vla), "oracle": np.maximum(test.y_plan, test.y_vla)}
    picks = {VLM_PLANNER: 0, VLA: len(test.keys), GATE: int(gated.sum()), ASM: int((~use).sum())}
    extra, extra_picks = baselines(asm, train, test, use, gated, p_plan)
    rows.update(extra)
    picks.update(extra_picks)
    print(f"\n{len(test.keys)} episodes, {len(set(units(test).tolist()))} episode indices:")
    print_rows(rows, picks, test)
    print_tests(rows, test, complete=not missing)
    print_tasks(rows, test)
    print(f"\nThe ASM decided {int((~gated).sum())} episodes (the gate gave {int(gated.sum())} to the VLA).")
    print_models({"P(planner)": (p_plan[~gated], test.y_plan[~gated]), "P(VLA)": (p_vla[~gated], test.y_vla[~gated])},
                 test.tasks[~gated], bins=True)
    mechanism.report(ungated(train), test, asm, use, gated, units(test), planner_name="VLM -> planner")
    print_plans(test)
    print_rule(test, rows, a.train, a.test)


if __name__ == "__main__":
    main()
