"""The missing-depth study (README.md, "Missing-depth study", has the plan that this scorer follows).

Clean depth, except that in each episode, with chance 1/2, the place target gives no depth, as glass or black cloth
would (planner/geometry.py MissingDepth; not in the eggplant task, whose basket is part of the sink). The planner
still finds the object and plans its first move; it fails only after the grasp, when it cannot find the target. The
system is the VLM-plan study's: the VLM plan's gate, then a two-model ASM trained on the training set's missing-depth
eval, with P(planner)'s features chosen there from CANDIDATES. Each choice is scored offline, as in the VLM-plan
study.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_missing_depth [--train iid11] [--test iid15]

Runs: planner-only miss_{set}_none (the VLM plan, with missing target depth) and VLA-only vla31_{set} (the VLA does
not use depth). Secondary: vlmplan_{test}_none (the same episodes with the target's depth) and vlmplan_{train}_none
(for the VLM-plan study's clean ASM, which never saw missing depth). The tests take one value per episode index, and
only the indices with all 4 tasks count.
"""

import argparse
import os
from collections import Counter
from typing import Dict

import numpy as np

from .. import selector as s
from . import mechanism, routers, runs
from .score_two_model import LayoutSet, load, print_models
from .score_vlm_plan import (ASM, GATE, VLA, VLA_RUNS, VLM_PLANNER, asm_choice, paired, print_rows, subset, ungated,
                             verdict, whole_indices)
from .score_vlm_plan import CANDIDATES as VLM_PLAN_CANDIDATES

# P(planner)'s candidate features: the VLM-plan study's, and the two-model study's with the share of each mask's
# pixels that have depth. The first one wins a tie.
CANDIDATES = [*VLM_PLAN_CANDIDATES, [*s.PLANNER_FEATURES, "grasp_valid_depth_frac", "place_valid_depth_frac"]]
PREFIX = "miss"  # the planner-only evals: miss_{set}_none
LEVEL = "none"
FALLBACK = "planner first: the gate, then the planner when it found a first plan, else the VLA"
CLEAN_ASM = "the VLM-plan study's clean ASM (trained without missing depth)"
PER_TASK = "per-task choice: the gate, then each task's better method on the training set"
RANDOM = "random: the gate, then a coin with the ASM's planner share (expected)"
SEQUENCE = (VLM_PLANNER, VLA)  # planned tests 1 and 2, in this order
SECONDARY = (FALLBACK, CLEAN_ASM, PER_TASK)


def missing(ls: LayoutSet) -> np.ndarray:
    """Which episodes' place target gave no depth (the records' target_depth_missing)."""
    return np.array([bool(r.get("target_depth_missing")) for r in ls.records])


def print_tests(rows: Dict[str, np.ndarray], ls: LayoutSet) -> None:
    """The planned tests in their fixed order (a test counts only when every test before it was a win), then the
    secondary baselines."""
    print("Planned tests (each two-sided; one value per episode index):")
    counts = True
    for i, base in enumerate(SEQUENCE, 1):
        line, p, mean = paired(rows[ASM], rows[base], ls.keys)
        print(f"{i}. vs {base}: {line}{verdict(p, mean, counts)}")
        counts = counts and p < 0.05 and mean > 0
    for base in SECONDARY:
        if base in rows:
            print(f"- vs {base} (secondary): {paired(rows[ASM], rows[base], ls.keys)[0]}")


def print_split(rows: Dict[str, np.ndarray], picks: np.ndarray, ls: LayoutSet) -> None:
    """Each setup on the episodes with and without the target's depth (secondary)."""
    m = missing(ls)
    names = [VLM_PLANNER, VLA, ASM, FALLBACK, CLEAN_ASM, "oracle"]
    names = [n for n in names if n in rows]
    print("\nWith and without the target's depth (secondary; the eggplant task always has it):")
    print("| Episodes | Count | " + " | ".join(names) + " | ASM's VLA picks |\n" + "|---" * (len(names) + 3) + "|")
    for label, sel in (("target without depth", m), ("target with depth", ~m)):
        if sel.any():
            print(f"| {label} | {int(sel.sum())} | " + " | ".join(f"{100 * rows[n][sel].mean():.1f}" for n in names)
                  + f" | {int(picks[sel].sum())} |")
    print("Why VLM -> planner failed:\n| Episodes | Failure | Count |\n|---|---|---|")
    for label, sel in (("target without depth", m), ("target with depth", ~m)):
        failed = Counter(r.get("failure") or "no failure logged" for r, ok, keep in
                         zip(ls.records, ls.y_plan, sel, strict=True) if keep and not ok)
        for why, n in failed.most_common():
            print(f"| {label} | {why} | {n} |")


def print_with_depth(rows: Dict[str, np.ndarray], ls: LayoutSet, name: str) -> None:
    """VLM -> planner with missing target depth against the same episodes with the target's depth (secondary)."""
    run = f"vlmplan_{name}_{LEVEL}"
    if not os.path.isdir(runs.path(run)):
        print(f"\nnote: no {run}: the cost of the missing depth is left out")
        return
    with_depth = whole_indices(load("vlmplan", name, LEVEL, VLA_RUNS), run)
    both = set(with_depth.keys)
    keys = [k for k in ls.keys if k in both]
    keys = [k for k in keys if sum((t, k[1]) in both for t in s.TASKS) == len(s.TASKS)]
    if not keys:
        print(f"\nnote: no episode index with all 4 tasks in both {run} and the missing-depth eval")
        return
    pos = {k: i for i, k in enumerate(ls.keys)}
    idx = np.array([pos[k] for k in keys], int)
    v = subset(with_depth, keys).y_plan
    print(f"\nThe cost of the missing depth to VLM -> planner (secondary): {run} on the same {len(keys)} episodes, "
          f"{100 * v.mean():.1f}; with missing target depth minus with the target's depth: "
          f"{paired(rows[VLM_PLANNER][idx], v, keys)[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="iid11", help="the training set")
    ap.add_argument("--test", default="iid15", help="the test set of the planned tests")
    a = ap.parse_args()
    vla_train = runs.paths([VLA_RUNS.format(set=a.train)])
    print(f"# Missing-depth study: trained on {PREFIX}_{a.train}_{LEVEL}, tested on {PREFIX}_{a.test}_{LEVEL}")
    asm = s.train_two_model(runs.path(f"{PREFIX}_{a.train}_{LEVEL}"), vla_train, CANDIDATES)
    print("\n".join(s.describe(asm)))
    train = load(PREFIX, a.train, LEVEL, VLA_RUNS)
    test = whole_indices(load(PREFIX, a.test, LEVEL, VLA_RUNS), f"{PREFIX}_{a.test}_{LEVEL}")
    use, gated, p_plan, p_vla = asm_choice(asm, test)
    rows = {VLM_PLANNER: test.y_plan, VLA: test.y_vla, GATE: np.where(gated, test.y_vla, test.y_plan),
            ASM: np.where(use, test.y_plan, test.y_vla), "oracle": np.maximum(test.y_plan, test.y_vla)}
    picks = {VLM_PLANNER: 0, VLA: len(test.keys), GATE: int(gated.sum()), ASM: int((~use).sum())}
    plan_ok = np.array([s.row_features(r, ["plan_ok"])[0] > 0 for r in test.records])
    choices = {FALLBACK: ~gated & plan_ok}
    clean_root = runs.path(f"vlmplan_{a.train}_{LEVEL}")
    if os.path.isdir(clean_root):
        choices[CLEAN_ASM] = asm_choice(s.train_two_model(clean_root, vla_train, VLM_PLAN_CANDIDATES), test)[0]
    else:
        print(f"note: no vlmplan_{a.train}_{LEVEL}: the clean ASM's row is left out")
    decided = ungated(train)
    choices[PER_TASK] = ~gated & routers.per_task_choice(decided.tasks, decided.y_plan, decided.y_vla, test.tasks)
    for name, u in choices.items():
        rows[name], picks[name] = np.where(u, test.y_plan, test.y_vla), int((~u).sum())
    share = use.sum() / max((~gated).sum(), 1)
    rows[RANDOM] = np.where(gated, test.y_vla, routers.expected_random(share, test.y_plan, test.y_vla))
    print(f"\n{len(test.keys)} episodes, {len(test.keys) // len(s.TASKS)} episode indices; the place target gave no "
          f"depth in {int(missing(test).sum())}:")
    print_rows(rows, picks, test)
    print_tests(rows, test)
    print_split(rows, ~use, test)
    print_with_depth(rows, test, a.test)
    print(f"\nThe ASM decided {int((~gated).sum())} episodes (the gate gave {int(gated.sum())} to the VLA).")
    print_models({"P(planner)": (p_plan[~gated], test.y_plan[~gated]), "P(VLA)": (p_vla[~gated], test.y_vla[~gated])},
                 test.tasks[~gated], bins=True)
    mechanism.report(decided, test, asm, use, gated, np.array([e for _, e in test.keys]), planner_name="VLM -> planner")


if __name__ == "__main__":
    main()
