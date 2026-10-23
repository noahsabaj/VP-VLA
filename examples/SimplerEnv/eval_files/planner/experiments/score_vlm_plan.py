"""The VLM-plan study (README.md, "VLM-plan study", has the plan that this scorer follows).

VLM -> planner (the VLM plan, planner-only) against the same system with the ASM and the VLA: a plan that is not
pick-and-place goes to the VLA (selector.vlm_plan_gate), and the two-model ASM decides the other episodes. The ASM is
trained on the training set as `selector.py --rule two-model --planner-candidates` trains it, with P(planner)'s
features chosen there from CANDIDATES. Each choice is scored offline: from the planner-only eval's first look, with
the outcome of the chosen method's eval of that episode.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_vlm_plan [--train iid11] [--test iid15]

Runs: planner-only vlmplan_{set}_{level} (the VLM plan) and VLA-only vla31_{set}, at the depth levels harsh (the
planned tests) and none (secondary); the fixed rule's planner-only pseed31_{set}_harsh (secondary). A secondary run
that is missing is left out, with a note. In the iidN sets the 4 tasks of an episode index share their layout, so
the tests and intervals take one value per episode index, and only the indices with all 4 tasks count.
"""

import argparse
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .. import selector as s
from . import runs
from .runs import LEVEL_NAMES, SHORT
from .score_two_model import LayoutSet, load, print_models
from .stats import bootstrap_interval, sign_flip

PRIMARY_LEVEL = "harsh"  # with harsh depth the planner fails most, so the choice matters most
LEVELS = [PRIMARY_LEVEL, "none"]
# P(planner)'s candidate features, fixed before any eval with the VLM plan ran; the first one wins a tie.
CANDIDATES = [s.PLANNER_FEATURES, [*s.PLANNER_FEATURES, "grasp_sam3_score", "place_sam3_score"]]
VLA_RUNS = "vla31_{set}"
VLM_PLANNER = "VLM -> planner (planner-only)"
VLA = "VLA-only"
GATE = "VLM -> planner, the VLA when the plan is not pick-and-place"
ASM = "VLM -> ASM: that gate, then the planner when P(planner) >= P(VLA)"
RULE_PLANNER = "fixed rule -> planner (planner-only)"
RULE_ASM = "fixed rule -> ASM: the planner when P(planner) >= P(VLA)"


def whole_indices(ls: LayoutSet, name: str) -> LayoutSet:
    """The episodes of the episode indices that have all 4 tasks; a note for the others."""
    tasks_of = Counter(e for _, e in ls.keys)
    keep = np.array([tasks_of[e] == len(s.TASKS) for _, e in ls.keys], bool)
    if not keep.all():
        print(f"note: {name}: {len({e for _, e in ls.keys if tasks_of[e] < len(s.TASKS)})} episode indices without "
              f"all {len(s.TASKS)} tasks, left out")
    return LayoutSet([k for k, ok in zip(ls.keys, keep, strict=True) if ok],
                     [r for r, ok in zip(ls.records, keep, strict=True) if ok],
                     ls.tasks[keep], ls.y_plan[keep], ls.y_vla[keep])


def subset(ls: LayoutSet, keys: List[tuple]) -> LayoutSet:
    pos = {k: i for i, k in enumerate(ls.keys)}
    idx = np.array([pos[k] for k in keys], int)
    return LayoutSet(list(keys), [ls.records[i] for i in idx], ls.tasks[idx], ls.y_plan[idx], ls.y_vla[idx])


def per_index(v: np.ndarray, keys: List[tuple]) -> np.ndarray:
    """v as [episode index, task]: one row per episode index (the keys must have all 4 tasks of each index)."""
    pos = {k: i for i, k in enumerate(keys)}
    return np.array([[v[pos[t, e]] for t in s.TASKS] for e in sorted({e for _, e in keys})])


def asm_choice(asm: s.TwoModelASM, ls: LayoutSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The selector mode's choice for each episode (planner_client._choose): the VLM plan's gate, then the ASM.
    (use the planner, gated, P(planner), P(VLA))."""
    gated = np.array([s.vlm_plan_gate(r) is not None for r in ls.records], bool)
    p_plan, p_vla = asm.probs(ls.records, ls.tasks)
    return ~gated & (p_plan >= p_vla), gated, p_plan, p_vla


def paired(new: np.ndarray, base: np.ndarray, keys: List[tuple]) -> Tuple[str, float, float]:
    """new - base with one value per episode index: (a line with the mean difference, its 95% bootstrap interval over
    the indices and the indices up and down; the exact paired sign-flip test's p on the per-index sums; the mean)."""
    d = per_index(new - base, keys)
    per = d.sum(1)
    lo, hi = bootstrap_interval(d.mean(1), np.zeros(len(d)))
    p = sign_flip(per, 1)
    return (f"{100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
            f"{int((per > 0).sum())} episode indices up / {int((per < 0).sum())} down, exact p = {p:.2g}",
            p, float(d.mean()))


def print_rows(rows: Dict[str, np.ndarray], picks: Dict[str, int], ls: LayoutSet) -> None:
    print("| Setup | " + " | ".join(SHORT[k] for k in s.TASKS) + " | All | 95% interval | VLA picks |")
    print("|---" * 8 + "|")
    for name, v in rows.items():
        m = per_index(v, ls.keys).mean(1)
        lo, hi = bootstrap_interval(m, np.zeros(len(m)))
        print(f"| {name} | " + " | ".join(f"{100 * v[ls.tasks == k].mean():.1f}" for k in s.TASKS)
              + f" | {100 * v.mean():.1f} | {100 * lo:.1f} to {100 * hi:.1f} | {picks.get(name, '')} |")


def print_tests(rows: Dict[str, np.ndarray], ls: LayoutSet, planned: bool) -> None:
    """The ASM against VLM -> planner, then against the VLA: in this fixed order for the planned tests, where a test
    counts only when every test before it was a win (significant, with the ASM higher)."""
    print("Planned tests (fixed order, each two-sided at 0.05; one value per episode index):" if planned
          else "Secondary tests (no claims; one value per episode index):")
    counts = True
    for i, base in enumerate((VLM_PLANNER, VLA), 1):
        line, p, mean = paired(rows[ASM], rows[base], ls.keys)
        if planned:
            win = p < 0.05 and mean > 0
            if not counts:
                line += ": does not count (a test before it was not a win)"
            else:
                line += ": a win" if win else ": significant, but the ASM is lower" if p < 0.05 else \
                    ": not significant"
            counts = counts and win
        print(f"{i}. vs {base}: {line}")
    gap = rows["oracle"].mean() - rows[VLM_PLANNER].mean()
    if gap > 0:
        print(f"The ASM gets {100 * (rows[ASM].mean() - rows[VLM_PLANNER].mean()) / gap:.0f}% of the oracle's gain "
              f"over VLM -> planner.")


def print_plans(ls: LayoutSet) -> None:
    """What the VLM planned on each first frame, per task."""
    print("| Task | Episodes | Pick-and-place plans | With more than one move | First move (most common) | "
          "Best SAM score of its names: object / target (mean) |\n|---|---|---|---|---|---|")
    for t in s.TASKS:
        rs = [r for r, k in zip(ls.records, ls.tasks, strict=True) if k == t]
        ok = [r for r in rs if (r.get("vlm_plan") or {}).get("feasible")]
        moves = Counter(f"{r['vlm_plan']['steps'][0]['object']} -> {r['vlm_plan']['steps'][0]['target']}" for r in ok)
        score = np.mean([s.row_features(r, ["grasp_sam3_score", "place_sam3_score"]) for r in ok], 0) if ok \
            else np.full(2, np.nan)
        print(f"| {SHORT[t]} | {len(rs)} | {len(ok)} | {sum(len(r['vlm_plan']['steps']) > 1 for r in ok)} | "
              + ", ".join(f"{m} ({n})" for m, n in moves.most_common(2)) + f" | {score[0]:.2f} / {score[1]:.2f} |")


def print_rule(vlm: LayoutSet, rows: Dict[str, np.ndarray], use: np.ndarray, train: str, test: str) -> None:
    """The fixed rule against the VLM plan (secondary): rule -> planner and the two-model study's ASM on the rule's
    records, trained on the training set's rule eval, on the episodes that both planner-only evals ran. use: the VLM
    plan's choices (asm_choice)."""
    names = [f"pseed31_{train}_{PRIMARY_LEVEL}", f"pseed31_{test}_{PRIMARY_LEVEL}"]
    missing = [n for n in names if not os.path.isdir(runs.path(n))]
    if missing:
        print(f"\nnote: the fixed-rule rows are left out: no {' or '.join(missing)}")
        return
    print(f"\n### The fixed rule (secondary): {names[1]}, the ASM trained on {names[0]}")
    asm = s.train_two_model(runs.path(names[0]), runs.paths([VLA_RUNS.format(set=train)]))
    rule = load("pseed31", test, PRIMARY_LEVEL, VLA_RUNS)
    both = set(rule.keys)
    keys = [k for k in vlm.keys if k in both]
    tasks_of = Counter(e for _, e in keys)
    keys = [k for k in keys if tasks_of[k[1]] == len(s.TASKS)]
    if not keys:
        print("note: no episode index with all 4 tasks in both planner-only evals")
        return
    rule, pos = subset(rule, keys), {k: i for i, k in enumerate(vlm.keys)}
    idx = np.array([pos[k] for k in keys], int)
    p_plan, p_vla = asm.probs(rule.records, rule.tasks)
    out = {VLM_PLANNER: rows[VLM_PLANNER][idx], RULE_PLANNER: rule.y_plan, ASM: rows[ASM][idx],
           RULE_ASM: np.where(p_plan >= p_vla, rule.y_plan, rule.y_vla), VLA: rule.y_vla}
    print_rows(out, {VLM_PLANNER: 0, RULE_PLANNER: 0, ASM: int((~use[idx]).sum()), RULE_ASM: int((p_plan < p_vla).sum()),
                     VLA: len(keys)}, rule)
    print("The VLM plan minus the fixed rule (no claims):")
    print(f"- planner-only: {paired(out[VLM_PLANNER], out[RULE_PLANNER], keys)[0]}")
    print(f"- with the ASM: {paired(out[ASM], out[RULE_ASM], keys)[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="iid11", help="the training set")
    ap.add_argument("--test", default="iid15", help="the test set of the planned tests")
    a = ap.parse_args()
    print(f"# VLM-plan study: trained on {a.train}, tested on {a.test}")
    for lvl in LEVELS:
        planned = lvl == PRIMARY_LEVEL
        names = [f"vlmplan_{a.train}_{lvl}", f"vlmplan_{a.test}_{lvl}"]
        missing = [n for n in names if not os.path.isdir(runs.path(n))]
        if missing and not planned:
            print(f"\nnote: {LEVEL_NAMES[lvl]} depth is left out: no {' or '.join(missing)}")
            continue
        print(f"\n## {a.test}, {LEVEL_NAMES[lvl]} depth" + (" (the planned tests)" if planned else " (secondary)"))
        asm = s.train_two_model(runs.path(names[0]), runs.paths([VLA_RUNS.format(set=a.train)]), CANDIDATES)
        print("\n".join(s.describe(asm)))
        test = whole_indices(load("vlmplan", a.test, lvl, VLA_RUNS), names[1])
        use, gated, p_plan, p_vla = asm_choice(asm, test)
        rows = {VLM_PLANNER: test.y_plan, VLA: test.y_vla, GATE: np.where(gated, test.y_vla, test.y_plan),
                ASM: np.where(use, test.y_plan, test.y_vla), "oracle": np.maximum(test.y_plan, test.y_vla)}
        picks = {VLM_PLANNER: 0, VLA: len(test.keys), GATE: int(gated.sum()), ASM: int((~use).sum())}
        print(f"{len(test.keys)} episodes, {len(test.keys) // len(s.TASKS)} episode indices:")
        print_rows(rows, picks, test)
        print_tests(rows, test, planned)
        print(f"The ASM decided {int((~gated).sum())} episodes (the gate gave {int(gated.sum())} to the VLA).")
        decided = ~gated
        print_models({"P(planner)": (p_plan[decided], test.y_plan[decided]),
                      "P(VLA)": (p_vla[decided], test.y_vla[decided])}, test.tasks[decided], bins=planned)
        if planned:
            print("\nThe VLM plans:")
            print_plans(test)
            print_rule(test, rows, use, a.train, a.test)


if __name__ == "__main__":
    main()
