"""The difficulty sweep (README.md, "Difficulty sweep", has the plan that this scorer follows).

The VLM-plan study's system at 7 depth-noise levels: clean depth, and harsh's errors times 0.25, 0.5, 0.75, 1
(harsh), 1.5 and 2 (planner/geometry.py). One ASM serves every level: it is trained on the training set at clean and
harsh depth together (selector.train_pooled), with P(planner)'s features chosen from the VLM-plan study's
CANDIDATES, and it is not told the level. The VLA does not use depth, so one VLA-only eval serves every level. Each
choice is scored offline, as in the VLM-plan study.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_sweep [--train iid11] [--test iid15]

Runs: planner-only vlmplan_{set}_{level} (the VLM plan) and VLA-only vla31_{set}. The planned test takes one value
per episode index (its sum over the levels and the 4 tasks), and only the indices with all 4 tasks at every level
count.
"""

import argparse
import os
from typing import Dict, List

import numpy as np

from .. import selector as s
from . import runs
from .mechanism import complementarity, failure_dependence, gain_decomposition
from .score_two_model import LayoutSet, load
from .score_vlm_plan import CANDIDATES, VLA_RUNS, asm_choice, per_index, subset, verdict, whole_indices
from .stats import bootstrap_interval, sign_flip

# (scale of harsh's errors, depth level): planner/geometry.py's sweep_level names the levels other than clean and
# harsh (a test checks that they match). Scale 0 is clean depth.
SWEEP = [(0.0, "none"), (0.25, "harsh_x0.25"), (0.5, "harsh_x0.5"), (0.75, "harsh_x0.75"), (1.0, "harsh"),
         (1.5, "harsh_x1.5"), (2.0, "harsh_x2")]
TRAIN_LEVELS = ["none", "harsh"]  # the training set's levels (the VLM-plan study's training evals)


def train(name: str) -> s.TwoModelASM:
    """The sweep's ASM: trained on the set `name` at TRAIN_LEVELS together."""
    vla = runs.paths([VLA_RUNS.format(set=name)])
    return s.train_pooled([s.Part(runs.path(f"vlmplan_{name}_{lvl}"), vla) for lvl in TRAIN_LEVELS], CANDIDATES)


def common(sets: Dict[str, LayoutSet]) -> Dict[str, LayoutSet]:
    """Each level's episodes of the episode indices that have all 4 tasks at every level."""
    whole = {lvl: whole_indices(ls, f"level {lvl}") for lvl, ls in sets.items()}
    shared = set.intersection(*[set(ls.keys) for ls in whole.values()])
    indices = sorted({e for _, e in shared if all((t, e) in shared for t in s.TASKS)})
    dropped = len({e for ls in whole.values() for _, e in ls.keys}) - len(indices)
    if dropped:
        print(f"note: {dropped} episode indices are not at every level with all 4 tasks, left out")
    keys = [(t, e) for t in s.TASKS for e in indices]
    return {lvl: subset(ls, keys) for lvl, ls in whole.items()}


def outcomes(asm: s.TwoModelASM, ls: LayoutSet) -> dict:
    """One level's outcome of each setup, and the ASM's choices."""
    use, gated, _, _ = asm_choice(asm, ls)
    plan_better = ls.y_plan.mean() >= ls.y_vla.mean()
    return {"planner": ls.y_plan, "vla": ls.y_vla, "better": ls.y_plan if plan_better else ls.y_vla,
            "better_name": "VLM -> planner" if plan_better else "VLA-only", "asm": np.where(use, ls.y_plan, ls.y_vla),
            "oracle": np.maximum(ls.y_plan, ls.y_vla), "use": use, "gated": gated}


def index_values(diffs: List[np.ndarray], keys: List[tuple]) -> np.ndarray:
    """[episode index, level and task]: the differences of each level, one row per episode index."""
    return np.hstack([per_index(d, keys) for d in diffs])


def paired_line(d: np.ndarray) -> tuple:
    """d: [episode index, episodes]. (a line with the mean difference, its 95% bootstrap interval over the indices
    and the indices up and down; the exact sign-flip test's p on the per-index sums; the mean)."""
    per = d.sum(1)
    lo, hi = bootstrap_interval(d.mean(1), np.zeros(len(d)))
    p = sign_flip(per, 1)
    return (f"{100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
            f"{int((per > 0).sum())} episode indices up / {int((per < 0).sum())} down, exact p = {p:.2g}",
            p, float(d.mean()))


def print_levels(out: Dict[str, dict], ls: Dict[str, LayoutSet]) -> None:
    print("| Depth | Scale | VLM -> planner | VLA-only | The better of the two | ASM | Oracle | Oracle's gain | "
          "ASM's gain | Share of the oracle's gain | VLA picks |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for scale, lvl in SWEEP:
        if lvl not in out:
            continue
        o = out[lvl]
        gain, top = o["oracle"].mean() - o["better"].mean(), o["asm"].mean() - o["better"].mean()
        share = f"{100 * top / gain:.0f}%" if gain > 0 else "-"
        print(f"| {lvl} | {scale:g} | {100 * o['planner'].mean():.1f} | {100 * o['vla'].mean():.1f} | "
              f"{100 * o['better'].mean():.1f} ({o['better_name']}) | {100 * o['asm'].mean():.1f} | "
              f"{100 * o['oracle'].mean():.1f} | {100 * gain:+.1f} | {100 * top:+.1f} | {share} | "
              f"{int((~o['use']).sum())} of {len(ls[lvl].keys)} |")


def print_mechanism(out: Dict[str, dict], ls: Dict[str, LayoutSet]) -> None:
    print("\nWhat makes the ASM better at each level (secondary, no tests):")
    print("| Depth | Both | Only VLM -> planner | Only the VLA | Neither | Failure dependence (95% interval) | "
          "Planner-only episodes to the planner | VLA-only episodes to the planner |\n|---|---|---|---|---|---|---|---|")
    for _, lvl in SWEEP:
        if lvl not in out:
            continue
        o, test = out[lvl], ls[lvl]
        c = complementarity(test.y_plan, test.y_vla)
        d, lo, hi = failure_dependence(test.y_plan, test.y_vla, test.tasks, np.array([e for _, e in test.keys]))
        g = gain_decomposition(o["use"], test.y_plan, test.y_vla)
        print(f"| {lvl} | {c['both']} | {c['planner only']} | {c['VLA only']} | {c['neither']} | {100 * d:+.1f} "
              f"({100 * lo:+.1f} to {100 * hi:+.1f}) | {g['planner-only to the planner']} of {g['planner only']} | "
              f"{g['VLA-only to the planner']} of {g['VLA only']} |")
    print("Failure dependence: P(VLA succeeds | VLM -> planner failed) - P(VLA succeeds | it succeeded), within tasks.")


def print_matched(asm: s.TwoModelASM, out: Dict[str, dict], ls: Dict[str, LayoutSet], name: str) -> None:
    """The pooled ASM against the VLM-plan study's ASM of each training level (trained on that level only)."""
    print("\nThe sweep's ASM against an ASM trained on one level only (the VLM-plan study's; secondary):")
    print("| Depth | The sweep's ASM | The ASM of that level | Difference |\n|---|---|---|---|")
    for lvl in TRAIN_LEVELS:
        if lvl not in out:
            continue
        one = s.train_two_model(runs.path(f"vlmplan_{name}_{lvl}"), runs.paths([VLA_RUNS.format(set=name)]),
                                CANDIDATES)
        use, _, _, _ = asm_choice(one, ls[lvl])
        v = np.where(use, ls[lvl].y_plan, ls[lvl].y_vla)
        line = paired_line(per_index(out[lvl]["asm"] - v, ls[lvl].keys))[0]
        print(f"| {lvl} | {100 * out[lvl]['asm'].mean():.1f} | {100 * v.mean():.1f} | {line} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="iid11", help="the training set")
    ap.add_argument("--test", default="iid15", help="the test set of the planned test")
    a = ap.parse_args()
    print(f"# Difficulty sweep: one ASM trained on {a.train} ({' and '.join(TRAIN_LEVELS)} depth), tested on "
          f"{a.test}")
    asm = train(a.train)
    print("\n".join(s.describe(asm)))
    levels = [lvl for _, lvl in SWEEP if os.path.isdir(runs.path(f"vlmplan_{a.test}_{lvl}"))]
    missing = [lvl for _, lvl in SWEEP if lvl not in levels]
    if missing:
        print(f"note: no eval of {a.test} at {', '.join(missing)}")
    if not levels:
        return
    ls = common({lvl: load("vlmplan", a.test, lvl, VLA_RUNS) for lvl in levels})
    keys = ls[levels[0]].keys
    out = {lvl: outcomes(asm, ls[lvl]) for lvl in levels}
    print(f"\n{len(keys)} episodes at each level, {len(keys) // len(s.TASKS)} episode indices:")
    print_levels(out, ls)
    print("\nPlanned test (two-sided; one value per episode index: its sum over the levels and the 4 tasks):")
    if missing:
        print("1. not run: it needs every level")
    else:
        line, p, mean = paired_line(index_values([out[lvl]["asm"] - out[lvl]["better"] for lvl in levels], keys))
        print(f"1. the ASM vs the better of VLM -> planner and VLA-only at each level: {line}{verdict(p, mean, True)}")
    print("\nAt each level (secondary, no claims; one value per episode index):")
    print("| Depth | ASM vs VLM -> planner | ASM vs VLA-only |\n|---|---|---|")
    for lvl in levels:
        o = out[lvl]
        print(f"| {lvl} | {paired_line(per_index(o['asm'] - o['planner'], keys))[0]} | "
              f"{paired_line(per_index(o['asm'] - o['vla'], keys))[0]} |")
    print_mechanism(out, ls)
    print_matched(asm, out, ls, a.train)


if __name__ == "__main__":
    main()
