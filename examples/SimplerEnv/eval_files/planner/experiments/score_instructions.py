"""The reworded-instruction study (README.md, "Reworded-instruction study", has the plan that this scorer follows).

For each instruction set of instruction_sets.py, the planner-only and the VLA-only evals of the test set (paper
layouts, clean camera) against the same evals with the tasks' own instructions, layout by layout.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_instructions [--asm two_model/asm_none.json]

Runs: planner-only {prefix}_{set}_paper and VLA-only vla_{prefix}_{set}_paper (prefix: words); with the tasks' own
instructions, pseed_paper_none and runs.FAST_VLA["paper"]. --asm adds the saved ASM's offline choice from each
planner-only eval's first look (secondary).
"""

import argparse
from typing import Dict, List

import numpy as np

from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS, TASKS

from .. import selector as s
from . import runs
from .runs import SHORT
from .stats import bootstrap_interval, mcnemar, sign_flip

DEFAULT = "default"


def load(planner: str, vla: List[str]) -> Dict[tuple, tuple]:
    """(task, episode) -> (planner record, planner success, VLA success) for the layouts both evals ran."""
    plan, vla_out, keys = s.matched_layouts(runs.path(planner), runs.paths(vla))
    return {k: (plan[k], float(plan[k]["success"]), vla_out[k]) for k in keys}


def per_layout(sets: Dict[str, Dict[tuple, tuple]], keys: List[tuple], which: int) -> np.ndarray:
    """[layouts, sets] success of the planner (which=1) or the VLA (which=2)."""
    return np.array([[sets[name][k][which] for name in sets] for k in keys])


def print_tests(plan: np.ndarray, vla: np.ndarray, tasks: np.ndarray) -> None:
    """The planned tests, in this fixed order; a test counts only when every test before it was significant. Column 0
    is the tasks' own instructions, the others the reworded sets. Each layout gives one value (its sum over the
    reworded sets), so the 4 sets of one layout are not counted as independent episodes."""
    did = (vla[:, 1:] - vla[:, :1]) - (plan[:, 1:] - plan[:, :1])
    gap = vla[:, 1:] - plan[:, 1:]
    sequence = [(did, "the planner loses more than the VLA when the instruction is reworded (the drop from the "
                      "tasks' own instructions, VLA minus planner)"),
                (gap, "the VLA beats the planner on the reworded instructions")]
    print("Planned tests (fixed order, each two-sided at 0.05; one value per layout):")
    counts = True
    for i, (d, what) in enumerate(sequence, 1):
        per = d.sum(axis=1)
        p = sign_flip(per, 1)
        lo, hi = bootstrap_interval(d.mean(axis=1), tasks)
        verdict = (": significant" if p < 0.05 else ": not significant") if counts else \
            ": does not count (a test before it was not significant)"
        counts = counts and p < 0.05
        print(f"{i}. {what}: {100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
              f"{int((per > 0).sum())} layouts up / {int((per < 0).sum())} down, exact p = {p:.2g}{verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="words", help="run-name prefix of the reworded-instruction evals")
    ap.add_argument("--asm", default=None, help="a saved ASM (selector.py --save) for the offline rows (secondary)")
    a = ap.parse_args()
    sets = {DEFAULT: load("pseed_paper_none", runs.FAST_VLA["paper"])}
    for name in INSTRUCTION_SETS:
        sets[name] = load(f"{a.prefix}_{name}_paper", [f"vla_{a.prefix}_{name}_paper"])
    keys = sorted(set.intersection(*(set(v) for v in sets.values())))
    for name, v in sets.items():
        if len(v) > len(keys):
            print(f"warning: {name}: {len(v) - len(keys)} layouts that another set lacks, left out")
    tasks = np.array([k[0] for k in keys])
    plan, vla = per_layout(sets, keys, 1), per_layout(sets, keys, 2)

    print(f"# Reworded instructions: {len(keys)} layouts of the test set, clean camera")
    print("\n| Instructions | Planner-only: " + " / ".join(SHORT[t] for t in TASKS) + " | Planner-only | VLA-only: "
          + " / ".join(SHORT[t] for t in TASKS) + " | VLA-only | Oracle |\n|---|---|---|---|---|---|")
    for j, name in enumerate(sets):
        def by_task(v):
            return " / ".join(f"{100 * v[tasks == t, j].mean():.0f}" for t in TASKS)
        print(f"| {name} | {by_task(plan)} | {100 * plan[:, j].mean():.1f} | {by_task(vla)} | "
              f"{100 * vla[:, j].mean():.1f} | {100 * np.maximum(plan[:, j], vla[:, j]).mean():.1f} |")
    print()
    print_tests(plan, vla, tasks)

    print("\nSecondary (no claims), each set on its own (exact McNemar):")
    for j, name in enumerate(list(sets)[1:], 1):
        lines = []
        for label, base, new in (("VLA vs planner", plan[:, j], vla[:, j]),
                                 ("planner vs own instructions", plan[:, 0], plan[:, j]),
                                 ("VLA vs own instructions", vla[:, 0], vla[:, j])):
            up, down, p = mcnemar(base, new)
            lines.append(f"{label} {100 * (new - base).mean():+.1f} ({up} up / {down} down, p = {p:.2g})")
        print(f"- {name}: " + "; ".join(lines))
    if a.asm:
        asm = s.load_asm(runs.path(a.asm))
        print(f"\nThe ASM {a.asm}, offline (its choice from each planner-only eval's first look):")
        print("| Instructions | ASM | VLA picks |\n|---|---|---|")
        for j, name in enumerate(sets):
            picks = [asm.decide(sets[name][k][0], k[0])["choice"] for k in keys]
            out = np.where(np.array(picks) == "planner", plan[:, j], vla[:, j])
            print(f"| {name} | {100 * out.mean():.1f} | {picks.count('vla')} |")


if __name__ == "__main__":
    main()
