"""The harder-scene study (README.md, "Harder-scene study", has the plan that this scorer follows).

The planner-only and the VLA-only evals of the harder scenes (eval_files/harder_scenes.py: their fixed layouts,
clean camera), layout by layout.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_harder_scenes

Runs: planner-only harder_paper and VLA-only vla_harder_paper. The descriptive rows of SimplerEnv's own spoon and
carrot tasks (no extra objects; other layouts): runs.OWN31_PLANNER and runs.OWN31_VLA (SAM 3.1, the plan's change of
2026-10-23).
"""

import argparse
import glob
import os
import re
from typing import Dict, List

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs

from .. import selector as s
from . import runs
from .stats import bootstrap_interval, mcnemar, sign_flip

TASKS = list(hs.TASKS)
ONE_STEP = ["ClutteredSpoonOnTowel-v0", "ClutteredCarrotOnPlate-v0"]  # the rule reads them
TWO_STEP = "TwoStepSpoonAndCarrot-v0"
RULE_BREAKING = [TWO_STEP, "CarrotOnNearPlate-v0", "CarrotOnFarPlate-v0"]  # the rule cannot read them
VIDEO = re.compile(r"/(?:success|failure)_obj_episode_(\d+)(_[^/]*)?\.mp4$")


def stat_names(task: str) -> List[str]:
    t = hs.TASKS[task]
    return [f"{t.names[a]}_on_{t.names[b]}" for a, b, _ in t.goals + t.logged] + ["moved_other_obj"]


def vla_stats(roots: List[str]) -> Dict[tuple, Dict[str, bool]]:
    """(task, episode) -> the episode stats at the end, from the VLA's video names (SimplerEnv's loop writes each
    stat into the name as _NAME_VALUE)."""
    out = {}
    for root in roots:
        for task in TASKS:
            for f in glob.glob(os.path.join(root, task, "**", "*.mp4"), recursive=True):
                m = VIDEO.search(f.replace("\\", "/"))
                if m:
                    out[task, int(m.group(1))] = {n: f"_{n}_True" in (m.group(2) or "") for n in stat_names(task)}
    return out


def tests(plan: Dict[tuple, float], vla: Dict[tuple, float]) -> list:
    """The planned tests' per-layout values, in their fixed order: (what, values, strata, scale for sign_flip)."""
    keys = [k for k in plan if k[0] in RULE_BREAKING and k in vla]
    gap = np.array([vla[k] - plan[k] for k in keys])
    kitchen = [*ONE_STEP, TWO_STEP]
    episodes = sorted({k[1] for k in plan if all((t, k[1]) in plan and (t, k[1]) in vla for t in kitchen)})

    def lead(t, e):  # the planner's lead over the VLA on one layout
        return plan[t, e] - vla[t, e]

    cross = np.array([np.mean([lead(t, e) for t in ONE_STEP]) - lead(TWO_STEP, e) for e in episodes])
    return [("the VLA beats the planner on the tasks the rule cannot read (two-step, near plate, far plate; "
             "VLA-only minus planner-only)", gap, np.array([k[0] for k in keys]), 1),
            ("in the kitchen scene the planner's lead over the VLA shrinks from the one-step tasks to the two-step "
             "task (planner-only minus VLA-only: the mean of the two one-step tasks minus the two-step task)",
             cross, np.array(["kitchen"] * len(episodes)), 2)]


def print_tests(plan: Dict[tuple, float], vla: Dict[tuple, float]) -> None:
    """Each test two-sided at 0.05, and it counts only when every test before it was significant."""
    print("Planned tests (fixed order, each two-sided at 0.05; one value per layout):")
    counts = True
    for i, (what, d, strata, scale) in enumerate(tests(plan, vla), 1):
        p = sign_flip(d, scale)
        lo, hi = bootstrap_interval(d, strata)
        verdict = (": significant" if p < 0.05 else ": not significant") if counts else \
            ": does not count (a test before it was not significant)"
        counts = counts and p < 0.05
        print(f"{i}. {what}: {100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
              f"{int((d > 0).sum())} layouts up / {int((d < 0).sum())} down, exact p = {p:.2g}{verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", default="harder_paper", help="the planner-only eval")
    ap.add_argument("--vla", default="vla_harder_paper", help="the VLA-only eval")
    a = ap.parse_args()
    records, vla, keys = s.matched_layouts(runs.path(a.planner), runs.paths([a.vla]), TASKS)
    plan = {k: float(records[k]["success"]) for k in keys}
    vla = {k: vla[k] for k in keys}
    vstats = vla_stats(runs.paths([a.vla]))

    print(f"# Harder scenes: {len(keys)} layouts, clean camera\n")
    print("| Task | Instruction | Layouts | Planner-only | VLA-only | Oracle |\n|---|---|---|---|---|---|")
    for t in TASKS:
        ks = [k for k in keys if k[0] == t]
        p, v = np.array([plan[k] for k in ks]), np.array([vla[k] for k in ks])
        print(f"| {hs.SHORT[t]} | {hs.TASKS[t].instruction} | {len(ks)} | {100 * p.mean():.1f} | "
              f"{100 * v.mean():.1f} | {100 * np.maximum(p, v).mean():.1f} |")
    print()
    print_tests(plan, vla)

    print("\nSecondary (no claims):")
    for t in TASKS:
        ks = [k for k in keys if k[0] == t]
        up, down, p = mcnemar(np.array([plan[k] for k in ks]), np.array([vla[k] for k in ks]))
        print(f"- {hs.SHORT[t]}, VLA vs planner (exact McNemar): {up} layouts up / {down} down, p = {p:.2g}")
    print("\nEpisode stats at the end (% of layouts; planner-only / VLA-only):")
    for t in TASKS:
        ks = [k for k in keys if k[0] == t]
        parts = []
        for n in stat_names(t):
            pr = np.mean([bool(records[k].get("episode_stats", {}).get(n)) for k in ks])
            vr = np.mean([vstats.get(k, {}).get(n, False) for k in ks])
            parts.append(f"{n} {100 * pr:.0f} / {100 * vr:.0f}")
        print(f"- {hs.SHORT[t]}: " + "; ".join(parts))

    print("\nSimplerEnv's own tasks without the extra objects (other layouts, so not paired):")
    own = {"PutSpoonOnTableClothInScene-v0": ONE_STEP[0], "PutCarrotOnPlateInScene-v0": ONE_STEP[1]}
    o_rec, o_vla, o_keys = s.matched_layouts(runs.path(runs.OWN31_PLANNER), runs.paths(runs.OWN31_VLA))
    for src, t in own.items():
        ks = [k for k in o_keys if k[0] == src]
        new = [k for k in keys if k[0] == t]
        print(f"- {runs.SHORT[src]}: planner-only {100 * np.mean([o_rec[k]['success'] for k in ks]):.1f} -> "
              f"{100 * np.mean([plan[k] for k in new]):.1f}, VLA-only {100 * np.mean([o_vla[k] for k in ks]):.1f} -> "
              f"{100 * np.mean([vla[k] for k in new]):.1f} (with the extra objects)")


if __name__ == "__main__":
    main()
