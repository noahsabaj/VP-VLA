"""The VLM plan-check study (README.md, "VLM plan-check study", has the plan that this scorer follows).

The gate (plan_check.gate): an episode goes to the VLA when the VLM's P(yes) that the planner's one pick-and-place
finishes the instruction is below 0.5, else to the planner. It is scored offline, layout by layout: the gate's
choice from the first frame of the planner-only eval, and the outcome of the chosen method's eval of that layout.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_plan_check [--asm two_model/asm_none.json]

Runs (the planner-only ones with PLAN_CHECK=1): the harder scenes, harder_paper and vla_harder_paper; the reworded
instructions, words_{set}_paper and vla_words_{set}_paper; SimplerEnv's own instructions, check_paper_none and
runs.FAST_VLA["paper"]. --asm adds the gate followed by the saved ASM (secondary; SimplerEnv's tasks only).
"""

import argparse
from typing import Dict, List, Optional

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS

from .. import plan_check
from .. import selector as s
from . import runs
from .stats import auc, bootstrap_interval, mcnemar, sign_flip


def load(planner: str, vla: List[str], tasks: Optional[List[str]] = None) -> Dict[tuple, dict]:
    """(task, episode) -> the planner record and both outcomes, for the layouts both evals ran. Every planner record
    must have the plan check."""
    records, vla_out, keys = s.matched_layouts(runs.path(planner), runs.paths(vla), tasks)
    out = {}
    for k in keys:
        p = records[k].get("features", {}).get("vlm_plan_yes")
        if p is None:
            raise SystemExit(f"{planner}: {k} has no plan check (run the planner-only eval with PLAN_CHECK=1)")
        out[k] = {"record": records[k], "planner": float(records[k]["success"]), "vla": vla_out[k], "p_yes": float(p)}
    return out


def columns(rows: Dict[tuple, dict], keys: List[tuple]) -> Dict[str, np.ndarray]:
    """planner, vla, p_yes and gate (the outcome of the method the gate picks) for the keys, in their order."""
    out = {c: np.array([rows[k][c] for k in keys]) for c in ("planner", "vla", "p_yes")}
    out["gate"] = np.array([rows[k][plan_check.gate(rows[k]["p_yes"])] for k in keys])
    return out


def tests(harder: Dict[tuple, dict], words: Dict[str, Dict[tuple, dict]]) -> list:
    """The planned tests' per-layout values, in their fixed order: (what, values, strata, scale for sign_flip). A
    layout of the reworded instructions gives one value, its sum over the 4 sets."""
    hk = sorted(harder)
    h = columns(harder, hk)
    wk = sorted(set.intersection(*(set(v) for v in words.values())))
    w = {name: columns(rows, wk) for name, rows in words.items()}
    reworded = sum(c["gate"] - c["planner"] for c in w.values())
    return [("on the harder scenes the gate beats planner-only", h["gate"] - h["planner"],
             np.array([k[0] for k in hk]), 1),
            ("on the harder scenes the gate beats VLA-only", h["gate"] - h["vla"], np.array([k[0] for k in hk]), 1),
            ("on the reworded instructions the gate beats planner-only (sum over the 4 sets)", reworded,
             np.array([k[0] for k in wk]), 1)]


def print_tests(harder: Dict[tuple, dict], words: Dict[str, Dict[tuple, dict]]) -> None:
    """Each test two-sided at 0.05, and it counts only when every test before it was significant."""
    print("Planned tests (fixed order, each two-sided at 0.05; one value per layout):")
    counts = True
    for i, (what, d, strata, scale) in enumerate(tests(harder, words), 1):
        p = sign_flip(d, scale)
        lo, hi = bootstrap_interval(d, strata)
        verdict = (": significant" if p < 0.05 else ": not significant") if counts else \
            ": does not count (a test before it was not significant)"
        counts = counts and p < 0.05
        print(f"{i}. {what}: {100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
              f"{int((d > 0).sum())} layouts up / {int((d < 0).sum())} down, exact p = {p:.2g}{verdict}")


def table(name: str, rows: Dict[tuple, dict], tasks: List[str], short: Dict[str, str]) -> None:
    for t in tasks:
        keys = [k for k in sorted(rows) if k[0] == t]
        if not keys:
            continue
        c = columns(rows, keys)
        picks = int((c["p_yes"] < plan_check.THRESHOLD).sum())
        print(f"| {name} | {short[t]} | {len(keys)} | {100 * c['planner'].mean():.1f} | {100 * c['vla'].mean():.1f} | "
              f"{100 * c['gate'].mean():.1f} | {100 * np.maximum(c['planner'], c['vla']).mean():.1f} | {picks} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asm", default=None, help="a saved ASM (selector.py --save) after the gate (secondary)")
    a = ap.parse_args()
    harder = load("harder_paper", ["vla_harder_paper"], list(hs.TASKS))
    words = {n: load(f"words_{n}_paper", [f"vla_words_{n}_paper"]) for n in INSTRUCTION_SETS}
    own = load("check_paper_none", runs.FAST_VLA["paper"])

    print("# VLM plan check: the gate (the VLA when P(yes) < 0.5, else the planner), offline\n")
    print("| Instructions | Task | Layouts | Planner-only | VLA-only | Gate | Oracle | VLA picks |\n"
          "|---|---|---|---|---|---|---|---|")
    table("harder scenes", harder, list(hs.TASKS), hs.SHORT)
    table("own", own, s.TASKS, runs.SHORT)
    for n, rows in words.items():
        table(n, rows, s.TASKS, runs.SHORT)
    print()
    print_tests(harder, words)

    print("\nSecondary (no claims):")
    for label, rows in [("own instructions", own), ("harder scenes", harder), *words.items()]:
        c = columns(rows, sorted(rows))
        up_p, down_p, pp = mcnemar(c["planner"], c["gate"])
        up_v, down_v, pv = mcnemar(c["vla"], c["gate"])
        print(f"- {label}: gate vs planner-only {100 * (c['gate'] - c['planner']).mean():+.1f} ({up_p} up / "
              f"{down_p} down, p = {pp:.2g}); gate vs VLA-only {100 * (c['gate'] - c['vla']).mean():+.1f} ({up_v} up "
              f"/ {down_v} down, p = {pv:.2g}); AUC of P(yes) for planner success {auc(c['p_yes'], c['planner']):.2f}")
    mass = np.array([r["record"].get("plan_check_answer_mass") or np.nan
                     for rows in [harder, own, *words.values()] for r in rows.values()])
    print(f"- the VLM's probability on yes/no as its first word: min {np.nanmin(mass):.3f}, "
          f"mean {np.nanmean(mass):.3f} (episodes with no plan to show: {int(np.isnan(mass).sum())})")
    if a.asm:
        asm = s.load_asm(runs.path(a.asm))
        print(f"\nThe gate, then the ASM {a.asm} (SimplerEnv's tasks only):")
        for label, rows in [("own", own), *words.items()]:
            out = [r["vla"] if plan_check.gate(r["p_yes"]) == "vla" or asm.decide(r["record"], k[0])["choice"] == "vla"
                   else r["planner"] for k, r in sorted(rows.items())]
            print(f"- {label}: {100 * np.mean(out):.1f}")


if __name__ == "__main__":
    main()
