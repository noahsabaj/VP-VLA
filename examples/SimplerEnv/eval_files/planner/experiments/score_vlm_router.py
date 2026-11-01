"""The VLM-router study (README.md, "VLM-router study", has the plan that this scorer follows).

The two-model ASM against the VLM router (planner/vlm_router.py), which gives each episode to a method from the VLM's
answers on the first frame alone, in its two versions (described, rates). Both choose between the same two evals of
the test set: the planner-only eval with the router's questions (router_{test}_harsh: the fixed rule, harsh depth,
SAM 3.1), whose records also give the ASM its first look, and the VLA-only eval vla31_{test}. The ASM is trained on
pseed31_{train}_harsh and vla31_{train}, as for the VLM-plan study's fixed-rule rows. Each choice is scored offline,
with the outcome of the chosen method's eval of that episode.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_vlm_router [--train iid11] [--test iid14]

In the iidN sets the 4 tasks of an episode index share their layout, so the tests and intervals take one value per
episode index, and only the indices with all 4 tasks count.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np

from .. import selector as s
from .. import vlm_router as vr
from . import routers, runs
from .linear_probe import within_task_auc
from .runs import SHORT
from .score_two_model import LayoutSet, load
from .score_vlm_plan import paired, print_rows, subset, whole_indices
from .stats import auc, delong, holm

LEVEL = "harsh"  # with harsh depth the planner fails most, so the choice matters most
VLA_RUNS = "vla31_{set}"
ROUTER_RUN = "router"  # router_{set}_harsh: planner-only, the fixed rule, with the VLM router's questions
RULE_RUN = "pseed31"  # pseed31_{set}_harsh: planner-only, the fixed rule, without the questions
PLANNER = "planner-only (the fixed rule)"
VLA = "VLA-only"
ASM = "two-model ASM: the planner when P(planner) >= P(VLA)"
ROUTER = {"described": "VLM router, described: the planner when the VLM's P(yes) for it >= its P(yes) for the VLA",
          "rates": "VLM router with each method's training successes on the task, same rule"}
PER_TASK = "per-task choice: each task's better method on the training set"
ORACLE = "oracle"
PLANNED = tuple(ROUTER[v] for v in vr.VERSIONS)  # the planned tests, together by Holm's method at 0.05


def router_choice(records, version: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(use the planner, the VLM's P(yes) for the planner, for the VLA) of each record, in this version."""
    p = np.array([vr.scores(r, version) for r in records], float).reshape(-1, 2)
    return vr.choose(p[:, 0], p[:, 1]), p[:, 0], p[:, 1]


def check_rates(train: LayoutSet) -> None:
    """A note when the rates in the questions (vlm_router.TRAINING_SUCCESSES) are not the training set's."""
    for t in s.TASKS:
        m = train.tasks == t
        got = {"planner": int(train.y_plan[m].sum()), "vla": int(round(train.y_vla[m].sum()))}
        if int(m.sum()) != vr.TRAINING_EPISODES or got != vr.TRAINING_SUCCESSES[t]:
            print(f"note: {SHORT[t]}: the questions say {vr.TRAINING_SUCCESSES[t]} of {vr.TRAINING_EPISODES}, the "
                  f"training set has {got} of {int(m.sum())}")


def print_tests(rows: Dict[str, np.ndarray], ls: LayoutSet) -> None:
    """The planned tests: the ASM against each VLM router, together by Holm's method at 0.05; then the secondary
    ones (no claims)."""
    print("Planned tests (each two-sided; one value per episode index; together by Holm's method at 0.05: the smaller "
          "p below 0.025, then the other below 0.05):")
    results = [paired(rows[ASM], rows[base], ls.keys) for base in PLANNED]
    for i, (base, (line, p, mean), sig) in enumerate(zip(PLANNED, results, holm([p for _, p, _ in results]),
                                                         strict=True), 1):
        word = (": a win" if mean > 0 else ": significant, but the ASM is lower") if sig else ": not significant"
        print(f"{i}. vs {base}: {line}{word}")
    better = PLANNER if rows[PLANNER].mean() >= rows[VLA].mean() else VLA
    print("Secondary tests (no claims):")
    for new, base in [*((r, better) for r in PLANNED), *((r, PER_TASK) for r in PLANNED), (ASM, PER_TASK),
                      (ROUTER["rates"], ROUTER["described"])]:
        print(f"- {new} vs {base}: {paired(rows[new], rows[base], ls.keys)[0]}")
    gap = rows[ORACLE].mean() - rows[better].mean()
    if gap > 0:
        print("Share of the oracle's gain over the better single method: "
              + ", ".join(f"{name} {100 * (rows[name].mean() - rows[better].mean()) / gap:.0f}%"
                          for name in (ASM, *PLANNED)))


def print_choices(use: Dict[str, np.ndarray], ls: LayoutSet) -> None:
    """Each router's share of episodes to the planner, per task, and how often it chooses as the ASM does."""
    print("Episodes to the planner (%), and agreement with the ASM's choice (%):")
    print("| Router | " + " | ".join(SHORT[t] for t in s.TASKS) + " | All | Same as the ASM |")
    print("|---" * 7 + "|")
    for name, u in use.items():
        print(f"| {name} | " + " | ".join(f"{100 * u[ls.tasks == t].mean():.0f}" for t in s.TASKS)
              + f" | {100 * u.mean():.1f} | {100 * (u == use[ASM]).mean():.1f} |")


def print_scores(p: Dict[str, Tuple[np.ndarray, np.ndarray]], ls: LayoutSet) -> None:
    """The VLM's P(yes) as a success model of its method, against the ASM's (DeLong's test, no claims)."""
    print("Success models (no claims): AUC, within-task AUC, and DeLong's p against the ASM's model of that method:")
    print("| Method | Model | AUC | Within-task AUC | DeLong p |\n|---|---|---|---|---|")
    for method, y in (("planner", ls.y_plan), ("vla", ls.y_vla)):
        y = (y >= 0.5).astype(float)  # one VLA run: 0/1 outcomes
        asm_p = p[ASM][0 if method == "planner" else 1]
        print(f"| {vr.NAMES[method]} | ASM | {auc(asm_p, y):.3f} | {within_task_auc(asm_p, y, ls.tasks):.3f} | |")
        for version in vr.VERSIONS:
            q = p[ROUTER[version]][0 if method == "planner" else 1]
            print(f"| {vr.NAMES[method]} | VLM, {version} | {auc(q, y):.3f} | {within_task_auc(q, y, ls.tasks):.3f} | "
                  f"{delong(asm_p, q, y)[2]:.2g} |")


def print_answer_mass(ls: LayoutSet) -> None:
    """How much of the VLM's first answer token is yes or no (near 1: it answers the question)."""
    print("The VLM's probability of answering yes or no (mean, minimum):")
    for version in vr.VERSIONS:
        for method in vr.METHODS:
            name = vr.feature(method, version)
            m = np.array([r.get("vlm_router_answer_mass", {}).get(name, np.nan) for r in ls.records], float)
            print(f"- {name}: {np.nanmean(m):.3f}, {np.nanmin(m):.3f}")


def check_planner(ls: LayoutSet, test: str) -> None:
    """The router eval's planner outcomes against the eval of the same episodes without the questions (the same
    seeds, so they should agree)."""
    name = f"{RULE_RUN}_{test}_{LEVEL}"
    if not os.path.isdir(runs.path(name)):
        print(f"note: no {name}, so no check of the router eval's planner outcomes")
        return
    rule = load(RULE_RUN, test, LEVEL, VLA_RUNS)
    ran = set(rule.keys)
    keys = [k for k in ls.keys if k in ran]
    a, b = subset(ls, keys).y_plan, subset(rule, keys).y_plan
    print(f"Check: the planner's outcome is the same as in {name} in {int((a == b).sum())} of {len(keys)} episodes "
          f"({100 * a.mean():.1f}% vs {100 * b.mean():.1f}% success).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="iid11", help="the training set")
    ap.add_argument("--test", default="iid14", help="the test set of the planned tests")
    a = ap.parse_args()
    print(f"# VLM-router study: trained on {a.train}, tested on {a.test}, {runs.LEVEL_NAMES[LEVEL]} depth")
    asm = s.train_two_model(runs.path(f"{RULE_RUN}_{a.train}_{LEVEL}"), runs.paths([VLA_RUNS.format(set=a.train)]))
    print("\n".join(s.describe(asm)))
    train = load(RULE_RUN, a.train, LEVEL, VLA_RUNS)
    check_rates(train)
    name = f"{ROUTER_RUN}_{a.test}_{LEVEL}"
    test = whole_indices(load(ROUTER_RUN, a.test, LEVEL, VLA_RUNS), name)
    p_plan, p_vla = asm.probs(test.records, test.tasks)
    use = {ASM: p_plan >= p_vla}
    probs = {ASM: (p_plan, p_vla)}
    for version in vr.VERSIONS:
        u, q_plan, q_vla = router_choice(test.records, version)
        use[ROUTER[version]], probs[ROUTER[version]] = u, (q_plan, q_vla)
    use[PER_TASK] = routers.per_task_choice(train.tasks, train.y_plan, train.y_vla, test.tasks)
    rows = {PLANNER: test.y_plan, VLA: test.y_vla, **{k: np.where(u, test.y_plan, test.y_vla) for k, u in use.items()},
            ORACLE: np.maximum(test.y_plan, test.y_vla)}
    picks = {PLANNER: 0, VLA: len(test.keys), **{k: int((~u).sum()) for k, u in use.items()}}
    print(f"\n{len(test.keys)} episodes, {len(test.keys) // len(s.TASKS)} episode indices:")
    print_rows(rows, picks, test)
    print_tests(rows, test)
    print()
    print_choices(use, test)
    print()
    print_scores(probs, test)
    print()
    print_answer_mass(test)
    check_planner(test, a.test)


if __name__ == "__main__":
    main()
