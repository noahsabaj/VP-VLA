"""The VLM-plan study (README.md, "VLM-plan study", has the plan that this scorer follows).

VLM -> planner (the VLM plan, planner-only) against the same system with the ASM and the VLA: a plan that is not
pick-and-place goes to the VLA (selector.vlm_plan_gate), and the two-model ASM decides the other episodes. The ASM is
trained on the training set as `selector.py --rule two-model --planner-candidates` trains it, with P(planner)'s
features chosen there from CANDIDATES. Each choice is scored offline: from the planner-only eval's first look, with
the outcome of the chosen method's eval of that episode. The role split (the planner to above the object, then the
VLA) is a closed-loop eval of its own (HANDOFF_RUN); the other baselines (routers.py) choose offline as the ASM does.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_vlm_plan [--train iid11] [--test iid15]

Runs: planner-only vlmplan_{set}_{level} (the VLM plan) and VLA-only vla31_{set}, at the depth levels harsh (the
planned tests) and none (secondary); the role split's handoff_vlmplan_{test}/{level}; the fixed rule's planner-only
pseed31_{set}_harsh (secondary). A run that is missing is left out, with a note. In the iidN sets the 4 tasks of an
episode index share their layout, so the tests and intervals take one value per episode index, and only the indices
with all 4 tasks count.
"""

import argparse
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import selector as s
from . import mechanism, routers, runs
from .check_closed_loop import live_records
from .runs import LEVEL_NAMES, SHORT
from .score_two_model import LayoutSet, load, print_models
from .stats import bootstrap_interval, holm, sign_flip

PRIMARY_LEVEL = "harsh"  # with harsh depth the planner fails most, so the choice matters most
LEVELS = [PRIMARY_LEVEL, "none"]
# P(planner)'s candidate features, fixed before any eval with the VLM plan ran; the first one wins a tie.
CANDIDATES = [s.PLANNER_FEATURES, [*s.PLANNER_FEATURES, "grasp_sam3_score", "place_sam3_score"]]
VLA_RUNS = "vla31_{set}"
HANDOFF_RUN = "handoff_vlmplan_{set}"  # selector_eval.sh with CONTROLLER=planner_then_vla, one folder per depth level
VLM_PLANNER = "VLM -> planner (planner-only)"
VLA = "VLA-only"
GATE = "VLM -> planner, the VLA when the plan is not pick-and-place"
ASM = "VLM -> ASM: that gate, then the planner when P(planner) >= P(VLA)"
HANDOFF = "role split: VLM -> planner to above the object, then the VLA (closed loop)"
PER_TASK = "per-task choice: the gate, then each task's better method on the training set"
RULE_A = "rule A: the gate, then the planner when P(planner) >= 0.5"
RULE_C = "rule C: the gate, then the planner when P(planner) >= the task's VLA rate"
KNN = "retrieval router: the gate, then the better method on the k nearest training episodes of the task"
RANDOM = "random: the gate, then a coin with the ASM's planner share (expected)"
RULE_PLANNER = "fixed rule -> planner (planner-only)"
RULE_ASM = "fixed rule -> ASM: the planner when P(planner) >= P(VLA)"
SEQUENCE = (VLM_PLANNER, VLA)  # planned tests 1 and 2, in this order
HOLM = (HANDOFF, PER_TASK)  # planned tests 3 and 4, by Holm's method, when tests 1 and 2 are wins
SECONDARY = (RULE_A, RULE_C, KNN)  # tested too, no claims


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


def ungated(ls: LayoutSet) -> LayoutSet:
    """The episodes the ASM decides: those the VLM plan's gate does not give to the VLA (the ASM's training data)."""
    return subset(ls, [k for k, r in zip(ls.keys, ls.records, strict=True) if s.vlm_plan_gate(r) is None])


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


def complete(v: np.ndarray, keys: List[tuple]) -> List[tuple]:
    """The keys of the episode indices whose 4 tasks all have a value in v (not nan)."""
    tasks_of = Counter(e for (_, e), x in zip(keys, v, strict=True) if not np.isnan(x))
    return [k for k in keys if tasks_of[k[1]] == len(s.TASKS)]


def paired(new: np.ndarray, base: np.ndarray, keys: List[tuple]) -> Tuple[str, float, float]:
    """new - base with one value per episode index: (a line with the mean difference, its 95% bootstrap interval over
    the indices and the indices up and down; the exact paired sign-flip test's p on the per-index sums; the mean).
    An index with a nan in base (an eval that did not run it) is left out."""
    keep = complete(base, keys)
    if len(keep) < len(keys):
        pos = {k: i for i, k in enumerate(keys)}
        idx = np.array([pos[k] for k in keep], int)
        new, base, keys = new[idx], base[idx], keep
    if not keys:
        return "no episode index with all 4 tasks", 1.0, 0.0
    d = per_index(new - base, keys)
    per = d.sum(1)
    lo, hi = bootstrap_interval(d.mean(1), np.zeros(len(d)))
    p = sign_flip(per, 1)
    return (f"{100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), "
            f"{int((per > 0).sum())} episode indices up / {int((per < 0).sum())} down, exact p = {p:.2g}",
            p, float(d.mean()))


def print_rows(rows: Dict[str, np.ndarray], picks: Dict[str, int], ls: LayoutSet) -> None:
    """One line per setup. A setup with nan for some episodes (an eval that did not run them) is scored on the episode
    indices it has."""
    print("| Setup | " + " | ".join(SHORT[k] for k in s.TASKS) + " | All | 95% interval | VLA picks |")
    print("|---" * 8 + "|")
    for name, v in rows.items():
        keys = complete(v, ls.keys)
        if not keys:
            continue
        pos = {k: i for i, k in enumerate(ls.keys)}
        idx = np.array([pos[k] for k in keys], int)
        w, tasks = v[idx], ls.tasks[idx]
        m = per_index(w, keys).mean(1)
        lo, hi = bootstrap_interval(m, np.zeros(len(m)))
        note = f" ({len(keys) // len(s.TASKS)} episode indices)" if len(keys) < len(ls.keys) else ""
        print(f"| {name}{note} | " + " | ".join(f"{100 * w[tasks == k].mean():.1f}" for k in s.TASKS)
              + f" | {100 * w.mean():.1f} | {100 * lo:.1f} to {100 * hi:.1f} | {picks.get(name, '')} |")


def verdict(p: float, mean: float, counts: bool, significant: Optional[bool] = None) -> str:
    """A planned test's word: a win (significant, the ASM higher), or why not."""
    if not counts:
        return ": does not count (a test before it was not a win)"
    significant = p < 0.05 if significant is None else significant
    if significant:
        return ": a win" if mean > 0 else ": significant, but the ASM is lower"
    return ": not significant"


def print_tests(rows: Dict[str, np.ndarray], ls: LayoutSet, planned: bool) -> None:
    """The ASM against VLM -> planner, then against the VLA: in this fixed order for the planned tests, where a test
    counts only when every test before it was a win (significant, with the ASM higher). Then tests 3 and 4, against the
    role split and the per-task choice, which count only when tests 1 and 2 were wins, and then by Holm's method at
    0.05. A baseline that did not run gives no test (for Holm: p = 1). Then the secondary baselines."""
    print("Planned tests (each two-sided; one value per episode index):" if planned
          else "Secondary tests (no claims; one value per episode index):")
    counts = True
    for i, base in enumerate(SEQUENCE, 1):
        line, p, mean = paired(rows[ASM], rows[base], ls.keys)
        if planned:
            win = p < 0.05 and mean > 0
            line += verdict(p, mean, counts)
            counts = counts and win
        print(f"{i}. vs {base}: {line}")
    results = [paired(rows[ASM], rows[b], ls.keys) if b in rows else (f"no result: {b} did not run", 1.0, 0.0)
               for b in HOLM]
    significant = holm([p for _, p, _ in results])
    if planned:
        print("Tests 3 and 4 (only when tests 1 and 2 were wins; Holm's method at 0.05: the smaller p below 0.025, "
              "then the other below 0.05):")
    for i, (base, (line, p, mean), sig) in enumerate(zip(HOLM, results, significant, strict=True), len(SEQUENCE) + 1):
        if planned and base in rows:
            line += verdict(p, mean, counts, sig)
        print(f"{i}. vs {base}: {line}")
    for base in SECONDARY:
        if base in rows:
            print(f"- vs {base} (secondary): {paired(rows[ASM], rows[base], ls.keys)[0]}")
    gap = rows["oracle"].mean() - rows[VLM_PLANNER].mean()
    if gap > 0:
        print(f"The ASM gets {100 * (rows[ASM].mean() - rows[VLM_PLANNER].mean()) / gap:.0f}% of the oracle's gain "
              f"over VLM -> planner.")


def handoff_row(ls: LayoutSet, name: str, level: str) -> Optional[np.ndarray]:
    """The role split's outcome of each episode (nan where its eval has none), or None when it did not run."""
    root = os.path.join(runs.path(HANDOFF_RUN.format(set=name)), level)
    if not os.path.isdir(root):
        return None
    live = live_records(root, mode="planner_then_vla")
    v = np.array([float(live[k]["success"]) if k in live else np.nan for k in ls.keys])
    if np.isnan(v).any():
        print(f"note: {HANDOFF_RUN.format(set=name)}/{level}: {int(np.isnan(v).sum())} episodes not run")
    return v


def baselines(asm: s.TwoModelASM, train: LayoutSet, test: LayoutSet, use: np.ndarray, gated: np.ndarray,
              p_plan: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """The offline routers, each behind the gate, trained on the training set's ungated episodes: (rows, VLA picks).
    The retrieval router uses the ASM's features and the k that routers.choose_k picks on the training set."""
    train = ungated(train)
    pick = {PER_TASK: routers.per_task_choice(train.tasks, train.y_plan, train.y_vla, test.tasks),
            RULE_A: p_plan >= 0.5, RULE_C: p_plan >= s.task_rate(train.y_vla, train.tasks, test.tasks)}
    features = list(dict.fromkeys([*asm.planner.features, *asm.vla.features]))
    tx = np.stack([s.row_features(r, features) for r in train.records])
    x = np.stack([s.row_features(r, features) for r in test.records])
    k, scores = routers.choose_k(tx, train.tasks, train.y_plan, train.y_vla, s.layout_groups(train.records))
    print(f"Retrieval router: k = {k} (cross-validated success on the training set: "
          + ", ".join(f"k {kk}: {100 * v:.1f}" for kk, v in zip(routers.KS, scores, strict=True)) + ")")
    pick[KNN] = routers.knn_choice(tx, train.tasks, train.y_plan, train.y_vla, x, test.tasks, k)
    rows = {name: np.where(~gated & u, test.y_plan, test.y_vla) for name, u in pick.items()}
    share = use.sum() / max((~gated).sum(), 1)
    rows[RANDOM] = np.where(gated, test.y_vla, routers.expected_random(share, test.y_plan, test.y_vla))
    return rows, {name: int((gated | ~u).sum()) for name, u in pick.items()}


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
        train = load("vlmplan", a.train, lvl, VLA_RUNS)
        extra, extra_picks = baselines(asm, train, test, use, gated, p_plan)
        handoff = handoff_row(test, a.test, lvl)
        if handoff is not None:
            rows[HANDOFF] = handoff
        else:
            print(f"note: no role-split eval {HANDOFF_RUN.format(set=a.test)}/{lvl}")
        rows.update(extra)
        picks.update(extra_picks)
        print(f"{len(test.keys)} episodes, {len(test.keys) // len(s.TASKS)} episode indices:")
        print_rows(rows, picks, test)
        print_tests(rows, test, planned)
        print(f"The ASM decided {int((~gated).sum())} episodes (the gate gave {int(gated.sum())} to the VLA).")
        decided = ~gated
        print_models({"P(planner)": (p_plan[decided], test.y_plan[decided]),
                      "P(VLA)": (p_vla[decided], test.y_vla[decided])}, test.tasks[decided], bins=planned)
        mechanism.report(ungated(train), test, asm, use, gated, np.array([e for _, e in test.keys]),
                         planner_name="VLM -> planner")
        if planned:
            print("\nThe VLM plans:")
            print_plans(test)
            print_rule(test, rows, use, a.train, a.test)


if __name__ == "__main__":
    main()
