"""The two-model ASM study (README.md, "Two-model ASM study", has the plan that this scorer follows).

The two-model ASM gives an episode to the planner when P(planner succeeds) >= P(VLA succeeds), with one success
model per method (selector.TwoModelASM). This scorer trains it on the training set, scores it on the test set
against the baselines and the other decision rules with the planned tests, and then scores it the same way on
the other layout sets.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_two_model \\
        [--runs pseed] [--train iid11] [--test iid12] [--other paper seed2] [--vla 'vla_{set}_{level}' --noise camera]

Each layout set needs the planner-only evals {runs}_{set}_{level} and the VLA-only evals runs.FAST_VLA[set], or
with --vla one VLA-only eval per set and level (the camera-noise study, where the VLA's picture is noisy too). The
rows with the action spread and the linear probe also need the first-step files runs.PERTURB[set] of the training
set and of the test set (not with --vla). selector.py --rule two-model saves the same ASMs for the closed-loop eval.
"""

import argparse
import glob
import os
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from examples.SimplerEnv.eval_files.vla_perturbation import action_spread

from .. import selector as s
from . import runs
from .linear_probe import probe, within_task_auc
from .runs import LEVEL_NAMES, LEVELS, SHORT
from .stats import auc, bootstrap_interval, brier, calibration_bins, ece, logloss, sign_flip

PRIMARY_LEVEL = "harsh"  # with harsh depth the planner fails most, so the choice matters most
ASM = "two-model ASM: P(planner) >= P(VLA)"
V3 = "ASM v3: P(planner) >= 0.5, no task columns"
RULE_A = "rule A: P(planner) >= 0.5"
RULE_C = "rule C: P(planner) >= the task's VLA rate"
SPREAD = "P(planner) >= P(VLA) with the action spread"
PROBE = "P(planner) >= the linear probe's P(VLA)"
MIN_SPREAD = 1e-6  # the log of the action spread needs a floor


class LayoutSet(NamedTuple):
    """One layout set at one depth level: the episodes that both the planner-only and the VLA-only evals ran."""

    keys: List[tuple]
    records: List[dict]  # the planner-only eval's records
    tasks: np.ndarray
    y_plan: np.ndarray
    y_vla: np.ndarray


def load(prefix: str, name: str, level: str, vla_pattern: Optional[str] = None) -> LayoutSet:
    vla_runs = [vla_pattern.format(set=name, level=level)] if vla_pattern else runs.FAST_VLA[name]
    planner, vla, keys = s.matched_layouts(runs.path(f"{prefix}_{name}_{level}"), runs.paths(vla_runs))
    records = [planner[k] for k in keys]
    return LayoutSet(keys, records, np.array([k[0] for k in keys]), np.array([float(r["success"]) for r in records]),
                     np.array([vla[k] for k in keys]))


def first_step(name: str) -> Dict[tuple, tuple]:
    """(task, episode) -> (the last layer's hidden state, the log action spread) from the first-step files of the
    set's VLA-only eval (runs.PERTURB); empty when the set has none."""
    out = {}
    root = runs.PERTURB.get(name)
    if root is None or not os.path.isdir(runs.path(root)):
        return out
    for f in sorted(glob.glob(os.path.join(runs.path(root), "**", "*.npz"), recursive=True)):
        env, _, ep = os.path.basename(f)[:-4].rsplit("_", 2)
        key = (env, int(ep))
        if key in out:
            raise SystemExit(f"{f}: a second first-step file for {key}")
        d = np.load(f)
        if "perturbed_actions" not in d:
            raise SystemExit(f"{f}: no perturbed actions (the VLA-only eval needs PERTURB)")
        out[key] = (d["hidden"][2], float(np.log(max(action_spread(d["perturbed_actions"]), MIN_SPREAD))))
    return out


def score(train: LayoutSet, test: LayoutSet, f_train: Optional[dict], f_test: Optional[dict], name: str):
    """The ASM trained on `train`, and on `test`: each setup's outcomes, its VLA picks, and each success model's
    (P, outcome). f_train and f_test are the first-step files (first_step), or None when they are not used."""
    asm = s.fit_two_model(train.records, train.tasks, train.y_plan, train.y_vla)
    p_plan, p_vla = asm.probs(test.records, test.tasks)
    p_rate = s.task_rate(train.y_vla, train.tasks, test.tasks)

    def v3(d):
        return np.stack([s.row_features(r, s.FEATURES_V3) for r in d.records])

    use_planner = {V3: s.LogisticRegression().fit(v3(train), train.y_plan).prob(v3(test)) >= 0.5,
                   RULE_A: p_plan >= 0.5, RULE_C: p_plan >= p_rate, ASM: p_plan >= p_vla}
    probs = {"P(planner)": (p_plan, test.y_plan),
             "P(planner): the task's rate": (s.task_rate(train.y_plan, train.tasks, test.tasks), test.y_plan),
             "P(VLA)": (p_vla, test.y_vla), "P(VLA): the task's rate": (p_rate, test.y_vla)}
    if f_train is not None and all(k in f_train for k in train.keys) and all(k in f_test for k in test.keys):
        spread = s.SuccessModel([*s.VLA_FEATURES, "log_action_spread"])

        def xs(d, f):
            return np.hstack([asm.vla.x(d.records), np.array([[f[k][1]] for k in d.keys])])

        def hidden(d, f):
            return np.stack([f[k][0] for k in d.keys])

        p_spread = spread.fit(xs(train, f_train), train.tasks, train.y_vla).prob(xs(test, f_test), test.tasks)
        p_probe = probe(hidden(train, f_train), train.y_vla, hidden(test, f_test), 10)
        use_planner.update({SPREAD: p_plan >= p_spread, PROBE: p_plan >= p_probe})
        probs.update({"P(VLA) with the action spread": (p_spread, test.y_vla),
                      "P(VLA): the linear probe": (p_probe, test.y_vla)})
    elif f_train is not None and name in runs.PERTURB:
        print(f"warning: {name}: episodes with no first-step file; the action-spread and probe rows are left out")
    hx = np.stack([s.row_features(r, s.HEURISTIC_FEATURES) for r in test.records])
    use_planner = {"planner-only": np.ones(len(test.keys), bool), "VLA-only": np.zeros(len(test.keys), bool),
                   "heuristic baseline (plan found, SAM3 >= 0.3)": s.heuristic_uses_planner(hx, s.HEURISTIC_FEATURES),
                   **use_planner}
    rows = {k: np.where(u, test.y_plan, test.y_vla) for k, u in use_planner.items()}
    rows["oracle"] = np.maximum(test.y_plan, test.y_vla)
    return asm, rows, {k: int((~u).sum()) for k, u in use_planner.items()}, probs


def print_rows(rows: Dict[str, np.ndarray], picks: Dict[str, int], tasks: np.ndarray) -> None:
    print("| Setup | " + " | ".join(SHORT[k] for k in s.TASKS) + " | All | 95% interval | VLA picks |")
    print("|---" * 8 + "|")
    for name, v in rows.items():
        lo, hi = bootstrap_interval(v, tasks)
        print(f"| {name} | " + " | ".join(f"{100 * v[tasks == k].mean():.1f}" for k in s.TASKS)
              + f" | {100 * v.mean():.1f} | {100 * lo:.1f} to {100 * hi:.1f} | {picks.get(name, '')} |")


def print_tests(rows: Dict[str, np.ndarray], train: LayoutSet, test: LayoutSet, planned: bool) -> None:
    """The two-model ASM against the better single method on the training set, rule A and rule C: in this fixed
    order for the planned tests, where a test counts only when every test before it was significant."""
    best = "planner-only" if train.y_plan.mean() >= train.y_vla.mean() else "VLA-only"
    sequence = [(best, "the better single method on the training set"), (RULE_A, "the gain from P(VLA)"),
                (RULE_C, "the gain from the episode's VLA features")]
    binary = bool(np.all((test.y_vla == 0) | (test.y_vla == 1)))
    print("Planned tests (fixed order, each two-sided at 0.05):" if planned else "Secondary tests (no claims):")
    counts = True
    for i, (base, why) in enumerate(sequence, 1):
        d = rows[ASM] - rows[base]
        lo, hi = bootstrap_interval(d, test.tasks)
        line = (f"{i}. vs {base} ({why}): {100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to "
                f"{100 * hi:+.1f}), {int((d > 0).sum())} episodes better / {int((d < 0).sum())} worse")
        if binary:
            p = sign_flip(d, 1)
            line += f", exact p = {p:.2g}"
            if planned:
                line += (": significant" if p < 0.05 else ": not significant") if counts else \
                    ": does not count (a test before it was not significant)"
                counts = counts and p < 0.05
        print(line)
    gap = rows["oracle"].mean() - rows[best].mean()
    if gap > 0:
        print(f"The two-model ASM gets {100 * (rows[ASM].mean() - rows[best].mean()) / gap:.0f}% of the oracle's "
              f"gain over {best}.")


def print_models(probs: Dict[str, tuple], tasks: np.ndarray, bins: bool) -> None:
    print("Success models on these episodes:")
    print("| P | AUC | within-task AUC | log-loss | Brier | ECE |\n|---|---|---|---|---|---|")
    for name, (p, y) in probs.items():
        print(f"| {name} | {auc(p, y):.3f} | {within_task_auc(p, y, tasks):.3f} | {logloss(p, y):.3f} | "
              f"{brier(p, y):.3f} | {ece(p, y):.3f} |")
    if bins:
        for name in ("P(planner)", "P(VLA)"):
            p, y = probs[name]
            print(f"Calibration of {name} (10 equal bins):")
            print("| Bin | Episodes | Mean P | Success rate |\n|---|---|---|---|")
            print("\n".join(f"| {lo:.1f} to {hi:.1f} | {n} | {mp:.2f} | {my:.2f} |"
                            for lo, hi, n, mp, my in calibration_bins(p, y)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="pseed", help="run-name prefix of the planner-only evals")
    ap.add_argument("--train", default="iid11", help="the training set")
    ap.add_argument("--test", default="iid12", help="the test set of the planned tests")
    ap.add_argument("--other", nargs="*", default=["paper", "seed2"], help="more test sets (secondary)")
    ap.add_argument("--vla", default=None, help="the VLA-only evals: a run name with {set} and {level}, one eval per "
                    "set and level (default: runs.FAST_VLA[set] at every level); without the first-step rows")
    ap.add_argument("--noise", default="depth", help="what the level is the noise of, for the headings")
    a = ap.parse_args()
    levels = [PRIMARY_LEVEL] + [lvl for lvl in LEVELS if lvl != PRIMARY_LEVEL]
    train = {lvl: load(a.runs, a.train, lvl, a.vla) for lvl in levels}
    f_train = first_step(a.train) if a.vla is None else None
    print(f"# Two-model ASM: trained on {a.train}, tested on {a.test} (planned tests)"
          + (f" and on {', '.join(a.other)} (secondary)" if a.other else ""))
    for name in [a.test, *a.other]:
        f_test = first_step(name) if a.vla is None else None
        for lvl in levels:
            planned = name == a.test and lvl == PRIMARY_LEVEL
            test = load(a.runs, name, lvl, a.vla)
            asm, rows, picks, probs = score(train[lvl], test, f_train, f_test, name)
            print(f"\n## {name}, {LEVEL_NAMES[lvl]} {a.noise}: {len(test.keys)} episodes"
                  + (" (the planned tests)" if planned else ""))
            if name == a.test:
                print(f"Trained on the {len(train[lvl].keys)} episodes of {a.train}:")
                print("\n".join(s.describe(asm)))
            print_rows(rows, picks, test.tasks)
            print_tests(rows, train[lvl], test, planned)
            print_models(probs, test.tasks, bins=planned)


if __name__ == "__main__":
    main()
