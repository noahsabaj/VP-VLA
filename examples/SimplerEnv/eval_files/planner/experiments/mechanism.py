"""What makes the ASM better than either method alone: analyses of a test set's outcomes and the ASM's choices (numpy
only). The newer studies' scorers report them as planned secondary results, with no tests. The outcomes must be 0 or
1 (one eval of each method): the identities below hold only then, and the code checks them.

* complementarity: how often only one method succeeds. The oracle's gain over the better method is
  min(planner-only, VLA-only) / N, the share of episodes that only the worse method solves: a choice per episode can
  gain only there.
* failure dependence: within each task, P(VLA succeeds | the planner failed) - P(VLA succeeds | the planner succeeded)
  (Mantel-Haenszel). Near 0, the two methods fail on different episodes; below 0, they fail on the same ones.
* gain decomposition: the ASM's gain over each method, from the episodes where only one method succeeds: those it
  gave to the method that succeeds, minus those it gave to the method that fails.
* predicted gain: the gain that the ASM's own probabilities predict, against the real gain (good enough to choose by?).
* ablations: the ASM trained again without each feature, and with only the task columns.
* learning curve: the ASM trained on part of the training set's layout groups.
"""

from typing import List, Optional

import numpy as np

from .. import selector as s
from .score_two_model import LayoutSet
from .stats import cluster_bootstrap_interval, mh_risk_difference, unit_sums

FRACTIONS = (0.1, 0.25, 0.5, 1.0)  # the learning curve's shares of the training set's layout groups
DRAWS = 20  # training subsets per share (seed 0)


def complementarity(y_plan: np.ndarray, y_vla: np.ndarray) -> dict:
    """Episodes where both, only the planner, only the VLA or neither succeeded, and the oracle's gain over the better
    method (equal to min(planner only, VLA only) / N)."""
    n = len(y_plan)
    out = {"both": int(((y_plan == 1) & (y_vla == 1)).sum()), "planner only": int((y_plan > y_vla).sum()),
           "VLA only": int((y_vla > y_plan).sum()), "neither": int(((y_plan == 0) & (y_vla == 0)).sum())}
    out["oracle gain"] = float(np.maximum(y_plan, y_vla).mean() - max(y_plan.mean(), y_vla.mean()))
    assert abs(out["oracle gain"] - min(out["planner only"], out["VLA only"]) / n) < 1e-9
    return out


def failure_dependence(y_plan: np.ndarray, y_vla: np.ndarray, tasks: np.ndarray, units: np.ndarray,
                       draws: int = 2000, seed: int = 0):
    """(the Mantel-Haenszel difference P(VLA ok | planner failed) - P(VLA ok | planner ok) within tasks, its 95%
    bootstrap interval over the units)."""
    est = mh_risk_difference(y_vla, y_plan == 0, tasks)
    names = sorted(set(np.asarray(units).tolist()))
    rows = {u: np.flatnonzero(units == u) for u in names}
    rng, got = np.random.default_rng(seed), []
    for _ in range(draws):
        idx = np.concatenate([rows[names[i]] for i in rng.integers(0, len(names), len(names))])
        got.append(mh_risk_difference(y_vla[idx], y_plan[idx] == 0, tasks[idx]))
    lo, hi = np.nanquantile(got, [0.025, 0.975])
    return float(est), float(lo), float(hi)


def gain_decomposition(use: np.ndarray, y_plan: np.ndarray, y_vla: np.ndarray) -> dict:
    """The ASM's gain over each method from the episodes that only one method solves. Over the VLA: (planner-only
    episodes it gave the planner - VLA-only episodes it gave the planner) / N; over the planner, the same with the
    VLA's picks."""
    p_only, v_only, n = y_plan > y_vla, y_vla > y_plan, len(use)
    out = {"planner-only to the planner": int((use & p_only).sum()), "planner only": int(p_only.sum()),
           "VLA-only to the planner": int((use & v_only).sum()), "VLA only": int(v_only.sum())}
    out["gain over the VLA"] = (out["planner-only to the planner"] - out["VLA-only to the planner"]) / n
    out["gain over the planner"] = (int((~use & v_only).sum()) - int((~use & p_only).sum())) / n
    chosen = np.where(use, y_plan, y_vla)
    assert abs(out["gain over the VLA"] - (chosen.mean() - y_vla.mean())) < 1e-9
    assert abs(out["gain over the planner"] - (chosen.mean() - y_plan.mean())) < 1e-9
    return out


def fit(train: LayoutSet, planner_features: List[str], vla_features: List[str],
        task_columns: Optional[List[str]] = None) -> s.TwoModelASM:
    asm = s.TwoModelASM(s.SuccessModel(planner_features, tasks=task_columns), s.SuccessModel(vla_features,
                                                                                             tasks=task_columns))
    asm.planner.fit(asm.planner.x(train.records), train.tasks, train.y_plan)
    asm.vla.fit(asm.vla.x(train.records), train.tasks, train.y_vla)
    return asm


def routed(asm: s.TwoModelASM, test: LayoutSet, gated: np.ndarray) -> np.ndarray:
    """Each test episode's outcome with this ASM behind the gate (the VLA for gated episodes)."""
    p_plan, p_vla = asm.probs(test.records, test.tasks)
    return np.where(~gated & (p_plan >= p_vla), test.y_plan, test.y_vla)


def ablations(train: LayoutSet, test: LayoutSet, asm: s.TwoModelASM, gated: np.ndarray):
    """(setup, success rate on the test set) for the ASM trained again without each of its features, and with only
    the task columns in one success model."""
    pf, vf, tc = asm.planner.features, asm.vla.features, asm.planner.tasks
    def score(planner_features, vla_features):
        return routed(fit(train, planner_features, vla_features, tc), test, gated).mean()

    out = [("the ASM", routed(asm, test, gated).mean())]
    out += [(f"P(planner) without {f}", score([g for g in pf if g != f], vf)) for f in pf]
    out += [(f"P(VLA) without {f}", score(pf, [g for g in vf if g != f])) for f in vf]
    out += [("P(planner): the task columns only", score([], vf)), ("P(VLA): the task columns only", score(pf, []))]
    return out


def learning_curve(train: LayoutSet, test: LayoutSet, asm: s.TwoModelASM, gated: np.ndarray, seed: int = 0):
    """(share of the training set's layout groups, the test success rate of each draw) for the ASM's features trained
    on that share (DRAWS draws of the groups, without replacement; one draw for the whole set)."""
    groups = s.layout_groups(train.records)
    names, rng, out = np.unique(groups), np.random.default_rng(seed), []
    for share in FRACTIONS:
        got = []
        for _ in range(1 if share == 1.0 else DRAWS):
            keep = np.isin(groups, rng.choice(names, max(2, round(share * len(names))), replace=False))
            part = LayoutSet([k for k, ok in zip(train.keys, keep, strict=True) if ok],
                             [r for r, ok in zip(train.records, keep, strict=True) if ok],
                             train.tasks[keep], train.y_plan[keep], train.y_vla[keep])
            if len(set(part.tasks.tolist())) < len(set(train.tasks.tolist())):
                continue  # a task with no episodes in the draw has no column to learn
            got.append(routed(fit(part, asm.planner.features, asm.vla.features, asm.planner.tasks), test,
                              gated).mean())
        out.append((share, np.array(got)))
    return out


def report(train: LayoutSet, test: LayoutSet, asm: s.TwoModelASM, use: np.ndarray, gated: np.ndarray,
           units: np.ndarray, planner_name: str = "planner") -> None:
    """The analyses above for one test set, as markdown. use: the ASM's choices (behind the gate); gated: the episodes
    the gate gave to the VLA; units: each episode's unit for the intervals (its episode index)."""
    y_plan, y_vla, n = test.y_plan, test.y_vla, len(use)
    print("\n### What makes the ASM better (planned secondary results, no tests)")
    c = complementarity(y_plan, y_vla)
    print(f"Complementarity: both succeed {c['both']}, only {planner_name} {c['planner only']}, only the VLA "
          f"{c['VLA only']}, neither {c['neither']} (of {n}). The oracle's gain over the better method, "
          f"{100 * c['oracle gain']:.1f} points, is the share of episodes that only the worse method solves.")
    d, lo, hi = failure_dependence(y_plan, y_vla, test.tasks, units)
    print(f"Failure dependence: P(VLA succeeds | {planner_name} failed) - P(VLA succeeds | {planner_name} succeeded), "
          f"within tasks: {100 * d:+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}).")
    g = gain_decomposition(use, y_plan, y_vla)
    print(f"Gain decomposition: of the {g['planner only']} episodes only {planner_name} solves, the ASM gave "
          f"{g['planner-only to the planner']} to it; of the {g['VLA only']} only the VLA solves, it gave "
          f"{g['VLA-only to the planner']} to {planner_name}. Gain over the VLA {100 * g['gain over the VLA']:+.1f}, "
          f"over {planner_name} {100 * g['gain over the planner']:+.1f} points.")
    p_plan, p_vla = asm.probs(test.records, test.tasks)
    predicted = np.where(use, p_plan, p_vla)
    real = np.where(use, y_plan, y_vla)
    print(f"Predicted against real (from the ASM's own probabilities): success {100 * predicted.mean():.1f} against "
          f"{100 * real.mean():.1f}; gain over the VLA {100 * (predicted - p_vla).mean():+.1f} against "
          f"{100 * (real - y_vla).mean():+.1f} points.")
    print("Ablations (trained again on the same training set):\n| Setup | Success | Minus the ASM |\n|---|---|---|")
    rows = ablations(train, test, asm, gated)
    print("\n".join(f"| {name} | {100 * v:.1f} | {100 * (v - rows[0][1]):+.1f} |" for name, v in rows))
    best = max(y_plan.mean(), y_vla.mean())
    print(f"Learning curve ({DRAWS} draws of the training set's layout groups per share, seed 0):\n"
          "| Share | Draws | Mean success | 5% to 95% of the draws | Draws above the better method |\n"
          "|---|---|---|---|---|")
    for share, got in learning_curve(train, test, asm, gated):
        if len(got):
            q5, q95 = np.quantile(got, [0.05, 0.95])
            print(f"| {share:g} | {len(got)} | {100 * got.mean():.1f} | {100 * q5:.1f} to {100 * q95:.1f} | "
                  f"{int((got > best).sum())} |")
    per_unit = unit_sums(real - np.maximum(y_plan, y_vla), units)
    lo, hi = cluster_bootstrap_interval(real - np.maximum(y_plan, y_vla), units)
    print(f"The ASM minus the oracle: {100 * per_unit.sum() / n:+.1f} points (95% interval {100 * lo:+.1f} to "
          f"{100 * hi:+.1f}).")
