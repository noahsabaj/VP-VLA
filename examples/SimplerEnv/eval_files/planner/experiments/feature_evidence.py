"""Evidence for each v2 selector feature (harsh depth, where the planner fails often enough to learn).

1. Why the planner fails (the failure field of each try).
2. Per feature: mean on planner successes vs failures, AUC alone (all layouts and within each task),
   share of zero values.
3. Drop one feature: change in selector score and in planner-success log-loss, in three tests.

    python -m examples.SimplerEnv.eval_files.planner.experiments.feature_evidence --runs pseed \\
        [--paper-vla fast3_train_g0 fast3_train_g1] [--heldout-vla heldout_vla_g0 heldout_vla_g1]

--runs: planner run-name prefix ({runs}_{paper,heldout}_{realistic,harsh}). --paper-vla, --heldout-vla: the VLA
runs of the paper and held-out layouts (several runs give mean labels; default: the fast VLA runs).
"""

import argparse
from collections import Counter

import numpy as np

from .. import selector as s
from . import runs
from .stats import auc, logloss

KINDS = ["held-out -> paper (final run setup)", "paper -> held-out", "leave one task out (288)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="run-name prefix of the planner-only evals")
    ap.add_argument("--paper-vla", nargs="+", default=runs.FAST_VLA["paper"], help="VLA runs of the paper layouts")
    ap.add_argument("--heldout-vla", nargs="+", default=runs.FAST_VLA["heldout"],
                    help="VLA runs of the held-out layouts")
    a = ap.parse_args()
    paper = {lvl: runs.path(f"{a.runs}_paper_{lvl}") for lvl in ("realistic", "harsh")}
    held = {lvl: runs.path(f"{a.runs}_heldout_{lvl}") for lvl in ("realistic", "harsh")}
    v2 = s.FEATURES_V2

    print("## 1. Why the planner fails (paper 96 + held-out 192 layouts)")
    for lvl in ("realistic", "harsh"):
        c = Counter()
        n = 0
        for root in (paper[lvl], held[lvl]):
            for r in s.load_planner(root).values():
                n += 1
                if not r["success"]:
                    fail = r.get("failure") or "no failure noted (ran to the end, task not done)"
                    c[fail.split(":")[0]] += 1
        print(f"{lvl}: {sum(c.values())} failures of {n} tries: " + "; ".join(f"{k} {v}" for k, v in c.most_common()))

    xp, yp_p, yv_p, tp = s.load_layouts(paper["harsh"], runs.paths(a.paper_vla), v2)
    xh, yp_h, yv_h, th = s.load_layouts(held["harsh"], runs.paths(a.heldout_vla), v2)
    X, Y, T = np.vstack([xp, xh]), np.r_[yp_p, yp_h], np.r_[tp, th]
    print(f"\n## 2. Each feature alone, harsh depth, {len(Y)} layouts, planner succeeds on {int(Y.sum())}")
    print("AUC 0.5 = no signal; below 0.5 = higher value means more planner failures. "
          "Within-task AUC = mean over the 4 tasks.")
    print("| Feature | mean if planner succeeds | mean if it fails | AUC | within-task AUC | zero share |")
    for j, n in enumerate(v2):
        x = X[:, j]
        wt = np.nanmean([auc(x[T == t], Y[T == t]) for t in s.TASKS])
        print(f"| {n} | {x[Y == 1].mean():.3g} | {x[Y == 0].mean():.3g} | {auc(x, Y):.2f} | {wt:.2f} | "
              f"{(x == 0).mean():.2f} |")
    print("\nPer task means (succeeds / fails) for the features with task-dependent values:")
    for n in ("grasp_grip_margin", "grasp_depth_roughness_rel", "pregrasp_corridor_points", "plan_path_ratio"):
        j = v2.index(n)
        print(f"  {n}: " + "; ".join(f"{runs.SHORT[t]} {X[(T == t) & (Y == 1), j].mean():.3g} / "
                                     f"{X[(T == t) & (Y == 0), j].mean():.3g}" for t in s.TASKS))

    def test(cols, kind):
        """Selector score (%) and planner-success log-loss for one feature subset."""
        if kind == "held-out -> paper (final run setup)":
            p = s.LogisticRegression().fit(xh[:, cols], yp_h).prob(xp[:, cols])
            return 100 * np.where(p >= 0.5, yp_p, yv_p).mean(), logloss(p, yp_p)
        if kind == "paper -> held-out":
            p = s.LogisticRegression().fit(xp[:, cols], yp_p).prob(xh[:, cols])
            return 100 * np.where(p >= 0.5, yp_h, yv_h).mean(), logloss(p, yp_h)
        # leave one task out on all 288 layouts
        p = s.leave_one_task_out(T, lambda train, test: s.LogisticRegression().fit(X[train][:, cols], Y[train])
                                 .prob(X[test][:, cols]))
        return 100 * np.where(p >= 0.5, Y, np.r_[yv_p, yv_h]).mean(), logloss(p, Y)

    allc = list(range(len(v2)))
    print("\n## 3. Drop one feature (harsh). Score = selector success %, higher is better; log-loss: lower is better.")
    print("A feature earns its place if dropping it lowers the score or raises the log-loss in most tests.")
    base = {k: test(allc, k) for k in KINDS}
    print("| Features | " + " | ".join(f"{k}: score / log-loss" for k in KINDS) + " |")
    print("| all 11 | " + " | ".join(f"{base[k][0]:.1f} / {base[k][1]:.3f}" for k in KINDS) + " |")
    for j, n in enumerate(v2):
        cols = [c for c in allc if c != j]
        r = {k: test(cols, k) for k in KINDS}
        print(f"| drop {n} | "
              + " | ".join(f"{r[k][0] - base[k][0]:+.1f} / {r[k][1] - base[k][1]:+.3f}" for k in KINDS) + " |")
    for label, names in [("only plan_ok + plan_path_ratio", ["plan_ok", "plan_path_ratio"]),
                         ("plan_ok + plan_path_ratio + pregrasp_corridor_points",
                          ["plan_ok", "plan_path_ratio", "pregrasp_corridor_points"]),
                         ("plan_ok only", ["plan_ok"])]:
        cols = [v2.index(n) for n in names]
        r = {k: test(cols, k) for k in KINDS}
        print(f"| {label} | " + " | ".join(f"{r[k][0]:.1f} / {r[k][1]:.3f}" for k in KINDS) + " |")
    print("| always planner | " + " | ".join(f"{v:.1f}" for v in (100 * yp_p.mean(), 100 * yp_h.mean(), 100 * Y.mean()))
          + " |")
    print("| perfect | " + " | ".join(f"{v:.1f}" for v in (100 * np.maximum(yp_p, yv_p).mean(),
                                                         100 * np.maximum(yp_h, yv_h).mean(),
                                                         100 * np.maximum(Y, np.r_[yv_p, yv_h]).mean())) + " |")
    m = s.LogisticRegression().fit(xh, yp_h)
    print("\nSaved harsh model weights (standardized features, + = more likely planner success): " +
          ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(v2, m.w, strict=True), key=lambda a: -abs(a[1]))))


if __name__ == "__main__":
    main()
