"""Planner-only evals after a change against the evals before it, then the ASM scores again (v3 = plan found +
path ratio; v2 and the hand rule for comparison).

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_change --before dfix --after pseed

--before and --after are run-name prefixes: {prefix}_paper_{level} and {prefix}_heldout_{level} for the 3 depth
levels. --models: the saved models of the --after evals (default final_{after}).
"""

import argparse
import math
import os

import numpy as np

from .. import selector as s
from . import runs
from .runs import LEVEL_NAMES, LEVELS
from .stats import mcnemar

SETUPS = ["always planner", "always VLA", "hand rule", "ASM v2", "ASM v3", "perfect"]


def compare(old, new):
    a, b = s.load_planner(old), s.load_planner(new)
    keys = sorted(set(a) & set(b))
    fa = {k: s.row_features(a[k], runs.CHECK) for k in keys}
    fb = {k: s.row_features(b[k], runs.CHECK) for k in keys}
    sf = sum(bool(a[k]["success"]) and not b[k]["success"] for k in keys)
    fs = sum(not a[k]["success"] and bool(b[k]["success"]) for k in keys)
    print(f"| {os.path.basename(old)} -> {os.path.basename(new)} | {len(keys)} | {len(keys) - sf - fs} | {sf} | {fs} | "
          f"{sum(np.array_equal(fa[k], fb[k]) for k in keys)} | "
          f"{sum(bool(a[k]['success']) for k in keys)} -> {sum(bool(b[k]['success']) for k in keys)} |")


def fit_prob(x, y, X=None, tasks=None):
    """P(planner succeeds): trained on x and scored on X, or leave one task out when X is None."""
    if X is not None:
        return s.LogisticRegression().fit(x, y).prob(X)
    return s.leave_one_task_out(tasks, lambda train, test: s.LogisticRegression().fit(x[train], y[train]).prob(x[test]))


def outcomes(data):
    """data(names) -> (x, yp, yv, t, X, YP, YV, T), with X None for leave one task out."""
    rows, picks = {}, {}
    for fs in ("v2", "v3"):
        names = s.FEATURE_SETS[fs]
        x, yp, yv, t, X, YP, YV, T = data(names)
        p = fit_prob(x, yp, X, t)
        if X is None:
            X, YP, YV, T = x, yp, yv, t
        if fs == "v2":
            rows.update({"always planner": YP, "always VLA": YV,
                         "hand rule": np.where(s.heuristic_uses_planner(X, names), YP, YV)})
        rows[f"ASM {fs}"] = np.where(p >= 0.5, YP, YV)
        picks[fs] = int((p < 0.5).sum())
    rows["perfect"] = np.maximum(rows["always planner"], rows["always VLA"])
    return rows, T, picks


def held_out(train_p, train_v, test_p, test_v):
    return lambda names: (*s.load_layouts(train_p, train_v, names), *s.load_layouts(test_p, test_v, names))


def loto(planner, vla):
    return lambda names: (*s.load_layouts(planner, vla, names), None, None, None, None)


def loto_both(paper, paper_vla, held, held_vla):
    return lambda names: (*s.Layouts.concat([s.load_layouts(paper, paper_vla, names),
                                             s.load_layouts(held, held_vla, names)]), None, None, None, None)


def report(title, make):
    """One table: each setup before and after at every depth level, then the tests after."""
    print(f"\n### {title}")
    res = {(lvl, when): outcomes(make(lvl, when)) for lvl in LEVELS for when in ("before", "after")}
    print(f"{len(res['none', 'after'][1])} layouts. Success rate (%):")
    print("| Setup | " + " | ".join(f"{LEVEL_NAMES[lvl]} before | {LEVEL_NAMES[lvl]} after" for lvl in LEVELS) + " |")
    print("|---" * (1 + 2 * len(LEVELS)) + "|")
    for name in SETUPS:
        cells = [f"{100 * res[lvl, when][0][name].mean():.1f}" for lvl in LEVELS for when in ("before", "after")]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("| VLA picks v3 | " + " | ".join(str(res[lvl, when][2]["v3"]) for lvl in LEVELS
                                         for when in ("before", "after")) + " |")
    for lvl in LEVELS:
        rows, T, _ = res[lvl, "after"]
        line = f"{LEVEL_NAMES[lvl]} after, v3 per task: " + " / ".join(
            f"{100 * rows['ASM v3'][T == k].mean():.1f}" for k in s.TASKS)
        yv = rows["always VLA"]
        if np.all((yv == 0) | (yv == 1)):
            for base in ("always planner", "hand rule", "ASM v2"):
                b, w, p = mcnemar(rows[base], rows["ASM v3"])
                line += f"; v3 vs {base}: {b} better / {w} worse, p={p:.2g}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="run-name prefix of the evals before the change")
    ap.add_argument("--after", required=True, help="run-name prefix of the evals after the change")
    ap.add_argument("--models", help="saved models of the --after evals (default final_{after})")
    a = ap.parse_args()
    held_vla, fast_vla, exact_vla = (runs.paths(runs.FAST_VLA["heldout"]), runs.paths(runs.FAST_VLA["paper"]),
                                     runs.paths(runs.EXACT_VLA))
    planner = {(when, lay, lvl): runs.path(f"{prefix}_{lay}_{lvl}") for when, prefix in (("before", a.before),
                                                                                       ("after", a.after))
               for lay in ("paper", "heldout") for lvl in LEVELS}
    final = runs.path(a.models or f"final_{a.after}")

    print("## 1. Planner runs before -> after, layout by layout (plan_time left out: wall-clock)")
    print("| Runs | Layouts | Same outcome | S->F | F->S | Same features (exact) | Successes |")
    print("|---|---|---|---|---|---|---|")
    for lay in ("paper", "heldout"):
        for lvl in LEVELS:
            compare(planner["before", lay, lvl], planner["after", lay, lvl])

    print("\n## 2. Planner alone after, per task (%)")
    print("| Layouts | Depth | " + " | ".join(runs.SHORT.values()) + " | Avg |")
    print("|---|---|---|---|---|---|---|")
    for label, lay in (("paper", "paper"), ("held-out", "heldout")):
        for lvl in LEVELS:
            r = s.load_planner(planner["after", lay, lvl])
            per = [np.mean([v["success"] for k, v in r.items() if k[0] == t]) for t in s.TASKS]
            print(f"| {label} | {LEVEL_NAMES[lvl]} | " + " | ".join(f"{100 * p:.1f}" for p in per) +
                  f" | {100 * np.mean([v['success'] for v in r.values()]):.1f} ({sum(v['success'] for v in r.values())}"
                  f"/{len(r)}) |")

    def evals(lvl, when):
        return planner[when, "paper", lvl], planner[when, "heldout", lvl]

    print("\n## 3. ASM scores before and after")
    report("Train on the 192 held-out layouts, test on the 96 paper layouts "
           "(VLA: fast run; the closed-loop eval's setup)",
           lambda lvl, when: held_out(evals(lvl, when)[1], held_vla, evals(lvl, when)[0], fast_vla))
    report("Train on the 96 paper layouts (VLA: mean of the 5 exact runs), test on the 192 held-out layouts",
           lambda lvl, when: held_out(evals(lvl, when)[0], exact_vla, evals(lvl, when)[1], held_vla))
    report("Leave one task out, all 288 layouts (VLA: fast runs)",
           lambda lvl, when: loto_both(evals(lvl, when)[0], fast_vla, evals(lvl, when)[1], held_vla))
    report("Leave one task out, the 96 paper layouts (VLA: mean of the 5 exact runs)",
           lambda lvl, when: loto(evals(lvl, when)[0], exact_vla))

    print(f"\n## 4. Saved models ({final}), trained on the 192 held-out layouts of the runs after")
    for lvl in LEVELS:
        path = f"{final}/sel_v3_{lvl}.json"
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        model, names = s.load_saved(path)
        w, b, m, sd = model.w / model.std, model.b, model.mean, model.std
        b0 = b - float((model.w * m / sd).sum())  # logit = b0 + w . x in plain units
        i_ok, i_r = names.index("plan_ok"), names.index("plan_path_ratio")
        cut = -(b0 + w[i_ok]) / w[i_r] if w[i_r] != 0 else float("nan")
        p_none = 1 / (1 + math.exp(-(b0 + w[i_r] * 1.0)))
        x, yp, yv, _ = s.load_layouts(planner["after", "paper", lvl], fast_vla, names)
        use = model.use_planner(x)
        same = np.array_equal(use, fit_prob(*s.load_layouts(planner["after", "heldout", lvl], held_vla, names)[:2], x)
                              >= 0.5)
        print(f"{LEVEL_NAMES[lvl]}: trained_on {model.trained_on}; logit = {b0:.2f} + {w[i_ok]:.2f} x plan found "
              f"{w[i_r]:+.2f} x path ratio; with a plan the VLA gets the try when the ratio is above {cut:.4g}; "
              f"with no plan P(planner) = {p_none:.2f}. On the 96 paper layouts: VLA picks {int((~use).sum())}, "
              f"ASM {100 * np.where(use, yp, yv).mean():.1f}% (planner {100 * yp.mean():.1f}%, "
              f"VLA {100 * yv.mean():.1f}%, perfect {100 * np.maximum(yp, yv).mean():.1f}%); "
              f"same picks as the scorer's model: {same}")


if __name__ == "__main__":
    main()
