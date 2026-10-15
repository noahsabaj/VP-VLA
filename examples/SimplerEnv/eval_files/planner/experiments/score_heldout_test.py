"""The held-out test set: the ASM on the 192 layouts of layout seed 2, which were not used to choose its 2
features or to train it. Planner: the {runs}_seed2_* evals; VLA: the --vla evals (mean per layout when several).

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_heldout_test --runs pseed \\
        [--vla seed2_vla rep1_seed2 rep2_seed2]

--runs: the run-name prefix of the planner-only evals ({runs}_heldout_*, {runs}_paper_*, {runs}_seed2_*).
"""

import argparse

import numpy as np

from .. import selector as s
from . import runs
from .runs import LEVEL_NAMES, LEVELS
from .stats import mcnemar

SETUPS = ["always planner", "always VLA", "hand rule", "ASM v2", "ASM v3", "perfect"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="run-name prefix of the planner-only evals")
    ap.add_argument("--vla", nargs="+", default=runs.FAST_VLA["seed2"],
                    help="VLA-only evals of the layout-seed-2 layouts")
    a = ap.parse_args()
    held_vla, fast_vla, seed2_vla = (runs.paths(runs.FAST_VLA["heldout"]), runs.paths(runs.FAST_VLA["paper"]),
                                     runs.paths(a.vla))

    def training(lvl, names, which):
        """Training layouts: the 192 held-out layouts (the final models), or those and the 96 paper layouts."""
        parts = [s.load_layouts(runs.path(f"{a.runs}_heldout_{lvl}"), held_vla, names)]
        if which == "288":
            parts.append(s.load_layouts(runs.path(f"{a.runs}_paper_{lvl}"), fast_vla, names))
        return s.Layouts.concat(parts)

    for which in ("192", "288"):
        trained = "the 192 held-out layouts (as the final models)" if which == "192" else "all 288 layouts"
        print(f"\n### Trained on {trained}, tested on the 192 layouts of layout seed 2")
        res = {}
        for lvl in LEVELS:
            rows, picks = {}, {}
            for fs in ("v2", "v3"):
                names = s.FEATURE_SETS[fs]
                x, yp, _, _ = training(lvl, names, which)
                X, YP, YV, T = s.load_layouts(runs.path(f"{a.runs}_seed2_{lvl}"), seed2_vla, names)
                p = s.LogisticRegression().fit(x, yp).prob(X)
                if fs == "v2":
                    rows.update({"always planner": YP, "always VLA": YV,
                                 "hand rule": np.where(s.heuristic_uses_planner(X, names), YP, YV)})
                rows[f"ASM {fs}"] = np.where(p >= 0.5, YP, YV)
                picks[fs] = int((p < 0.5).sum())
            rows["perfect"] = np.maximum(rows["always planner"], rows["always VLA"])
            res[lvl] = (rows, T, picks)
        print(f"{len(res['harsh'][1])} layouts. Success rate (%):")
        print("| Setup | " + " | ".join(LEVEL_NAMES[lvl] for lvl in LEVELS) + " | harsh "
              + " / ".join(runs.SHORT.values()) + " |")
        print("|---" * (2 + len(LEVELS)) + "|")
        for name in SETUPS:
            rows_h, T_h, _ = res["harsh"]
            print(f"| {name} | " + " | ".join(f"{100 * res[lvl][0][name].mean():.1f}" for lvl in LEVELS) + " | "
                  + " / ".join(f"{100 * rows_h[name][T_h == k].mean():.1f}" for k in s.TASKS) + " |")
        print("| VLA picks v3 | " + " | ".join(str(res[lvl][2]["v3"]) for lvl in LEVELS) + " | |")
        for lvl in LEVELS:
            rows = res[lvl][0]
            if not np.all((rows["always VLA"] == 0) | (rows["always VLA"] == 1)):
                continue  # McNemar needs 0/1 outcomes, not the mean of several VLA runs
            tests = ["v3 vs {}: {} better / {} worse, p={:.2g}".format(base, *mcnemar(rows[base], rows["ASM v3"]))
                     for base in ("always planner", "always VLA", "hand rule", "ASM v2")]
            print(f"{LEVEL_NAMES[lvl]}: " + "; ".join(tests))


if __name__ == "__main__":
    main()
