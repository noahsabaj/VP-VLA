"""Two planner-only evals of the same layouts, episode by episode: outcomes and first-look features.

    python -m examples.SimplerEnv.eval_files.planner.experiments.compare_runs RUN_A:RUN_B [RUN_A:RUN_B ...]

(run names in $RUNS_DIR, or absolute paths)
"""

import argparse
import os

import numpy as np

from .. import selector as s
from . import runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", metavar="RUN_A:RUN_B")
    args = ap.parse_args()
    print("| Runs | Layouts | Same outcome | S->F | F->S | Same features (exact) | Same plan found + path ratio "
          "| Largest path-ratio change | Successes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for pair in args.pairs:
        old, new = runs.paths(pair.split(":"))
        a, b = s.load_planner(old), s.load_planner(new)
        keys = sorted(set(a) & set(b))
        if len(keys) < max(len(a), len(b)):
            print(f"warning: {pair}: {max(len(a), len(b)) - len(keys)} layouts are in one run only")
        sf = sum(bool(a[k]["success"]) and not b[k]["success"] for k in keys)
        fs = sum(not a[k]["success"] and bool(b[k]["success"]) for k in keys)
        feat = sum(np.array_equal(s.row_features(a[k], runs.CHECK), s.row_features(b[k], runs.CHECK)) for k in keys)
        va = {k: s.row_features(a[k], s.FEATURES_V3) for k in keys}
        vb = {k: s.row_features(b[k], s.FEATURES_V3) for k in keys}
        v3 = sum(np.array_equal(va[k], vb[k]) for k in keys)
        dr = max(abs(va[k][1] - vb[k][1]) for k in keys)
        print(f"| {os.path.basename(old)} -> {os.path.basename(new)} | {len(keys)} | {len(keys) - sf - fs} | {sf} | "
              f"{fs} | {feat} | {v3} | {dr:.3g} | {sum(bool(a[k]['success']) for k in keys)} -> "
              f"{sum(bool(b[k]['success']) for k in keys)} |")


if __name__ == "__main__":
    main()
