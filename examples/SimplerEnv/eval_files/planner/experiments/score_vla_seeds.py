"""More seeds per episode: the VLA-only baseline runs 3 times on every episode (VLM seed offsets 0, 1, 2).

1. Label noise: how often the VLA outcome of one episode changes with the VLM seed, per set and task.
2. The ASM (saved v3 models, trained on the training set), the baselines and the oracle on the test set and the
   held-out test set, with each VLA run as the label and with the mean of the runs (the expected success over
   VLM seeds). With the mean label the oracle knows which controller is better on average, not which VLM
   sample succeeds, so it is the fair upper bound.
3. With the mean label: the ASM against each baseline, paired sign-flip test.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_vla_seeds --runs pseed

--runs: planner run-name prefix ({runs}_{heldout,paper,seed2}_*); --models: folder with sel_v3_*.json (default
final_{runs}); --reps: the extra VLA runs, {rep}_{heldout,paper,seed2} (default rep1 rep2).
"""

import argparse
import os

import numpy as np

from .. import selector as s
from . import runs
from .runs import LEVEL_NAMES, LEVELS
from .stats import fleiss_kappa, sign_flip


def vla_matrix(evals):
    """(episode keys, outcomes [episodes, runs]) for the episodes that every run has."""
    per_run = [s.load_vla(folders) for folders in evals]
    keys = sorted(set.intersection(*(set(r) for r in per_run)))
    for folders, r in zip(evals, per_run, strict=True):
        if len(r) > len(keys):
            print(f"warning: {', '.join(os.path.basename(f) for f in folders)}: {len(r) - len(keys)} episodes are "
                  "not in every run, left out")
    return keys, np.array([[r[k] for r in per_run] for k in keys])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="run-name prefix of the planner-only evals")
    ap.add_argument("--models", help="folder with the saved sel_v3_*.json models (default final_{runs})")
    ap.add_argument("--reps", nargs="+", default=["rep1", "rep2"], help="the extra VLA runs, {rep}_{layouts}")
    a = ap.parse_args()
    models = runs.path(a.models or f"final_{a.runs}")
    # set: (planner layouts, VLA runs; a run can be split over several folders)
    sets = {name: (lay, [runs.paths(runs.FAST_VLA[lay])] + [[runs.path(f"{r}_{lay}")] for r in a.reps])
            for name, lay in (("training set", "heldout"), ("test set", "paper"), ("held-out test set", "seed2"))}
    vla = {name: vla_matrix(evals) for name, (_, evals) in sets.items()}

    print(f"## 1. Label noise: the VLA-only baseline {len(a.reps) + 1} times on every episode (VLM seed offsets "
          f"0 to {len(a.reps)})")
    for name, (keys, v) in vla.items():
        n = v.shape[1]
        tasks = np.array([k[0] for k in keys])
        print(f"\n{name} ({len(keys)} episodes):")
        print("| Task | " + " | ".join(f"run {i + 1} (%)" for i in range(n)) + " | mean (%) | "
              + " | ".join(f"{i}/{n} succeed" for i in range(n + 1))
              + " | outcome changes with the seed | Fleiss' kappa |")
        print("|---" * (2 * n + 5) + "|")
        for label, m in [(runs.SHORT[t], tasks == t) for t in s.TASKS] + [("all", np.ones(len(keys), bool))]:
            k = v[m].sum(1)
            print(f"| {label} | " + " | ".join(f"{100 * v[m, i].mean():.1f}" for i in range(n))
                  + f" | {100 * v[m].mean():.1f} | " + " | ".join(str(int((k == i).sum())) for i in range(n + 1))
                  + f" | {int(((k > 0) & (k < n)).sum())} of {int(m.sum())} | {fleiss_kappa(v[m]):.2f} |")

    print("\n## 2. Success rate (%) with each VLA run as the label, and with the mean of the runs")
    results = {}
    for lvl in LEVELS:
        model, names = s.load_saved(f"{models}/sel_v3_{lvl}.json")
        for name in ("test set", "held-out test set"):
            lay = sets[name][0]
            planner = s.load_planner(runs.path(f"{a.runs}_{lay}_{lvl}"))
            keys_all, v_all = vla[name]
            idx = [i for i, k in enumerate(keys_all) if k in planner]
            if len(idx) < len(keys_all) or len(idx) < len(planner):
                print(f"warning: {a.runs}_{lay}_{lvl}: {len(keys_all) - len(idx)} VLA episodes without a planner "
                      f"episode, {len(planner) - len(idx)} planner episodes without VLA runs; both left out")
            keys, v = [keys_all[i] for i in idx], v_all[idx]
            yp = np.array([float(planner[k]["success"]) for k in keys])
            use_plan = model.use_planner(np.stack([s.row_features(planner[k], names) for k in keys]))
            heuristic = s.heuristic_uses_planner(
                np.stack([s.row_features(planner[k], s.HEURISTIC_FEATURES) for k in keys]), s.HEURISTIC_FEATURES)
            labels = [v[:, i] for i in range(v.shape[1])] + ([v.mean(1)] if v.shape[1] > 1 else [])
            rows = {"planner-only baseline": [yp] * len(labels), "VLA-only baseline": labels,
                    "heuristic baseline": [np.where(heuristic, yp, yv) for yv in labels],
                    "ASM v3": [np.where(use_plan, yp, yv) for yv in labels],
                    "oracle": [np.maximum(yp, yv) for yv in labels]}
            results[lvl, name] = rows, v.shape[1]
            print(f"\n{LEVEL_NAMES[lvl]} depth, {name} ({len(keys)} episodes; the ASM gives {int((~use_plan).sum())} "
                  "to the VLA):")
            print("| Controller | " + " | ".join(f"VLA run {i + 1}" for i in range(v.shape[1]))
                  + (" | mean of the runs |" if v.shape[1] > 1 else " |"))
            print("|---" * (len(labels) + 1) + "|")
            for r, vals in rows.items():
                print(f"| {r} | " + " | ".join(f"{100 * x.mean():.1f}" for x in vals) + " |")

    print("\n## 3. The ASM against each baseline with the mean label (points, two-sided paired sign-flip test)")
    print("| Depth | Set | " + " | ".join(f"vs {b}" for b in ("planner-only", "VLA-only", "heuristic")) + " |")
    print("|---|---|---|---|---|")
    for (lvl, name), (rows, n_runs) in results.items():
        asm = rows["ASM v3"][-1]
        cells = []
        for b in ("planner-only baseline", "VLA-only baseline", "heuristic baseline"):
            d = asm - rows[b][-1]
            cells.append(f"{100 * d.mean():+.1f} (p={sign_flip(d, n_runs):.2g})")
        print(f"| {LEVEL_NAMES[lvl]} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
