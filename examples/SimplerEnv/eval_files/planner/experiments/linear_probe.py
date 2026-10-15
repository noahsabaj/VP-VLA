"""Linear probe: does the VLA's first-step hidden state tell whether its episode will succeed?

Input: the policy's hidden states at the action tokens on each episode's first step (mean over the chunk,
3 depths: middle, 3/4 and last layer), saved by `--vla-hidden-dir` in VLA-only runs with the fast settings,
and the VLA outcomes of those episodes. The planned setting is the last layer with PCA to 10 numbers (fit
on the training layouts only); the other layers and sizes are shown as a check, not to pick the best.
Then the ASM's two-model rule with the probe's P(VLA succeeds), with harsh depth.

    python -m examples.SimplerEnv.eval_files.planner.experiments.linear_probe --planner 'pseed_{}_harsh' \\
        --set paper=probe_paper:fast3_train_g0,fast3_train_g1 --set heldout=probe_heldout:heldout_vla_g0,heldout_vla_g1

--set LABEL=HIDDEN:VLA_RUN,VLA_RUN...: one layout set (LABEL = paper, heldout or seed2), with the folder of its
hidden states and the VLA runs that give its labels; several runs of the same episodes give mean labels (McNemar
is then left out). --planner: the planner-only evals, a run name with {} for the LABEL.
"""

import argparse
import glob
import os

import numpy as np

from .. import selector as s
from . import runs
from .stats import auc, logloss, mcnemar

LAYER_NAMES = ["middle", "3/4", "last"]


def load_set(label, hidden_root, vla_roots):
    """(hidden [N, 3, H], VLA outcome [N], task [N], keys) for the episodes of one layout set."""
    outcome = s.load_vla(vla_roots)
    hidden, y, tasks, keys = [], [], [], []
    for f in sorted(glob.glob(os.path.join(hidden_root, "**", "*.npz"), recursive=True)):
        env, _, ep = os.path.basename(f)[:-4].rsplit("_", 2)
        key = (env, int(ep))
        if key not in outcome:
            raise SystemExit(f"{f}: no VLA outcome for {key}")
        if key in keys:
            raise SystemExit(f"{f}: a second hidden-state file for {key}")
        hidden.append(np.load(f)["hidden"])
        y.append(outcome[key])
        tasks.append(env)
        keys.append(key)
    if len(keys) < len(outcome):
        print(f"warning: {label}: {len(outcome) - len(keys)} tries have an outcome but no hidden states")
    return np.stack(hidden), np.array(y), np.array(tasks), keys


def pca(train, d):
    mean = train.mean(0)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    return lambda x: (x - mean) @ vt[:d].T


def probe(h_train, y_train, h_test, d):
    """P(VLA succeeds) on the test episodes; PCA and the model see only the training episodes."""
    f = pca(h_train, d)
    return s.LogisticRegression().fit(f(h_train), y_train).prob(f(h_test))


def probe_other_tasks(h, y, tasks, d):
    """Leave one task out: P(VLA succeeds) on each episode from a probe trained on the other tasks."""
    return s.leave_one_task_out(tasks, lambda train, test: probe(h[train], y[train], h[test], d))


def within_task_auc(p, y, t):
    return float(np.nanmean([auc(p[t == k], y[t == k]) for k in s.TASKS if (t == k).any()]))


def metrics(p, y, t, base):
    return (f"AUC {auc(p, y):.2f}, within-task AUC {within_task_auc(p, y, t):.2f}, "
            f"log-loss {logloss(p, y):.3f} (task rate {logloss(base, y):.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, dest="sets", metavar="LABEL=HIDDEN:VLA_RUN,...",
                    help="one layout set: its hidden states and VLA runs (repeat for each set)")
    ap.add_argument("--planner", required=True, help="planner-only evals: a run name with {} for the set's LABEL")
    a = ap.parse_args()
    data = {}
    for spec in a.sets:
        label, rest = spec.split("=", 1)
        hidden_root, vla = rest.split(":", 1)
        data[label] = load_set(label, runs.path(hidden_root), runs.paths(vla.split(",")))
    for label, (h, y, t, _) in data.items():
        print(f"{label}: {len(y)} tries, hidden size {h.shape[-1]}, VLA success {100 * y.mean():.1f}% ("
              + ", ".join(f"{runs.SHORT[k]} {100 * y[t == k].mean():.0f}%" for k in s.TASKS if (t == k).any()) + ")")
    H = np.concatenate([d[0] for d in data.values()])
    Y = np.concatenate([d[1] for d in data.values()])
    T = np.concatenate([d[2] for d in data.values()])
    binary = bool(np.all((Y == 0) | (Y == 1)))

    print("\n## 1. Probe alone (planned: last layer, 10 numbers)")
    tests = []
    if "heldout" in data and "paper" in data:
        tests += [("held-out -> paper", "heldout", "paper"), ("paper -> held-out", "paper", "heldout")]
    if "heldout" in data and "seed2" in data:
        tests += [("held-out -> seed 2", "heldout", "seed2")]
    for layer, d in [(2, 10), (2, 5), (2, 20), (0, 10), (1, 10)]:  # the planned one first
        lines = []
        for name, ta, tb in tests:
            ha, ya, tta, _ = data[ta]
            hb, yb, ttb, _ = data[tb]
            p = probe(ha[:, layer], ya, hb[:, layer], d)
            lines.append(f"{name}: {metrics(p, yb, ttb, s.task_rate(ya, tta, ttb))}")
        # leave one task out on every episode: the task is new, so its rate is unknown (overall rate as the base)
        p = probe_other_tasks(H[:, layer], Y, T, d)
        base = s.leave_one_task_out(T, lambda train, test: Y[train].mean())
        lines.append(f"leave one task out: {metrics(p, Y, T, base)}")
        # 8 folds inside each task: the task is seen, new layouts
        rng = np.random.default_rng(0)
        fold = np.empty(len(Y), int)
        for k in s.TASKS:
            idx = np.flatnonzero(T == k)
            fold[rng.permutation(idx)] = np.arange(len(idx)) % 8
        p, base = np.zeros(len(Y)), np.zeros(len(Y))
        for i in range(8):
            te = fold == i
            p[te] = probe(H[~te, layer], Y[~te], H[te, layer], d)
            base[te] = s.task_rate(Y[~te], T[~te], T[te])
        lines.append(f"8 folds within tasks: {metrics(p, Y, T, base)}")
        print(f"- {LAYER_NAMES[layer]} layer, {d} numbers" + ("" if (layer, d) != (2, 10) else " (planned)") + ":")
        print("\n".join(f"  - {x}" for x in lines))

    print("\n## 2. In the ASM: planner if P(planner) >= P(VLA), harsh depth, last layer, 10 numbers")
    if tests:
        names = s.FEATURES_V3
        for name, ta, tb in tests:
            pa = s.load_planner(runs.path(a.planner.format(ta)))
            pb = s.load_planner(runs.path(a.planner.format(tb)))
            ha, ya, tta, ka = data[ta]
            hb, yb, ttb, kb = data[tb]
            kb_ok = [i for i, k in enumerate(kb) if k in pb]
            ka_ok = [i for i, k in enumerate(ka) if k in pa]
            xa = np.stack([s.row_features(pa[ka[i]], names) for i in ka_ok])
            ya_plan = np.array([float(pa[ka[i]]["success"]) for i in ka_ok])
            xb = np.stack([s.row_features(pb[kb[i]], names) for i in kb_ok])
            yb_plan = np.array([float(pb[kb[i]]["success"]) for i in kb_ok])
            yb_vla, tb_ok = yb[kb_ok], ttb[kb_ok]
            p_plan = s.LogisticRegression().fit(xa, ya_plan).prob(xb)
            p_vla = probe(ha[ka_ok, 2], ya[ka_ok], hb[kb_ok, 2], 10)
            rows = {"always planner": yb_plan, "always VLA": yb_vla,
                    "P(planner) >= 0.5": np.where(p_plan >= 0.5, yb_plan, yb_vla),
                    "P(planner) >= the task's VLA rate": np.where(p_plan >= s.task_rate(ya[ka_ok], tta[ka_ok], tb_ok),
                                                                   yb_plan, yb_vla),
                    "P(planner) >= probe P(VLA)": np.where(p_plan >= p_vla, yb_plan, yb_vla),
                    "perfect": np.maximum(yb_plan, yb_vla)}
            print(f"\n{name} ({len(yb_plan)} layouts):")
            print("| Setup | " + " | ".join(runs.SHORT[k] for k in s.TASKS) + " | all |\n|---|---|---|---|---|---|")
            for r, v in rows.items():
                print(f"| {r} | " + " | ".join(f"{100 * v[tb_ok == k].mean():.1f}" for k in s.TASKS)
                      + f" | {100 * v.mean():.1f} |")
            if binary:
                for base in ("P(planner) >= 0.5", "P(planner) >= the task's VLA rate"):
                    bt, w, p = mcnemar(rows[base], rows["P(planner) >= probe P(VLA)"])
                    print(f"probe rule vs {base}: {bt} better / {w} worse, McNemar p={p:.2g}")


if __name__ == "__main__":
    main()
