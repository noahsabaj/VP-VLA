"""The VLM plan-check v2 study (README.md, "VLM plan-check v2 study", has the plan that this scorer follows).

v2 marks the object the planner will pick up with a green outline instead of a green tint (plan_check.MARKS). Both
checks run on each first frame (PLAN_CHECK_MARKS=outline,tint), so they are compared on the same episodes: v1 is the
feature vlm_plan_yes (tint), v2 is vlm_plan_yes_outline.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_plan_check_v2 --dev --save plan_check_v2.json
    python -m examples.SimplerEnv.eval_files.planner.experiments.score_plan_check_v2 --threshold-file plan_check_v2.json

--dev: the dev step, on episodes the first study already ran (SimplerEnv's test-set layouts and the harder scenes'
fixed layouts): check2_own_paper, check2_words_{set}_paper, check2_harder_paper. It fits v2's threshold and says
whether the test runs (go); --save writes both (in $RUNS_DIR unless the path is absolute).
The test, on new layouts (layout seed 13): check2_own_iid13, check2_words_{set}_iid13, check2_harder_iid13 (test 1
needs only these), and the VLA-only vla_words_{set}_iid13 and vla_harder_iid13 (tests 2 to 4; vla_own_iid13 is
secondary).
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS

from .. import plan_check
from .. import selector as s
from . import runs
from .stats import auc, auc_ranks, bootstrap_interval, delong, mcnemar, sign_flip

SETS = ["own", *INSTRUCTION_SETS]
GROUPS = [*SETS, "harder"]
V1, V2 = plan_check.feature("tint"), plan_check.feature("outline")
TEST_LAYOUTS = "iid13"


def planner_run(group: str, layouts: str) -> str:
    return f"check2_{group if group in ('own', 'harder') else 'words_' + group}_{layouts}"


def vla_run(group: str, layouts: str) -> str:
    return f"vla_{group if group in ('own', 'harder') else 'words_' + group}_{layouts}"


def group_tasks(group: str) -> List[str]:
    return list(hs.TASKS) if group == "harder" else list(s.TASKS)


def load_group(group: str, layouts: str) -> Dict[tuple, dict]:
    """(task, episode) -> the planner record, its outcome and both checks' P(yes)."""
    run = planner_run(group, layouts)
    out = {}
    for k, r in s.load_planner(runs.path(run), group_tasks(group)).items():
        f = r.get("features", {})
        if V1 not in f or V2 not in f:
            raise SystemExit(f"{run}: {k} lacks {V1} or {V2} (run with PLAN_CHECK=1 PLAN_CHECK_MARKS=outline,tint)")
        out[k] = {"record": r, "planner": float(r["success"]), "v1": float(f[V1]), "v2": float(f[V2])}
    if not out:
        raise SystemExit(f"{run}: no planner records")
    return out


def pooled(groups: Dict[str, Dict[tuple, dict]]) -> Dict[str, np.ndarray]:
    """v1, v2, planner (0/1) and the stratum (group/task) of every episode of every group."""
    rows = [(g, k, r) for g, rs in groups.items() for k, r in sorted(rs.items())]
    out = {c: np.array([r[c] for _, _, r in rows]) for c in ("v1", "v2", "planner")}
    out["strata"] = np.array([f"{g}/{k[0]}" for g, k, _ in rows])
    return out


def fit_threshold(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """The threshold t (planner when P(yes) >= t) with the largest Youden's J = (share of planner successes with a
    yes) + (share of planner failures with a no) - 1, among the P(yes) values seen; ties: the t nearest 0.5.
    Returns (t, J)."""
    y = y.astype(bool)
    if y.all() or not y.any():
        raise ValueError("the threshold needs planner successes and failures")
    best = None
    for t in np.unique(p):
        j = (p[y] >= t).mean() + (p[~y] < t).mean() - 1
        key = (round(j, 12), -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), float(j))
    return best[1], best[2]


def auc_difference_interval(v1: np.ndarray, v2: np.ndarray, y: np.ndarray, strata: np.ndarray, draws: int = 10000,
                            seed: int = 0) -> Tuple[float, float]:
    """95% percentile bootstrap interval of AUC(v2) - AUC(v1): each draw takes each stratum's episodes again, with
    replacement."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == k) for k in sorted(set(strata.tolist()))]
    d = np.empty(draws)
    for i in range(draws):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        d[i] = auc_ranks(v2[idx], y[idx]) - auc_ranks(v1[idx], y[idx])
    lo, hi = np.quantile(d, [0.025, 0.975])
    return float(lo), float(hi)


def describe(groups: Dict[str, Dict[tuple, dict]]) -> None:
    print("| Instructions | Episodes | Planner-only | AUC v1 (tint) | AUC v2 (outline) | Yes v1 | Yes v2 |\n"
          "|---|---|---|---|---|---|---|")
    for g, rows in groups.items():
        c = pooled({g: rows})
        y = c["planner"]
        print(f"| {g} | {len(y)} | {100 * y.mean():.1f} | {auc(c['v1'], y):.2f} | {auc(c['v2'], y):.2f} | "
              f"{int((c['v1'] >= 0.5).sum())} | {int((c['v2'] >= 0.5).sum())} |")


def dev(save: Optional[str]) -> None:
    groups = {g: load_group(g, "paper") for g in GROUPS}
    print("# VLM plan check v2: the dev step (episodes the first study already ran)\n")
    describe(groups)
    c = pooled(groups)
    a1, a2, p = delong(c["v1"], c["v2"], c["planner"])
    t, j = fit_threshold(c["v2"], c["planner"])
    go = a2 > a1
    print(f"\nAll {len(c['planner'])} episodes: AUC v1 {a1:.3f}, v2 {a2:.3f} (DeLong p = {p:.2g}, exploratory)")
    print(f"v2 threshold (largest Youden's J on these episodes): {t:.6g} (J = {j:.3f})")
    print(f"Go: {'yes' if go else 'no'} (the test runs only if v2's AUC is higher than v1's here)")
    if save:
        path = runs.path(save)
        with open(path, "w") as fh:
            json.dump({"threshold": t, "youden_j": j, "dev_auc_v1": a1, "dev_auc_v2": a2, "go": go,
                       "dev_runs": [planner_run(g, "paper") for g in GROUPS]}, fh, indent=1)
        print(f"saved {path}")


def gate_columns(rows: Dict[tuple, dict], vla: Dict[tuple, float], keys: List[tuple], t: float) -> Dict[str, np.ndarray]:
    out = {"planner": np.array([rows[k]["planner"] for k in keys]), "vla": np.array([vla[k] for k in keys])}
    for name, col, thr in (("gate", "v2", t), ("gate_v1", "v1", plan_check.THRESHOLD)):
        out[name] = np.array([vla[k] if plan_check.gate(rows[k][col], thr) == "vla" else rows[k]["planner"]
                              for k in keys])
    out["picks"] = np.array([rows[k]["v2"] < t for k in keys])
    return out


def load_vla(group: str) -> Optional[Dict[tuple, float]]:
    path = runs.path(vla_run(group, TEST_LAYOUTS))
    return s.load_vla([path], group_tasks(group)) if os.path.isdir(path) else None


def tests_2_to_4(groups, vlas, t: float) -> list:
    """Tests 2 to 4's per-layout values, in their fixed order: (what, values, strata)."""
    hk = sorted(k for k in groups["harder"] if k in vlas["harder"])
    h = gate_columns(groups["harder"], vlas["harder"], hk, t)
    wk = sorted(set.intersection(*({k for k in groups[n] if k in vlas[n]} for n in INSTRUCTION_SETS)))
    w = [gate_columns(groups[n], vlas[n], wk, t) for n in INSTRUCTION_SETS]
    reworded = sum(c["gate"] - c["planner"] for c in w)
    strata_h, strata_w = np.array([k[0] for k in hk]), np.array([k[0] for k in wk])
    return [("on the harder scenes the v2 gate beats planner-only", h["gate"] - h["planner"], strata_h),
            ("on the harder scenes the v2 gate beats VLA-only", h["gate"] - h["vla"], strata_h),
            ("on the reworded instructions the v2 gate beats planner-only (sum over the 4 sets)", reworded, strata_w)]


def test(threshold_file: str) -> None:
    with open(runs.path(threshold_file)) as fh:
        dev_result = json.load(fh)
    if not dev_result["go"]:
        raise SystemExit(f"{threshold_file}: the dev step said no go, so the plan runs no test")
    t = dev_result["threshold"]
    groups = {g: load_group(g, TEST_LAYOUTS) for g in GROUPS}
    print(f"# VLM plan check v2: the test ({TEST_LAYOUTS} layouts; v2 threshold {t:.6g} from the dev step)\n")
    describe(groups)
    c = pooled(groups)
    a1, a2, p = delong(c["v1"], c["v2"], c["planner"])
    lo, hi = auc_difference_interval(c["v1"], c["v2"], c["planner"], c["strata"])
    print("\nPlanned tests (fixed order, each two-sided at 0.05; each counts only if every test before it was "
          "significant):")
    ok = p < 0.05 and a2 > a1
    print(f"1. v2 predicts planner success better than v1 ({len(c['planner'])} episodes): AUC {a2:.3f} vs {a1:.3f}, "
          f"{a2 - a1:+.3f} (95% interval {lo:+.3f} to {hi:+.3f}), DeLong p = {p:.2g}: "
          f"{'significant' if ok else 'not significant'}")
    vlas = {g: load_vla(g) for g in GROUPS}
    missing = [vla_run(g, TEST_LAYOUTS) for g in ["harder", *INSTRUCTION_SETS] if vlas[g] is None]
    if missing:
        print(f"Tests 2 to 4 wait for the VLA-only evals: {', '.join(missing)}")
    else:
        for i, (what, d, strata) in enumerate(tests_2_to_4(groups, vlas, t), 2):
            sp = sign_flip(d, 1)
            blo, bhi = bootstrap_interval(d, strata)
            verdict = (": significant" if sp < 0.05 else ": not significant") if ok else \
                ": does not count (a test before it was not significant)"
            ok = ok and sp < 0.05
            print(f"{i}. {what}: {100 * d.mean():+.1f} points (95% interval {100 * blo:+.1f} to {100 * bhi:+.1f}), "
                  f"{int((d > 0).sum())} layouts up / {int((d < 0).sum())} down, exact p = {sp:.2g}{verdict}")

    print("\nSecondary (no claims):")
    for g, rows in groups.items():
        tasks = sorted({k[0] for k in rows})
        w1 = [auc(np.array([rows[k]["v1"] for k in rows if k[0] == tk]),
                  np.array([rows[k]["planner"] for k in rows if k[0] == tk])) for tk in tasks]
        w2 = [auc(np.array([rows[k]["v2"] for k in rows if k[0] == tk]),
                  np.array([rows[k]["planner"] for k in rows if k[0] == tk])) for tk in tasks]
        mass = np.array([r["record"].get("plan_check_answer_mass_outline") or np.nan for r in rows.values()])
        line = (f"- {g}: mean within-task AUC v1 {np.nanmean(w1):.2f}, v2 {np.nanmean(w2):.2f}; v2's probability on "
                f"yes/no as its first word: min {np.nanmin(mass):.3f}")
        if vlas[g] is not None:
            keys = sorted(k for k in rows if k in vlas[g])
            gc = gate_columns(rows, vlas[g], keys, t)
            _, _, pp = mcnemar(gc["planner"], gc["gate"])
            _, _, pv = mcnemar(gc["vla"], gc["gate"])
            line += (f"; planner-only {100 * gc['planner'].mean():.1f}, VLA-only {100 * gc['vla'].mean():.1f}, "
                     f"v2 gate {100 * gc['gate'].mean():.1f} (vs planner p = {pp:.2g}, vs VLA p = {pv:.2g}), "
                     f"v1 gate {100 * gc['gate_v1'].mean():.1f}, oracle "
                     f"{100 * np.maximum(gc['planner'], gc['vla']).mean():.1f}, "
                     f"v2 gives the VLA {int(gc['picks'].sum())}")
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="the dev step (threshold and go/no-go)")
    ap.add_argument("--save", default=None, help="with --dev: write the threshold and the go/no-go here")
    ap.add_argument("--threshold-file", default="plan_check_v2.json", help="the dev step's saved result")
    a = ap.parse_args()
    if a.dev:
        dev(a.save)
    else:
        test(a.threshold_file)


if __name__ == "__main__":
    main()
