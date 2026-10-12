"""Adaptive selector: predict from the first frame whether the planner will succeed.

Training data is one row per layout: the camera features the planner logged before moving
(`planner_stats.jsonl`, under "features" and "plans"), the planner's outcome, and the VLA's
outcome on the same layout (read from SimplerEnv's video file names). The models, a small
logistic regression and gradient-boosted trees, are plain numpy, so running them needs nothing
beyond what the eval uses.

    python -m examples.SimplerEnv.eval_files.planner.selector \\
        --planner-runs <dir with <task>/planner_stats.jsonl> --vla-runs <dir with <task>/...mp4> \\
        [--save selector.json [--model logreg|trees]] \\
        [--test-planner-runs <held-out dir> --test-vla-runs <held-out VLA dir>]

prints leave-one-task-out results for both models against "always planner", "always VLA", a hand
rule and the perfect selector, and optionally saves one model trained on everything. With the
--test-* options it instead trains on all of --planner-runs and scores the held-out layouts.

Two decision rules are scored. The threshold rule uses the planner when P(planner succeeds) >= 0.5.
The two-model rule uses it when P(planner succeeds) >= P(VLA succeeds), with P(VLA succeeds)
either learned from the same features or taken as the task's VLA success rate. --save stores
only the planner model; the eval runs it with `--controller selector --selector <model.json>`.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

TASKS = ["PutSpoonOnTableClothInScene-v0", "PutCarrotOnPlateInScene-v0",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutEggplantInBasketScene-v0"]
FEATURES = ["grasp_sam3_score", "grasp_sam3_count", "grasp_mask_px", "grasp_points", "grasp_height",
            "grasp_extent_long", "grasp_extent_short", "grasp_depth_spread", "grasp_turn_rad", "grasp_reach_m",
            "place_sam3_score", "place_mask_px", "place_points", "plan_ok", "plan_time", "plan_ee_length"]
# v2: one feature per way the planner fails, with no units, so no feature says which task it is.
FEATURES_V2 = [
    "grasp_sam3_score", "place_sam3_score",            # SAM3 finds the object / the target
    "grasp_sam3_ambiguity", "place_sam3_ambiguity",    # SAM3 confuses two objects (2nd-best / best score)
    "grasp_valid_depth_frac", "place_valid_depth_frac",  # depth has holes on the object / target
    "grasp_depth_roughness_rel",                       # depth noise on the object, against its width
    "grasp_grip_margin",                               # the object fits the open hand
    "plan_ok",                                         # a path to above the object exists
    "plan_path_ratio",                                 # the path detours (planned / straight length)
    "pregrasp_corridor_points",                        # obstacle points on the straight way there
]
FEATURE_SETS = {"v1": FEATURES, "v2": FEATURES_V2}


def row_features(record: dict, names: List[str] = FEATURES) -> np.ndarray:
    """Feature vector for one try, from what the planner knew before its first move."""
    f = dict(record.get("features", {}))
    first = record["plans"][0] if record.get("plans") else {}
    f["plan_ok"] = float(bool(first.get("success", False)))
    f["plan_time"] = first.get("time", 0.0)
    f["plan_ee_length"] = first.get("ee_path_length", 0.0)
    straight = first.get("ee_straight_length", 0.0)
    f["plan_path_ratio"] = first["ee_path_length"] / straight if f["plan_ok"] and straight > 1e-3 else 1.0
    return np.array([float(f.get(k, 0.0) or 0.0) for k in names])


def load_planner(root: str) -> Dict[tuple, dict]:
    out = {}
    for task in TASKS:
        path = os.path.join(root, task, "planner_stats.jsonl")
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                out[task, r["obj_episode_id"]] = r
    return out


def load_vla(roots: List[str]) -> Dict[tuple, float]:
    """VLA success rate per (task, episode), averaged over every run folder given."""
    runs = defaultdict(list)
    for root in roots:
        for task in TASKS:
            for f in glob.glob(os.path.join(root, f"**/*{task}*/**/*.mp4"), recursive=True) + \
                     glob.glob(os.path.join(root, task, "**/*.mp4"), recursive=True):
                m = re.search(r"/(success|failure)_obj_episode_(\d+)_", f.replace("\\", "/"))
                if m:
                    runs[task, int(m.group(2)), root].append(m.group(1) == "success")
    per_layout = defaultdict(list)
    for (task, ep, _), v in runs.items():
        per_layout[task, ep].append(float(v[0]))
    return {k: float(np.mean(v)) for k, v in per_layout.items()}


class Selector:
    """Logistic regression on standardized features: P(planner succeeds)."""

    def __init__(self, l2: float = 1.0, threshold: float = 0.5):
        self.l2, self.threshold = l2, threshold

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.1) -> "Selector":
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mean) / self.std
        self.w, self.b = np.zeros(z.shape[1]), 0.0
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(z @ self.w + self.b)))
            self.w -= lr * (z.T @ (p - y) / len(y) + self.l2 * self.w / len(y))
            self.b -= lr * float(np.mean(p - y))
        return self

    def prob(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(((x - self.mean) / self.std) @ self.w + self.b)))

    def use_planner(self, x: np.ndarray) -> np.ndarray:
        return self.prob(x) >= self.threshold

    def to_json(self) -> dict:
        return {"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "w": self.w.tolist(), "b": self.b, "threshold": self.threshold}

    @classmethod
    def from_json(cls, d: dict) -> "Selector":
        s = cls(threshold=d["threshold"])
        s.mean, s.std, s.w, s.b = (np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]), d["b"])
        return s


class BoostedTrees:
    """Gradient-boosted trees on the logistic loss: P(planner succeeds).

    Each round fits a small tree to Newton steps (gradient over hessian, as in XGBoost). Leaves
    need at least `min_leaf` layouts, so a tree can't single out one layout.
    """

    def __init__(self, rounds: int = 100, depth: int = 2, lr: float = 0.1, min_leaf: int = 5,
                 l2: float = 1.0, threshold: float = 0.5):
        self.rounds, self.depth, self.lr, self.min_leaf, self.l2 = rounds, depth, lr, min_leaf, l2
        self.threshold = threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BoostedTrees":
        p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        self.base, self.trees = float(np.log(p0 / (1 - p0))), []
        f = np.full(len(y), self.base)
        for _ in range(self.rounds):
            p = 1 / (1 + np.exp(-f))
            tree = self._grow(x, p - y, p * (1 - p), self.depth)
            self.trees.append(tree)
            f += self.lr * self._tree_value(tree, x)
        return self

    def _grow(self, x: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> dict:
        leaf = {"value": float(-g.sum() / (h.sum() + self.l2))}
        n = len(g)
        if depth == 0 or n < 2 * self.min_leaf:
            return leaf
        parent, best = g.sum() ** 2 / (h.sum() + self.l2), (1e-9, None, None)
        left_n = np.arange(1, n)  # split after sorted position i puts i + 1 layouts on the left
        for j in range(x.shape[1]):
            order = np.argsort(x[:, j], kind="stable")
            xs, gl, hl = x[order, j], np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1]
            gain = gl ** 2 / (hl + self.l2) + (g.sum() - gl) ** 2 / (h.sum() - hl + self.l2) - parent
            ok = (xs[:-1] < xs[1:]) & (left_n >= self.min_leaf) & (n - left_n >= self.min_leaf)
            if ok.any() and gain[ok].max() > best[0]:
                i = np.flatnonzero(ok)[np.argmax(gain[ok])]
                best = (gain[i], j, float((xs[i] + xs[i + 1]) / 2))
        if best[1] is None:
            return leaf
        _, j, t = best
        left = x[:, j] <= t
        return {"feature": j, "threshold": t, "left": self._grow(x[left], g[left], h[left], depth - 1),
                "right": self._grow(x[~left], g[~left], h[~left], depth - 1)}

    def _tree_value(self, node: dict, x: np.ndarray) -> np.ndarray:
        if "value" in node:
            return np.full(len(x), node["value"])
        left, out = x[:, node["feature"]] <= node["threshold"], np.empty(len(x))
        out[left], out[~left] = self._tree_value(node["left"], x[left]), self._tree_value(node["right"], x[~left])
        return out

    def prob(self, x: np.ndarray) -> np.ndarray:
        f = self.base + self.lr * sum(self._tree_value(t, x) for t in self.trees)
        return 1 / (1 + np.exp(-f))

    def use_planner(self, x: np.ndarray) -> np.ndarray:
        return self.prob(x) >= self.threshold

    def to_json(self) -> dict:
        return {"features": FEATURES, "base": self.base, "lr": self.lr, "trees": self.trees,
                "threshold": self.threshold}

    @classmethod
    def from_json(cls, d: dict) -> "BoostedTrees":
        s = cls(lr=d["lr"], threshold=d["threshold"])
        s.base, s.trees = d["base"], d["trees"]
        return s


MODELS = {"logreg": Selector, "trees": BoostedTrees}


def load_saved(path: str):
    """A model written by --save, for `--controller selector`: (model, its feature names)."""
    with open(path) as fh:
        d = json.load(fh)
    return MODELS[d["model"]].from_json(d), d["features"]


def load_layouts(planner_root: str, vla_roots: List[str], names: List[str] = FEATURES):
    """Features, planner outcome, VLA outcome and task name for every layout both controllers ran."""
    planner, vla = load_planner(planner_root), load_vla(vla_roots)
    keys = [k for k in planner if k in vla]
    if not keys:
        raise SystemExit(f"no layouts with both planner and VLA results in {planner_root}")
    x = np.stack([row_features(planner[k], names) for k in keys])
    y_plan = np.array([float(planner[k]["success"]) for k in keys])
    y_vla = np.array([vla[k] for k in keys])
    return x, y_plan, y_vla, np.array([k[0] for k in keys])


def with_tasks(x: np.ndarray, tasks: np.ndarray, seen: List[str]) -> np.ndarray:
    """Add one column per training task, so a VLA model can learn each task's VLA success rate."""
    return np.hstack([x, np.stack([(tasks == k).astype(float) for k in seen], 1)])


def choose(p_plan: np.ndarray, bar, y_plan: np.ndarray, y_vla: np.ndarray) -> np.ndarray:
    """Outcome of each layout's chosen controller: the planner where p_plan >= bar, else the VLA."""
    return np.where(p_plan >= bar, y_plan, y_vla)


def print_table(x, y_plan, y_vla, tasks, chosen: Dict[str, np.ndarray], how: str,
                names: List[str] = FEATURES) -> None:
    rule = (x[:, names.index("plan_ok")] > 0) & (x[:, names.index("grasp_sam3_score")] >= 0.3)
    rows = {"always planner": y_plan, "always VLA": y_vla,
            "hand rule (plan ok and SAM3 >= 0.3)": np.where(rule, y_plan, y_vla),
            **{f"{name} ({how})": v for name, v in chosen.items()},
            "perfect selector": np.maximum(y_plan, y_vla)}
    print(f"{len(tasks)} layouts")
    print("| Setup | Spoon | Carrot | Stack | Eggplant | Avg |")
    for name, v in rows.items():
        per = [100 * v[tasks == t].mean() if (tasks == t).any() else float("nan") for t in TASKS]
        print(f"| {name} | " + " | ".join(f"{p:.1f}" for p in per) + f" | {100 * v.mean():.1f} |")


def rule_rows(p_plan: Dict[str, np.ndarray], p_vla: np.ndarray, task_rate: np.ndarray,
              y_plan: np.ndarray, y_vla: np.ndarray) -> Dict[str, np.ndarray]:
    """The two decision rules. The threshold rule treats the VLA as a coin flip on every layout;
    the two-model rule (Bayes-optimal when both estimates are right) compares the planner's
    chance with the VLA's."""
    return {"logistic regression, P >= 0.5": choose(p_plan["logreg"], 0.5, y_plan, y_vla),
            "boosted trees, P >= 0.5": choose(p_plan["trees"], 0.5, y_plan, y_vla),
            "two-model: logistic regression P >= learned P_VLA": choose(p_plan["logreg"], p_vla, y_plan, y_vla),
            "two-model: logistic regression P >= the task's VLA rate": choose(p_plan["logreg"], task_rate, y_plan, y_vla)}


def evaluate(planner_root: str, vla_roots: List[str], save: Optional[str] = None,
             model: str = "logreg", test_planner_root: Optional[str] = None,
             test_vla_roots: Optional[List[str]] = None, features: str = "v1") -> None:
    names = FEATURE_SETS[features]
    x, y_plan, y_vla, tasks = load_layouts(planner_root, vla_roots, names)

    if test_planner_root:  # held out: train on every training layout, test on layouts never seen
        tx, ty_plan, ty_vla, ttasks = load_layouts(test_planner_root, test_vla_roots or vla_roots, names)
        p_plan = {name: cls().fit(x, y_plan).prob(tx) for name, cls in MODELS.items()}
        # The tasks were seen in training, so P_VLA may use them; the task's VLA rate comes from training.
        seen = [k for k in TASKS if (tasks == k).any()]
        p_vla = Selector().fit(with_tasks(x, tasks, seen), y_vla).prob(with_tasks(tx, ttasks, seen))
        rate = {k: y_vla[tasks == k].mean() for k in seen}
        task_rate = np.array([rate.get(k, y_vla.mean()) for k in ttasks])
        print(f"trained on {len(tasks)} layouts from {planner_root}, tested on {test_planner_root}")
        print_table(tx, ty_plan, ty_vla, ttasks, rule_rows(p_plan, p_vla, task_rate, ty_plan, ty_vla),
                    "held-out layouts", names)
    else:
        p_plan = {name: np.zeros(len(tasks)) for name in MODELS}
        p_vla, task_rate = np.zeros(len(tasks)), np.zeros(len(tasks))
        for task in TASKS:  # leave one task out: train on the other three, test on this one
            test = tasks == task
            if not test.any() or (~test).sum() == 0:
                continue
            for name, cls in MODELS.items():
                p_plan[name][test] = cls().fit(x[~test], y_plan[~test]).prob(x[test])
            p_vla[test] = Selector().fit(x[~test], y_vla[~test]).prob(x[test])
            # The unseen task's VLA rate from its other layouts: what a few VLA tries on a new task give.
            v = y_vla[test]
            task_rate[test] = (v.sum() - v) / max(len(v) - 1, 1)
        print_table(x, y_plan, y_vla, tasks, rule_rows(p_plan, p_vla, task_rate, y_plan, y_vla),
                    "leave one task out", names)
    if save:
        with open(save, "w") as fh:
            json.dump({"model": model, **MODELS[model]().fit(x, y_plan).to_json(), "features": names}, fh, indent=1)
        print("saved", save)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-runs", required=True)
    ap.add_argument("--vla-runs", nargs="+", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--model", choices=sorted(MODELS), default="logreg", help="model that --save trains")
    ap.add_argument("--test-planner-runs", default=None,
                    help="held-out layouts (--layout-seed runs): train on --planner-runs, test on these")
    ap.add_argument("--test-vla-runs", nargs="+", default=None, help="VLA runs on the held-out layouts")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="v1",
                    help="v1: the first 16 features; v2: one per planner failure mode, with no units")
    a = ap.parse_args()
    if bool(a.test_planner_runs) != bool(a.test_vla_runs):
        ap.error("--test-planner-runs and --test-vla-runs go together")
    evaluate(a.planner_runs, a.vla_runs, a.save, a.model, a.test_planner_runs, a.test_vla_runs, a.features)
