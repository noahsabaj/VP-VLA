"""Adaptive selector (KAN-72): predict from the first frame whether the planner will succeed.

Training data is one row per layout: the camera features the planner logged before moving
(`planner_stats.jsonl`, under "features" and "plans"), the planner's outcome, and the VLA's
outcome on the same layout (read from SimplerEnv's video file names). The models, a small
logistic regression and gradient-boosted trees, are plain numpy, so running them needs nothing
beyond what the eval uses.

    python -m examples.SimplerEnv.eval_files.planner.selector \\
        --planner-runs <dir with <task>/planner_stats.jsonl> --vla-runs <dir with <task>/...mp4> \\
        [--save selector.json [--model logreg|trees]]

prints leave-one-task-out results for both models against "always planner", "always VLA", a hand
rule and the perfect selector, and optionally saves one model trained on everything.
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


def row_features(record: dict) -> np.ndarray:
    """Feature vector for one try, from what the planner knew before its first move."""
    f = dict(record.get("features", {}))
    first = record["plans"][0] if record.get("plans") else {}
    f["plan_ok"] = float(bool(first.get("success", False)))
    f["plan_time"] = first.get("time", 0.0)
    f["plan_ee_length"] = first.get("ee_path_length", 0.0)
    return np.array([float(f.get(k, 0.0) or 0.0) for k in FEATURES])


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


def evaluate(planner_root: str, vla_roots: List[str], save: Optional[str] = None,
             model: str = "logreg") -> None:
    planner, vla = load_planner(planner_root), load_vla(vla_roots)
    keys = [k for k in planner if k in vla]
    if not keys:
        raise SystemExit("no layouts with both planner and VLA results")
    x = np.stack([row_features(planner[k]) for k in keys])
    y_plan = np.array([float(planner[k]["success"]) for k in keys])
    y_vla = np.array([vla[k] for k in keys])
    tasks = np.array([k[0] for k in keys])

    chosen = {name: np.zeros(len(keys)) for name in MODELS}
    for task in TASKS:  # leave one task out: train on the other three, test on this one
        test = tasks == task
        if not test.any() or (~test).sum() == 0:
            continue
        for name, cls in MODELS.items():
            sel = cls().fit(x[~test], y_plan[~test])
            chosen[name][test] = np.where(sel.use_planner(x[test]), y_plan[test], y_vla[test])
    rule = (x[:, FEATURES.index("plan_ok")] > 0) & (x[:, FEATURES.index("grasp_sam3_score")] >= 0.3)
    rows = {"always planner": y_plan, "always VLA": y_vla,
            "hand rule (plan ok and SAM3 >= 0.3)": np.where(rule, y_plan, y_vla),
            "logistic regression (leave one task out)": chosen["logreg"],
            "boosted trees (leave one task out)": chosen["trees"],
            "perfect selector": np.maximum(y_plan, y_vla)}
    print(f"{len(keys)} layouts")
    print("| Setup | Spoon | Carrot | Stack | Eggplant | Avg |")
    for name, v in rows.items():
        per = [100 * v[tasks == t].mean() if (tasks == t).any() else float("nan") for t in TASKS]
        print(f"| {name} | " + " | ".join(f"{p:.1f}" for p in per) + f" | {100 * v.mean():.1f} |")
    if save:
        with open(save, "w") as fh:
            json.dump({"model": model, **MODELS[model]().fit(x, y_plan).to_json()}, fh, indent=1)
        print("saved", save)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-runs", required=True)
    ap.add_argument("--vla-runs", nargs="+", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--model", choices=sorted(MODELS), default="logreg", help="model that --save trains")
    a = ap.parse_args()
    evaluate(a.planner_runs, a.vla_runs, a.save, a.model)
