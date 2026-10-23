"""Adaptive selector (ASM): from the first frame, give each episode to the planner or to the VLA.

Training data is one row per layout: the camera features the planner logged before moving
(`planner_stats.jsonl`, under "features" and "plans"), the planner's outcome, and the VLA's
outcome on the same layout (read from SimplerEnv's video file names). The models, a small
logistic regression and gradient-boosted trees, are plain numpy, so running them needs nothing
beyond what the eval uses.

    python -m examples.SimplerEnv.eval_files.planner.selector \\
        --planner-runs <dir with <task>/planner_stats.jsonl> --vla-runs <dir with <task>/...mp4> \\
        [--features v1|v2|v3] [--save selector.json [--model logreg|trees]] \\
        [--test-planner-runs <held-out dir> --test-vla-runs <held-out VLA dir>]

prints leave-one-task-out results for both models against "always planner", "always VLA", a hand
rule and the perfect selector, and optionally saves one model trained on everything. With the
--test-* options it instead trains on all of --planner-runs and scores the held-out layouts.

Two decision rules are scored. The threshold rule uses the planner when P(planner succeeds) >= 0.5.
The two-model rule uses it when P(planner succeeds) >= P(VLA succeeds), with P(VLA succeeds)
either learned from the same features or taken as the task's VLA success rate. --save stores
only the planner model; the eval runs it with `--controller selector --selector <model.json>`.

    python -m examples.SimplerEnv.eval_files.planner.selector --rule two-model \\
        --planner-runs <training set> --vla-runs <its VLA runs> --save asm.json

trains and saves the two-model ASM (TwoModelASM): one success model per method, each a logistic
regression on the task and its method's first-frame features (PLANNER_FEATURES, VLA_FEATURES), and
the planner when P(planner succeeds) >= P(VLA succeeds). experiments/score_two_model.py scores it.
With --planner-candidates (feature lists, each comma-separated), P(planner) uses the list with the
lowest log-loss in 10-fold cross-validation on the training set (choose_features). Episodes that the
VLM plan gives to the VLA before the ASM decides (vlm_plan_gate) are not training data.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

TASKS = ["PutSpoonOnTableClothInScene-v0", "PutCarrotOnPlateInScene-v0",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutEggplantInBasketScene-v0"]
# v1: the first 16 features.
FEATURES_V1 = ["grasp_sam3_score", "grasp_sam3_count", "grasp_mask_px", "grasp_points", "grasp_height",
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
# v3: only the two plan features. With harsh depth they are the only v2 features with signal; the other
# nine score near chance alone, and dropping any one of them moves the selector by about 1 layout.
FEATURES_V3 = ["plan_ok", "plan_path_ratio"]
FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2, "v3": FEATURES_V3}
# The two-model ASM's success models, each with one column per task as well. The planner fails when it finds no
# path to above the object, or a long detour (v3). The VLA fails when the grasp needs a large turn of the wrist
# (on the test set: spoon 100% at 0 deg, 33% at 90 deg; carrot 92% at 180 deg, 3% at -90 deg) or the object is
# far from the hand. Both come from the planner's first look at the object.
PLANNER_FEATURES = FEATURES_V3
VLA_FEATURES = ["grasp_turn_rad", "grasp_reach_m"]
# The hand rule (the heuristic baseline) needs these features; v3 leaves out the SAM3 score.
HEURISTIC_FEATURES = ["plan_ok", "grasp_sam3_score"]
# Written into every planner record since the first-frame fix. Older records found the place target
# only after a successful grasp, so their place features give the outcome away.
FEATURES_VERSION = 2
# Settings a folder's runs must share; --save stores them and the eval checks them.
RUN_SETTINGS = ("depth_noise", "rgb_noise", "noise_seed", "robot_mask", "layout_seed", "planner_seed",
                "layout_per_episode", "task_source", "vlm_plan_min_score", "sam_model", "plan_check_marks")


def row_features(record: dict, names: List[str]) -> np.ndarray:
    """Feature vector for one try, from what the planner knew before its first move."""
    f = dict(record.get("features", {}))
    first = record["plans"][0] if record.get("plans") else {}
    f["plan_ok"] = float(bool(first.get("success", False)))
    f["plan_time"] = first.get("time", 0.0)
    f["plan_ee_length"] = first.get("ee_path_length", 0.0)
    straight = first.get("ee_straight_length", 0.0)
    f["plan_path_ratio"] = first["ee_path_length"] / straight if f["plan_ok"] and straight > 1e-3 else 1.0
    return np.array([float(f.get(k, 0.0) or 0.0) for k in names])


def heuristic_uses_planner(x: np.ndarray, names: List[str]) -> np.ndarray:
    """The hand rule: the planner when it found a plan and SAM3's best score for the object is at least 0.3."""
    return (x[:, names.index("plan_ok")] > 0) & (x[:, names.index("grasp_sam3_score")] >= 0.3)


def planner_records(root: str, tasks: Optional[List[str]] = None) -> List[dict]:
    """Every record under root (`<task>/planner_stats.jsonl`, or run_eval.sh's `*<task>*/planner_stats.jsonl`),
    checked so that a wrong join stops the load: planner-only mode, written since the first-frame
    fix, one record per layout, and one value of each run setting in the folder. tasks: default TASKS."""
    records, seen = [], set()
    for task in tasks or TASKS:
        paths = {os.path.normpath(p) for p in [os.path.join(root, task, "planner_stats.jsonl"),
                                               *glob.glob(os.path.join(root, f"**/*{task}*/planner_stats.jsonl"),
                                                          recursive=True)]}
        for path in sorted(p for p in paths if os.path.exists(p)):
            with open(path) as fh:
                lines = list(fh)
            for n, line in enumerate(lines, 1):
                r, where = json.loads(line), f"{path}:{n}"
                if r.get("mode") != "planner":  # a selector or hand-off try's success is not the planner's
                    raise SystemExit(f"{where}: mode {r.get('mode')!r}; selector data must come from planner-only runs")
                f = r.get("features", {})
                if "features_version" not in r and "grasp_sam3_score" in f and "grasp_valid_depth_frac" not in f:
                    raise SystemExit(f"{where}: written before the first-frame fix, so its place features leak "
                                     "the outcome")
                if (task, r["obj_episode_id"]) in seen:
                    raise SystemExit(f"{where}: a second record for {task} episode {r['obj_episode_id']}")
                seen.add((task, r["obj_episode_id"]))
                records.append({**r, "task_folder": task})
    planner_settings(records, root)
    return records


def planner_settings(records: List[dict], root: str = "") -> dict:
    """The run settings the records share (settings older logs did not write are left out)."""
    out = {}
    for name in RUN_SETTINGS:
        values = {json.dumps(r[name]) for r in records if name in r}
        if len(values) > 1:
            raise SystemExit(f"{root}: runs with different {name} ({', '.join(sorted(values))}) in one folder")
        if values:
            out[name] = json.loads(values.pop())
    return out


def load_planner(root: str, tasks: Optional[List[str]] = None) -> Dict[tuple, dict]:
    return {(r["task_folder"], r["obj_episode_id"]): r for r in planner_records(root, tasks)}


VIDEO = re.compile(r"/(success|failure)_obj_episode_(\d+)(?:_[^/]*)?\.mp4$")


def load_vla(roots: List[str], tasks: Optional[List[str]] = None) -> Dict[tuple, float]:
    """VLA success rate per (task, episode), averaged over the run folders given. Inside one folder a
    layout must have one outcome: a success and a failure video for it (a rerun, or a planner run
    written to the same folder) stop the load instead of one of them being picked at random. tasks: default
    TASKS."""
    per_layout = defaultdict(list)
    for root in roots:
        outcomes = defaultdict(set)
        for task in tasks or TASKS:
            files = set(glob.glob(os.path.join(root, f"**/*{task}*/**/*.mp4"), recursive=True)) | \
                    set(glob.glob(os.path.join(root, task, "**/*.mp4"), recursive=True))
            for f in files:
                m = VIDEO.search(f.replace("\\", "/"))
                if m:
                    outcomes[task, int(m.group(2))].add(m.group(1) == "success")
        for (task, ep), v in outcomes.items():
            if len(v) > 1:
                raise SystemExit(f"{root}: {task} episode {ep} has both a success and a failure video")
            per_layout[task, ep].append(float(v.pop()))
    return {k: float(np.mean(v)) for k, v in per_layout.items()}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class Classifier:
    """P(y = 1) from the features: with y = the planner's success, the planner gets the try when P >= threshold.
    `trained_on` holds the run settings of the training data (load_saved fills it in), which the eval checks."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.trained_on: dict = {}

    def prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def use_planner(self, x: np.ndarray) -> np.ndarray:
        return self.prob(x) >= self.threshold


class LogisticRegression(Classifier):
    """Logistic regression on standardized features, with an L2 penalty, fit by gradient descent."""

    def __init__(self, l2: float = 1.0, threshold: float = 0.5):
        super().__init__(threshold)
        self.l2 = l2

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.1) -> "LogisticRegression":
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mean) / self.std
        self.w, self.b = np.zeros(z.shape[1]), 0.0
        for _ in range(steps):
            p = sigmoid(z @ self.w + self.b)
            self.w -= lr * (z.T @ (p - y) / len(y) + self.l2 * self.w / len(y))
            self.b -= lr * float(np.mean(p - y))
        return self

    def prob(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(((x - self.mean) / self.std) @ self.w + self.b)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "w": self.w.tolist(), "b": self.b,
                "threshold": self.threshold}

    @classmethod
    def from_json(cls, d: dict) -> "LogisticRegression":
        s = cls(threshold=d["threshold"])
        s.mean, s.std, s.w, s.b = (np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]), d["b"])
        return s


class BoostedTrees(Classifier):
    """Gradient-boosted trees on the logistic loss.

    Each round fits a small tree to Newton steps (gradient over hessian, as in XGBoost). Leaves
    need at least `min_leaf` layouts, so a tree can't single out one layout.
    """

    def __init__(self, rounds: int = 100, depth: int = 2, lr: float = 0.1, min_leaf: int = 5,
                 l2: float = 1.0, threshold: float = 0.5):
        super().__init__(threshold)
        self.rounds, self.depth, self.lr, self.min_leaf, self.l2 = rounds, depth, lr, min_leaf, l2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BoostedTrees":
        p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        self.base, self.trees = float(np.log(p0 / (1 - p0))), []
        f = np.full(len(y), self.base)
        for _ in range(self.rounds):
            p = sigmoid(f)
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
        return sigmoid(self.base + self.lr * sum(self._tree_value(t, x) for t in self.trees))

    def to_json(self) -> dict:
        return {"base": self.base, "lr": self.lr, "trees": self.trees, "threshold": self.threshold}

    @classmethod
    def from_json(cls, d: dict) -> "BoostedTrees":
        s = cls(lr=d["lr"], threshold=d["threshold"])
        s.base, s.trees = d["base"], d["trees"]
        return s


MODELS = {"logreg": LogisticRegression, "trees": BoostedTrees}


def load_saved(path: str) -> Tuple[Classifier, List[str]]:
    """A planner model written by --save: (model, its feature names)."""
    with open(path) as fh:
        d = json.load(fh)
    model = MODELS[d["model"]].from_json(d)
    model.trained_on = d.get("trained_on") or {}
    return model, d["features"]


class SuccessModel:
    """P(one method succeeds) from first-frame features: a logistic regression on the features and one column per
    task (so each task has its own base rate; the tasks are the same in training and in use)."""

    def __init__(self, features: List[str], l2: float = 1.0):
        self.features = list(features)
        self.model = LogisticRegression(l2=l2)

    def x(self, records: List[dict]) -> np.ndarray:
        return np.stack([row_features(r, self.features) for r in records])

    def fit(self, x: np.ndarray, tasks: np.ndarray, y: np.ndarray) -> "SuccessModel":
        self.model.fit(with_tasks(x, np.asarray(tasks), TASKS), y)
        return self

    def prob(self, x: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        return self.model.prob(with_tasks(x, np.asarray(tasks), TASKS))

    def plain(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """The model in the features' own units: logit P = intercept[task] + sum of slope[feature] * feature."""
        m, k = self.model, len(self.features)
        slope = m.w / m.std
        base = m.b - float(slope @ m.mean)
        return ({t: base + float(slope[k + i]) for i, t in enumerate(TASKS)},
                {f: float(slope[i]) for i, f in enumerate(self.features)})

    def to_json(self) -> dict:
        return {"features": self.features, "tasks": TASKS, **self.model.to_json()}

    @classmethod
    def from_json(cls, d: dict) -> "SuccessModel":
        if d["tasks"] != TASKS:
            raise ValueError(f"saved with the task columns {d['tasks']}, not {TASKS}")
        m = cls(d["features"])
        m.model = LogisticRegression.from_json(d)
        return m


class ThresholdASM:
    """The first ASM: the planner when P(planner succeeds) >= the model's threshold (0.5), else the VLA. It has no
    estimate of P(VLA succeeds), so it treats the VLA as a coin flip in every episode."""

    def __init__(self, model: Classifier, features: List[str]):
        self.model, self.features, self.trained_on = model, features, model.trained_on

    def decide(self, record: dict, task: str) -> dict:
        p = float(self.model.prob(row_features(record, self.features)[None])[0])
        return {"p_planner": p, "choice": "planner" if p >= self.model.threshold else "vla"}


class TwoModelASM:
    """The ASM with a success model for each method: the planner when P(planner succeeds) >= P(VLA succeeds), else
    the VLA. When both probabilities are right, no other choice gives a higher expected success."""

    def __init__(self, planner: SuccessModel, vla: SuccessModel, trained_on: Optional[dict] = None):
        self.planner, self.vla, self.trained_on = planner, vla, trained_on or {}

    def probs(self, records: List[dict], tasks) -> Tuple[np.ndarray, np.ndarray]:
        tasks = np.asarray(tasks)
        return self.planner.prob(self.planner.x(records), tasks), self.vla.prob(self.vla.x(records), tasks)

    def decide(self, record: dict, task: str) -> dict:
        p_plan, p_vla = (float(p[0]) for p in self.probs([record], [task]))
        return {"p_planner": p_plan, "p_vla": p_vla, "choice": "planner" if p_plan >= p_vla else "vla"}

    def to_json(self) -> dict:
        return {"rule": "two-model", "planner": self.planner.to_json(), "vla": self.vla.to_json(),
                "trained_on": self.trained_on}

    @classmethod
    def from_json(cls, d: dict) -> "TwoModelASM":
        return cls(SuccessModel.from_json(d["planner"]), SuccessModel.from_json(d["vla"]), d.get("trained_on"))


def load_asm(path: str):
    """The ASM in a file written by --save, for `--controller selector`: a TwoModelASM (--rule two-model), or a
    ThresholdASM for a planner model."""
    with open(path) as fh:
        d = json.load(fh)
    return TwoModelASM.from_json(d) if d.get("rule") == "two-model" else ThresholdASM(*load_saved(path))


class Layouts(NamedTuple):
    """One row per layout that both controllers ran."""

    x: np.ndarray       # features
    y_plan: np.ndarray  # planner success (0/1)
    y_vla: np.ndarray   # VLA success: 0/1 with one VLA run, the mean over the runs with several
    tasks: np.ndarray   # task names

    @staticmethod
    def concat(parts: List["Layouts"]) -> "Layouts":
        return Layouts(*[np.concatenate(z) for z in zip(*parts, strict=True)])


def matched_layouts(planner_root: str, vla_roots: List[str],
                    tasks: Optional[List[str]] = None) -> Tuple[Dict[tuple, dict], Dict[tuple, float], list]:
    """(planner records, VLA success rates, the keys of the layouts both controllers ran); the other layouts are
    left out, with a warning. tasks: default TASKS."""
    planner, vla = load_planner(planner_root, tasks), load_vla(vla_roots, tasks)
    keys = [k for k in planner if k in vla]
    if not keys:
        raise SystemExit(f"no layouts with both planner and VLA results in {planner_root}")
    unmatched = {"planner only": [k for k in planner if k not in vla], "VLA only": [k for k in vla if k not in planner]}
    for side, ks in unmatched.items():
        if ks:  # left out of every score, so say so
            count = Counter(k[0] for k in ks)
            per_task = ", ".join(f"{t.split('-')[0]} {count[t]}" for t in tasks or TASKS if count[t])
            print(f"warning: {len(ks)} layouts {side} ({per_task}), left out: {planner_root} / {', '.join(vla_roots)}")
    return planner, vla, keys


def load_layouts(planner_root: str, vla_roots: List[str], names: List[str]) -> Layouts:
    """Features, planner outcome, VLA outcome and task name for every layout both controllers ran."""
    planner, vla, keys = matched_layouts(planner_root, vla_roots)
    x = np.stack([row_features(planner[k], names) for k in keys])
    y_plan = np.array([float(planner[k]["success"]) for k in keys])
    y_vla = np.array([vla[k] for k in keys])
    return Layouts(x, y_plan, y_vla, np.array([k[0] for k in keys]))


def leave_one_task_out(tasks: np.ndarray, predict: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """predict(train, test) once per task, with test = that task's rows and train = all other rows (boolean
    masks): every row's prediction comes from a model that never saw its task."""
    out = np.zeros(len(tasks))
    for task in TASKS:
        test = tasks == task
        if test.any():
            out[test] = predict(~test, test)
    return out


def task_rate(y: np.ndarray, tasks: np.ndarray, test_tasks: np.ndarray) -> np.ndarray:
    """For each test row, the success rate of its task in (y, tasks); the overall rate for a task not there."""
    rate = {k: y[tasks == k].mean() for k in TASKS if (tasks == k).any()}
    return np.array([rate.get(k, y.mean()) for k in test_tasks])


def other_layouts_rate(v: np.ndarray) -> np.ndarray:
    """Each layout's rate from the other layouts of its task: what a few VLA tries on a new task give."""
    return (v.sum() - v) / max(len(v) - 1, 1)


def with_tasks(x: np.ndarray, tasks: np.ndarray, seen: List[str]) -> np.ndarray:
    """Add one column per training task, so a VLA model can learn each task's VLA success rate."""
    return np.hstack([x, np.stack([(tasks == k).astype(float) for k in seen], 1)])


def choose(p_plan: np.ndarray, bar, y_plan: np.ndarray, y_vla: np.ndarray) -> np.ndarray:
    """Outcome of each layout's chosen controller: the planner where p_plan >= bar, else the VLA."""
    return np.where(p_plan >= bar, y_plan, y_vla)


def print_table(x, y_plan, y_vla, tasks, chosen: Dict[str, np.ndarray], how: str, names: List[str]) -> None:
    rows = {"always planner": y_plan, "always VLA": y_vla}
    if all(n in names for n in HEURISTIC_FEATURES):
        rows["hand rule (plan ok and SAM3 >= 0.3)"] = np.where(heuristic_uses_planner(x, names), y_plan, y_vla)
    rows.update({f"{name} ({how})": v for name, v in chosen.items()})
    rows["perfect selector"] = np.maximum(y_plan, y_vla)
    print(f"{len(tasks)} layouts")
    print("| Setup | Spoon | Carrot | Stack | Eggplant | Avg |")
    for name, v in rows.items():
        per = [100 * v[tasks == t].mean() if (tasks == t).any() else float("nan") for t in TASKS]
        print(f"| {name} | " + " | ".join(f"{p:.1f}" for p in per) + f" | {100 * v.mean():.1f} |")


def rule_rows(p_plan: Dict[str, np.ndarray], p_vla: np.ndarray, task_rates: np.ndarray,
              y_plan: np.ndarray, y_vla: np.ndarray) -> Dict[str, np.ndarray]:
    """The two decision rules. The threshold rule treats the VLA as a coin flip on every layout;
    the two-model rule (Bayes-optimal when both estimates are right) compares the planner's
    chance with the VLA's."""
    return {"logistic regression, P >= 0.5": choose(p_plan["logreg"], 0.5, y_plan, y_vla),
            "boosted trees, P >= 0.5": choose(p_plan["trees"], 0.5, y_plan, y_vla),
            "two-model: logistic regression P >= learned P_VLA": choose(p_plan["logreg"], p_vla, y_plan, y_vla),
            "two-model: logistic regression P >= the task's VLA rate": choose(p_plan["logreg"], task_rates, y_plan,
                                                                               y_vla)}


def evaluate(planner_root: str, vla_roots: List[str], save: Optional[str] = None,
             model: str = "logreg", test_planner_root: Optional[str] = None,
             test_vla_roots: Optional[List[str]] = None, features: str = "v3") -> None:
    names = FEATURE_SETS[features]
    x, y_plan, y_vla, tasks = load_layouts(planner_root, vla_roots, names)

    if test_planner_root:  # held out: train on every training layout, test on layouts never seen
        tx, ty_plan, ty_vla, ttasks = load_layouts(test_planner_root, test_vla_roots or vla_roots, names)
        p_plan = {name: cls().fit(x, y_plan).prob(tx) for name, cls in MODELS.items()}
        # The tasks were seen in training, so P_VLA may use them; the task's VLA rate comes from training.
        seen = [k for k in TASKS if (tasks == k).any()]
        p_vla = LogisticRegression().fit(with_tasks(x, tasks, seen), y_vla).prob(with_tasks(tx, ttasks, seen))
        print(f"trained on {len(tasks)} layouts from {planner_root}, tested on {test_planner_root}")
        print_table(tx, ty_plan, ty_vla, ttasks,
                    rule_rows(p_plan, p_vla, task_rate(y_vla, tasks, ttasks), ty_plan, ty_vla),
                    "held-out layouts", names)
    else:
        if len(set(tasks)) < 2:  # every fold would be skipped and the table would show zeros as results
            raise SystemExit(f"leave one task out needs at least 2 tasks; {planner_root} has {len(set(tasks))}")

        def fit_prob(cls, y):  # leave one task out: train on the other three tasks, test on this one
            return leave_one_task_out(tasks, lambda train, test: cls().fit(x[train], y[train]).prob(x[test]))

        p_plan = {name: fit_prob(cls, y_plan) for name, cls in MODELS.items()}
        p_vla = fit_prob(LogisticRegression, y_vla)
        # The unseen task's VLA rate from its other layouts.
        rates = leave_one_task_out(tasks, lambda train, test: other_layouts_rate(y_vla[test]))
        print_table(x, y_plan, y_vla, tasks, rule_rows(p_plan, p_vla, rates, y_plan, y_vla),
                    "leave one task out", names)
    if save:
        with open(save, "w") as fh:
            json.dump({"model": model, "features": names, **MODELS[model]().fit(x, y_plan).to_json(),
                       "trained_on": planner_settings(planner_records(planner_root), planner_root)}, fh, indent=1)
        print("saved", save)


def vlm_plan_gate(record: dict) -> Optional[dict]:
    """The selector mode's choice before the ASM (planner_client.py): the VLA when the VLM plan found the instruction
    not to be pick-and-place. None when the ASM decides (always with the fixed rule)."""
    if record.get("task_source") == "vlm" and not record.get("vlm_plan_feasible"):
        return {"choice": "vla", "reason": "VLM plan: not pick-and-place"}
    return None


def grouped_folds(episodes: np.ndarray, k: int, seed: int) -> np.ndarray:
    """A fold (0 to k - 1) for each row, with all the rows of an episode index in one fold. In the iidN sets the tasks
    of an episode index share their layout (spoon and carrot have the same position), so they are held out together."""
    order = np.random.default_rng(seed).permutation(sorted(set(np.asarray(episodes).tolist())))
    fold = {e: i % k for i, e in enumerate(order.tolist())}
    return np.array([fold[e] for e in np.asarray(episodes).tolist()])


def cv_logloss(records: List[dict], tasks: np.ndarray, y: np.ndarray, features: List[str], episodes: np.ndarray,
               k: int = 10, seed: int = 0) -> float:
    """Log-loss of the out-of-fold P of a SuccessModel on `features`, with folds by episode index (grouped_folds)."""
    x, folds, p = SuccessModel(features).x(records), grouped_folds(episodes, k, seed), np.zeros(len(y))
    for i in range(k):
        test = folds == i
        if test.any():
            p[test] = SuccessModel(features).fit(x[~test], tasks[~test], y[~test]).prob(x[test], tasks[test])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_features(records: List[dict], tasks: np.ndarray, y: np.ndarray, candidates: List[List[str]],
                    episodes: np.ndarray, k: int = 10, seed: int = 0) -> Tuple[List[str], List[float]]:
    """The candidate feature list with the lowest cv_logloss (the first one on a tie), and each candidate's
    log-loss."""
    losses = [cv_logloss(records, tasks, y, list(c), episodes, k, seed) for c in candidates]
    return list(candidates[int(np.argmin(losses))]), losses


def fit_two_model(records: List[dict], tasks: np.ndarray, y_plan: np.ndarray, y_vla: np.ndarray,
                  trained_on: Optional[dict] = None, planner_features: Optional[List[str]] = None) -> TwoModelASM:
    """The two-model ASM fit to these episodes: their planner records, tasks and the two controllers' outcomes.
    planner_features: P(planner)'s features (default PLANNER_FEATURES)."""
    asm = TwoModelASM(SuccessModel(planner_features or PLANNER_FEATURES), SuccessModel(VLA_FEATURES), trained_on)
    asm.planner.fit(asm.planner.x(records), tasks, y_plan)
    asm.vla.fit(asm.vla.x(records), tasks, y_vla)
    return asm


def train_two_model(planner_root: str, vla_roots: List[str],
                    planner_candidates: Optional[List[List[str]]] = None) -> TwoModelASM:
    """The two-model ASM, with both success models trained on every layout both controllers ran, except the
    episodes the VLM plan gives to the VLA first (vlm_plan_gate). With planner_candidates, P(planner) uses the
    candidate that choose_features picks on these episodes."""
    planner, vla, keys = matched_layouts(planner_root, vla_roots)
    keys = [k for k in keys if vlm_plan_gate(planner[k]) is None]
    records = [planner[k] for k in keys]
    tasks, y_plan = np.array([k[0] for k in keys]), np.array([float(r["success"]) for r in records])
    features = None
    if planner_candidates:
        features, losses = choose_features(records, tasks, y_plan, planner_candidates, np.array([k[1] for k in keys]))
        print("P(planner) features, cross-validated log-loss: "
              + "; ".join(f"{', '.join(c)}: {v:.4f}" for c, v in zip(planner_candidates, losses, strict=True))
              + f" -> {', '.join(features)}")
    return fit_two_model(records, tasks, y_plan, np.array([vla[k] for k in keys]),
                         planner_settings(list(planner.values()), planner_root), features)


def describe(asm: TwoModelASM) -> List[str]:
    """The two success models in the features' own units, one line each."""
    lines = []
    for name, m in (("planner", asm.planner), ("VLA", asm.vla)):
        intercepts, slopes = m.plain()
        lines.append(f"P({name} succeeds): logit = " + " + ".join(f"{v:.3f} x {f}" for f, v in slopes.items())
                     + " + intercept (" + ", ".join(f"{t.split('-')[0]} {v:.3f}" for t, v in intercepts.items()) + ")")
    return lines


def save_two_model(planner_root: str, vla_roots: List[str], save: str,
                   planner_candidates: Optional[List[List[str]]] = None) -> None:
    asm = train_two_model(planner_root, vla_roots, planner_candidates)
    print("\n".join(describe(asm)))
    with open(save, "w") as fh:
        json.dump(asm.to_json(), fh, indent=1)
    print("saved", save)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-runs", required=True)
    ap.add_argument("--vla-runs", nargs="+", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--model", choices=sorted(MODELS), default="logreg", help="model that --save trains")
    ap.add_argument("--test-planner-runs", default=None,
                    help="held-out layouts (--layout-seed runs): train on --planner-runs, test on these")
    ap.add_argument("--test-vla-runs", nargs="+", default=None, help="VLA runs on the held-out layouts")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="v3",
                    help="v3 (default): plan_ok and plan_path_ratio only; v2: one per planner failure mode, "
                         "with no units; v1: the first 16 features")
    ap.add_argument("--rule", choices=["threshold", "two-model"], default="threshold",
                    help="two-model: train the two-model ASM on --planner-runs and --vla-runs and save it (--save); "
                         "experiments/score_two_model.py scores it")
    ap.add_argument("--planner-candidates", nargs="+", default=None,
                    help="with --rule two-model: P(planner)'s candidate feature lists, each comma-separated; the one "
                         "with the lowest cross-validated log-loss is used (default: PLANNER_FEATURES)")
    a = ap.parse_args()
    if bool(a.test_planner_runs) != bool(a.test_vla_runs):
        ap.error("--test-planner-runs and --test-vla-runs go together")
    if a.planner_candidates and a.rule != "two-model":
        ap.error("--planner-candidates is for --rule two-model")
    if a.rule == "two-model":
        if not a.save or a.test_planner_runs:
            ap.error("--rule two-model trains and saves the ASM: give --save, and no --test-* runs")
        save_two_model(a.planner_runs, a.vla_runs, a.save,
                       [c.split(",") for c in a.planner_candidates] if a.planner_candidates else None)
    else:
        evaluate(a.planner_runs, a.vla_runs, a.save, a.model, a.test_planner_runs, a.test_vla_runs, a.features)


if __name__ == "__main__":
    main()
