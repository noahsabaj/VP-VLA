"""Routers that the newer studies compare with the two-model ASM (numpy only). Each gives, for each test episode,
whether it goes to the planner (else to the VLA), from the training set's episodes: their features, tasks and both
methods' outcomes.

* per_task_choice: one method per task, the better one on the training set (a router that does not look at the
  episode);
* knn_choice: the retrieval router: the method that did better on the k most similar training episodes of the task,
  with k chosen on the training set (choose_k);
* expected_random: a coin with the ASM's chance of the planner (the expected outcome, so no test).
"""

from typing import List, Tuple

import numpy as np

from .. import selector as s

KS = (1, 3, 5, 9, 15, 25, 45)  # the values of k that choose_k compares


def per_task_choice(train_tasks: np.ndarray, train_y_plan: np.ndarray, train_y_vla: np.ndarray,
                    test_tasks: np.ndarray) -> np.ndarray:
    """For each task, the method with the higher success rate on the training set (the planner on a tie)."""
    better = {t: train_y_plan[train_tasks == t].mean() >= train_y_vla[train_tasks == t].mean()
              for t in set(train_tasks.tolist())}
    unknown = sorted(set(test_tasks.tolist()) - set(better))
    if unknown:
        raise ValueError(f"no training episodes of {', '.join(unknown)}")
    return np.array([better[t] for t in test_tasks.tolist()], bool)


def knn_choice(train_x: np.ndarray, train_tasks: np.ndarray, train_y_plan: np.ndarray, train_y_vla: np.ndarray,
               test_x: np.ndarray, test_tasks: np.ndarray, k: int) -> np.ndarray:
    """The retrieval router: for each test episode, its k nearest training episodes of the same task (Euclidean
    distance on the features, each standardized on the training set; the earlier episode on a tie); the planner when
    their planner success rate is at least their VLA success rate."""
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    a, b = (train_x - mean) / std, (test_x - mean) / std
    use = np.zeros(len(test_x), bool)
    for t in sorted(set(test_tasks.tolist())):
        tr, te = np.flatnonzero(train_tasks == t), np.flatnonzero(test_tasks == t)
        if not len(tr):
            raise ValueError(f"no training episodes of {t}")
        d = ((b[te, None, :] - a[None, tr, :]) ** 2).sum(-1)
        near = np.argsort(d, axis=1, kind="stable")[:, :min(k, len(tr))]
        use[te] = train_y_plan[tr][near].mean(1) >= train_y_vla[tr][near].mean(1)
    return use


def choose_k(x: np.ndarray, tasks: np.ndarray, y_plan: np.ndarray, y_vla: np.ndarray, groups: np.ndarray,
             ks=KS, folds: int = 10, seed: int = 0) -> Tuple[int, List[float]]:
    """k for knn_choice, by cross-validation on the training set with folds by layout group (selector.grouped_folds,
    as for the ASM's features): the k whose routed success on the held-out folds is the highest (the larger k on a
    tie), and each k's success rate."""
    fold = s.grouped_folds(groups, folds, seed)
    scores = []
    for k in ks:
        got = np.zeros(len(x))
        for i in range(folds):
            te = fold == i
            if te.any():
                use = knn_choice(x[~te], tasks[~te], y_plan[~te], y_vla[~te], x[te], tasks[te], k)
                got[te] = np.where(use, y_plan[te], y_vla[te])
        scores.append(float(got.mean()))
    best = max(range(len(ks)), key=lambda i: (scores[i], ks[i]))
    return ks[best], scores


def expected_random(share, y_plan: np.ndarray, y_vla: np.ndarray) -> np.ndarray:
    """Each episode's expected outcome when a coin gives it to the planner with chance `share` (one number, or one per
    episode): share * planner + (1 - share) * VLA."""
    return share * y_plan + (1 - share) * y_vla
