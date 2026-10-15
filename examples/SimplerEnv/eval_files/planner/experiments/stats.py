"""Tests and scores for paired episode outcomes (numpy only)."""

import math
from typing import Tuple

import numpy as np


def mcnemar(base: np.ndarray, new: np.ndarray) -> Tuple[int, int, float]:
    """Exact two-sided McNemar test on paired 0/1 outcomes: (episodes better, episodes worse, p)."""
    better, worse = int((new > base).sum()), int((new < base).sum())
    n, k = better + worse, min(better, worse)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return better, worse, p


def sign_flip(d: np.ndarray, scale: int) -> float:
    """Exact two-sided paired sign-flip test of mean(d) = 0. d * scale must be whole numbers (scale = the number
    of VLA runs); for 0/1 outcomes it is the exact McNemar test."""
    z = np.round(d * scale).astype(int)
    assert np.allclose(z, d * scale)
    obs, z = abs(int(z.sum())), np.abs(z[z != 0])
    if len(z) == 0:
        return 1.0
    total = int(z.sum())
    dist = np.zeros(2 * total + 1)  # P(sum of the randomly signed values = i - total)
    dist[total] = 1.0
    for x in z:  # the sum never leaves [-total, total], so np.roll never wraps mass around
        dist = 0.5 * (np.roll(dist, x) + np.roll(dist, -x))
    return float(min(1.0, dist[np.abs(np.arange(-total, total + 1)) >= obs].sum()))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """AUC: of the pairs of episodes with different labels, the share where the higher label gets the higher p
    (ties count half). With 0/1 labels it is P(p of a success > p of a failure); with mean labels (several VLA
    runs per episode) it is the concordance."""
    higher = y[:, None] > y[None, :]
    if not higher.any():
        return float("nan")
    above, tied = (p[:, None] > p[None, :]) & higher, (p[:, None] == p[None, :]) & higher
    return float(above.sum() + 0.5 * tied.sum()) / higher.sum()


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fleiss_kappa(v: np.ndarray) -> float:
    """Agreement of the runs (columns of v, 0/1) on each episode beyond chance (1 = always the same outcome,
    0 = chance)."""
    n = v.shape[1]
    if n < 2:
        return float("nan")
    k = v.sum(1)
    agree = ((k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))).mean()
    q = v.mean()
    chance = q ** 2 + (1 - q) ** 2
    return float((agree - chance) / (1 - chance)) if chance < 1 else float("nan")
