"""Tests and scores for paired episode outcomes (numpy only)."""

import math
from typing import List, Tuple

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


def auc_ranks(p: np.ndarray, y: np.ndarray) -> float:
    """The AUC of `auc` for 0/1 labels, from midranks (fast enough for bootstrap draws)."""
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m == 0 or n == 0:
        return float("nan")
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    midrank = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    return float((midrank[y].sum() - m * (m + 1) / 2.0) / (m * n))


def delong(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """DeLong's test of two AUCs on the same episodes (DeLong, DeLong and Clarke-Pearson, 1988): (AUC of p_a, AUC of
    p_b, two-sided p of equal AUCs). y: 0/1 labels. Each AUC's structural components are, for each positive, the
    share of negatives it beats (ties count half), and for each negative, the share of positives that beat it."""
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong's test needs at least 2 episodes of each label")
    v10, v01 = [], []
    for p in (np.asarray(p_a, float), np.asarray(p_b, float)):
        psi = (p[y][:, None] > p[~y][None, :]) + 0.5 * (p[y][:, None] == p[~y][None, :])
        v10.append(psi.mean(1))
        v01.append(psi.mean(0))
    a, b = float(v10[0].mean()), float(v10[1].mean())
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 1e-15:
        return a, b, 1.0 if abs(a - b) < 1e-12 else 0.0
    return a, b, float(math.erfc(abs(a - b) / math.sqrt(var) / math.sqrt(2)))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibration_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> List[Tuple[float, float, int, float, float]]:
    """(low, high, episodes, mean p, mean y) for each of `bins` equal-width bins of p that has episodes."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [(float(edges[b]), float(edges[b + 1]), int((idx == b).sum()), float(p[idx == b].mean()),
             float(y[idx == b].mean())) for b in range(bins) if (idx == b).any()]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: the gap between p and the success rate in each equal-width bin of p, weighted
    by the bin's episodes."""
    return float(sum(n * abs(mp - my) for _, _, n, mp, my in calibration_bins(p, y, bins)) / len(p))


def bootstrap_interval(v: np.ndarray, strata: np.ndarray, draws: int = 10000, seed: int = 0,
                       level: float = 0.95) -> Tuple[float, float]:
    """Percentile bootstrap interval of mean(v): each draw takes, in each stratum (task), as many of its episodes
    as it has, with replacement, as the layout sets do. For a paired difference, v = new - base."""
    rng = np.random.default_rng(seed)
    sums = np.zeros(draws)
    for k in sorted(set(strata.tolist())):
        g = v[strata == k]
        sums += g[rng.integers(0, len(g), size=(draws, len(g)))].sum(1)
    lo, hi = np.quantile(sums / len(v), [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


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
