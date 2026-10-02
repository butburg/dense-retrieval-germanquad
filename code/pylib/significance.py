"""Paired significance tools following docs/evaluation_protocol.md (cluster level).

Estimators: cluster sign-flip test with one sign per gold-document cluster, ratio-estimator
cluster bootstrap and Holm correction.
Every call creates its own ``numpy.random.default_rng(seed)`` so results do not depend on call order.
"""
from __future__ import annotations

import numpy as np

SEED, B_PERM, B_BOOT, EXACT_MAX_N = 42, 100_000, 10_000, 16
TOL = 1e-9  # cluster sums with |x| <= TOL count as zero (float noise of rank differences)


def make_clusters(cluster_labels: list[str]) -> tuple[np.ndarray, int, np.ndarray]:
    """Map cluster labels to indices.

    Returns:
        Tuple ``(cinv, n_clusters, sizes)``: cluster index per query, number of clusters, queries per cluster.
    """
    _, cinv = np.unique(np.asarray(cluster_labels), return_inverse=True)
    return cinv, int(cinv.max()) + 1, np.bincount(cinv)


def cluster_sums(d: np.ndarray, cinv: np.ndarray, ncl: int) -> np.ndarray:
    """Sum the per-query differences ``d`` within each cluster."""
    return np.bincount(cinv, weights=d, minlength=ncl)


def cluster_sign_flip(d: np.ndarray, cinv: np.ndarray, ncl: int, seed: int = SEED, b: int = B_PERM) -> dict:
    """Two-sided paired sign-flip test with one random sign per cluster on cluster-summed differences.

    Exact enumeration of all 2^m sign patterns over the m clusters with non-zero sum when
    ``m <= EXACT_MAX_N`` (p = hits / 2^m); otherwise Monte-Carlo over all clusters with a fresh
    ``default_rng(seed)`` and p = (hits + 1) / (b + 1).

    Returns:
        Dict with ``p``, ``method`` ("exact" | "monte_carlo"), ``n_nonzero`` (clusters with sum != 0),
        ``mc_floor`` (True if a Monte-Carlo p sits at the floor 1/(b+1), i.e. is only an upper bound).
    """
    n = len(d)
    dc = cluster_sums(d, cinv, ncl) / n
    nz = dc[np.abs(dc) > TOL / n]
    m = len(nz)
    obs = abs(dc.sum())
    if m <= EXACT_MAX_N:
        if m == 0:
            return {"p": 1.0, "method": "exact", "n_nonzero": 0, "mc_floor": False}
        bits = (np.arange(2 ** m)[:, None] >> np.arange(m)) & 1
        stats = np.abs((2.0 * bits - 1.0) @ nz)
        return {"p": float((stats >= obs - 1e-12).mean()), "method": "exact", "n_nonzero": m, "mc_floor": False}
    rng, cnt, chunk = np.random.default_rng(seed), 0, 10_000
    for _ in range(b // chunk):
        s = rng.choice([-1.0, 1.0], size=(chunk, ncl))
        cnt += int((np.abs(s @ dc) >= obs - 1e-12).sum())
    p = (cnt + 1) / (b + 1)
    return {"p": p, "method": "monte_carlo", "n_nonzero": m, "mc_floor": p <= 1 / (b + 1) * (1 + 1e-9)}


def cluster_bootstrap_ci(d: np.ndarray, cinv: np.ndarray, ncl: int, sizes: np.ndarray,
                         seed: int = SEED, b: int = B_BOOT) -> list[float]:
    """95% percentile CI of the query-weighted mean of ``d`` under cluster resampling (fresh rng)."""
    rng = np.random.default_rng(seed)
    w = rng.multinomial(ncl, np.full(ncl, 1 / ncl), size=b)
    est = (w @ cluster_sums(d, cinv, ncl)) / (w @ sizes)
    return [float(x) for x in np.percentile(est, [2.5, 97.5])]


def holm(ps: list[float], family_size: int | None = None) -> list[float]:
    """Holm step-down adjusted p-values.

    Args:
        ps: Raw p-values of the tests present.
        family_size: Size of the full family (default ``len(ps)``); larger values keep the correction
            of the planned family when only some of its tests are available yet.
    """
    m = family_size or len(ps)
    out, run = [0.0] * len(ps), 0.0
    for rnk, i in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (m - rnk) * ps[i]))
        out[i] = run
    return out
