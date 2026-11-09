"""Paired significance tools following docs/evaluation_protocol.md (cluster level).

Estimators: cluster sign-flip test with one sign per gold-document cluster, ratio-estimator
cluster bootstrap and Holm correction.
Every call creates its own ``numpy.random.default_rng(seed)`` so results do not depend on call order.
"""
from __future__ import annotations

import numpy as np

SEED, B_PERM, B_BOOT, EXACT_MAX_N = 42, 100_000, 10_000, 16
TOL = 1e-9  # cluster sums with |x| <= TOL count as zero (float noise of rank differences)
# Permuted statistics count as "at least as extreme" if they reach observed_statistic - STATISTIC_TOL
# (absolute slack against float noise when comparing equal sums computed in different order).
STATISTIC_TOL = 1e-12
# A Monte-Carlo p-value counts as sitting at the floor 1 / (n_permutations + 1) up to this relative slack.
MC_FLOOR_REL_TOL = 1e-9


def make_clusters(cluster_labels: list[str]) -> tuple[np.ndarray, int, np.ndarray]:
    """Map cluster labels to consecutive cluster indices.

    Args:
        cluster_labels: One label per query, e.g. the gold-document ID that defines the
            cluster, ``["docA", "docB", "docA"]``.

    Returns:
        Tuple ``(cluster_indices, n_clusters, cluster_sizes)``: cluster index (0 to
        ``n_clusters - 1``, ordered by sorted label) per query, number of distinct clusters
        and number of queries per cluster.
    """
    # np.unique(..., return_inverse=True) maps each label to its rank among the sorted unique labels,
    # e.g. ["docB", "docA", "docB"] -> [1, 0, 1]; bincount then counts queries per cluster -> [1, 2].
    _, cluster_indices = np.unique(np.asarray(cluster_labels), return_inverse=True)
    n_clusters = int(cluster_indices.max()) + 1
    return cluster_indices, n_clusters, np.bincount(cluster_indices)


def cluster_sums(differences: np.ndarray, cluster_indices: np.ndarray, n_clusters: int) -> np.ndarray:
    """Sum the per-query differences within each cluster.

    Args:
        differences: Per-query score differences, shape ``(n_queries,)``.
        cluster_indices: Cluster index per query, same order as ``differences``
            (first element of :func:`make_clusters`).
        n_clusters: Number of clusters (second element of :func:`make_clusters`).

    Returns:
        Array of shape ``(n_clusters,)`` with the summed difference per cluster.
    """
    # bincount with weights adds up the differences per cluster index,
    # e.g. cluster_indices [0, 1, 0] with differences [0.5, -1, 0.25] -> [0.75, -1.0].
    return np.bincount(cluster_indices, weights=differences, minlength=n_clusters)


def cluster_sign_flip(differences: np.ndarray, cluster_indices: np.ndarray, n_clusters: int,
                      seed: int = SEED, n_permutations: int = B_PERM) -> dict:
    """Run a two-sided paired sign-flip test with one random sign per cluster.

    The test statistic is the absolute mean difference. Each cluster contributes its summed
    difference, and the sign of the whole cluster is flipped at random. If at most
    ``EXACT_MAX_N`` clusters have a non-zero sum, all ``2^m`` sign patterns over these ``m``
    clusters are enumerated (p = hits / 2^m). Otherwise signs are drawn Monte-Carlo style
    over all clusters with a fresh ``default_rng(seed)`` (p = (hits + 1) / (n_permutations + 1)).

    Args:
        differences: Per-query score differences between two retrievers, shape ``(n_queries,)``.
        cluster_indices: Cluster index per query, same order as ``differences``.
        n_clusters: Number of clusters (see :func:`make_clusters`).
        seed: Seed of the Monte-Carlo random generator; unused for exact enumeration.
        n_permutations: Number of Monte-Carlo sign patterns, drawn in blocks of 10,000
            plus one smaller remainder block, so exactly this many patterns are drawn.

    Returns:
        Dict with ``p``, ``method`` ("exact" | "monte_carlo"), ``n_nonzero`` (clusters with
        sum != 0) and ``mc_floor`` (True if a Monte-Carlo p sits at the floor
        1 / (n_permutations + 1), i.e. is only an upper bound).
    """
    n_queries = len(differences)
    cluster_effects = cluster_sums(differences, cluster_indices, n_clusters) / n_queries
    # Boolean mask keeps only clusters whose effect differs from zero beyond float noise,
    # e.g. [0.0, 0.2, -0.1] -> [0.2, -0.1]; only these clusters can change the sign-flip statistic.
    nonzero_effects = cluster_effects[np.abs(cluster_effects) > TOL / n_queries]
    n_nonzero = len(nonzero_effects)
    observed_statistic = abs(cluster_effects.sum())

    if n_nonzero <= EXACT_MAX_N:
        if n_nonzero == 0:
            return {"p": 1.0, "method": "exact", "n_nonzero": 0, "mc_floor": False}
        # Row i of sign_bits is the binary representation of i: one sign pattern per row.
        # Broadcasting (2^m, 1) >> (m,) shifts every i by every bit position, e.g. for m = 2 the rows
        # are 00, 01, 10, 11. Mapping {0, 1} to {-1, +1} gives all 2^m sign patterns.
        sign_bits = (np.arange(2 ** n_nonzero)[:, None] >> np.arange(n_nonzero)) & 1
        # Matrix product: each sign pattern (row) times the cluster effects = permuted mean difference,
        # e.g. signs [+1, -1] with effects [0.2, -0.1] -> 0.2 + 0.1 = 0.3; abs() gives the two-sided statistic.
        permuted_statistics = np.abs((2.0 * sign_bits - 1.0) @ nonzero_effects)
        p_value = float((permuted_statistics >= observed_statistic - STATISTIC_TOL).mean())
        return {"p": p_value, "method": "exact", "n_nonzero": n_nonzero, "mc_floor": False}

    rng = np.random.default_rng(seed)
    chunk_size = 10_000
    hit_count = 0
    # Full blocks of chunk_size patterns first, then one remainder block, so that exactly
    # n_permutations patterns are drawn (full blocks stay identical for multiples of chunk_size).
    n_full_blocks, remainder = divmod(n_permutations, chunk_size)
    block_sizes = [chunk_size] * n_full_blocks + ([remainder] if remainder else [])
    # Per block: draw random sign patterns (one +-1 per cluster each), compute their permuted
    # statistics and count how many are at least as extreme as the observed one (Monte-Carlo p-value).
    for block_size in block_sizes:
        # Matrix of shape (block_size, n_clusters), e.g. a row [+1, -1, +1] flips the sign of cluster B only.
        signs = rng.choice([-1.0, 1.0], size=(block_size, n_clusters))
        # Row-wise weighted sum over clusters = mean difference of one sign-flipped replicate.
        permuted_statistics = np.abs(signs @ cluster_effects)
        # The boolean comparison yields True/False per replicate; .sum() counts the Trues (hits).
        hit_count += int((permuted_statistics >= observed_statistic - STATISTIC_TOL).sum())
    p_value = (hit_count + 1) / (n_permutations + 1)
    is_at_floor = p_value <= 1 / (n_permutations + 1) * (1 + MC_FLOOR_REL_TOL)
    return {"p": p_value, "method": "monte_carlo", "n_nonzero": n_nonzero, "mc_floor": is_at_floor}


def cluster_bootstrap_ci(differences: np.ndarray, cluster_indices: np.ndarray, n_clusters: int,
                         cluster_sizes: np.ndarray, seed: int = SEED,
                         n_resamples: int = B_BOOT) -> list[float]:
    """Compute the 95% percentile CI of the mean difference under cluster resampling.

    Clusters are resampled with replacement (multinomial counts); each resample's estimate is
    the ratio of the resampled difference sum to the resampled number of queries, so the mean
    stays query-weighted. A fresh ``default_rng(seed)`` is created on every call.

    Args:
        differences: Per-query score differences, shape ``(n_queries,)``.
        cluster_indices: Cluster index per query, same order as ``differences``.
        n_clusters: Number of clusters (see :func:`make_clusters`).
        cluster_sizes: Number of queries per cluster, shape ``(n_clusters,)``.
        seed: Seed of the random generator.
        n_resamples: Number of bootstrap resamples.

    Returns:
        ``[lower, upper]`` bounds (2.5th and 97.5th percentile) of the resampled estimates.
    """
    rng = np.random.default_rng(seed)
    # Multinomial draw = bootstrap over clusters: n_clusters draws with equal probability 1/n_clusters,
    # returned as counts per cluster; one row per resample, shape (n_resamples, n_clusters).
    cluster_draw_counts = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_resamples)
    # Each row of cluster_draw_counts says how often each cluster was drawn in one resample,
    # e.g. [2, 0, 1] for clusters A, B, C. The matrix product (@) turns every row into a weighted
    # sum over clusters: 2*sum_A + 0*sum_B + 1*sum_C = total score difference of that resample.
    # The same product with cluster_sizes gives the number of queries in that resample.
    resampled_difference_sums = cluster_draw_counts @ cluster_sums(differences, cluster_indices, n_clusters)
    resampled_query_counts = cluster_draw_counts @ cluster_sizes
    estimates = resampled_difference_sums / resampled_query_counts
    # np.percentile sorts the estimates and reads off the values below which 2.5 % / 97.5 % of them lie
    # (95 % percentile CI); the comprehension converts the two NumPy floats to plain floats for JSON output.
    return [float(bound) for bound in np.percentile(estimates, [2.5, 97.5])]


def holm(p_values: list[float], family_size: int | None = None) -> list[float]:
    """Apply the Holm step-down correction to raw p-values.

    Args:
        p_values: Raw p-values of the tests present, e.g. ``[0.01, 0.04, 0.03]``.
        family_size: Size of the full family (default ``len(p_values)``); larger values keep
            the correction of the planned family when only some of its tests are available yet.

    Returns:
        Adjusted p-values (capped at 1, monotone in the p-value order), same order as ``p_values``.
    """
    n_tests = family_size or len(p_values)
    # List repetition creates one 0.0 placeholder per test, e.g. 3 tests -> [0.0, 0.0, 0.0].
    adjusted_p_values = [0.0] * len(p_values)
    running_maximum = 0.0
    # Smallest raw p-value first; each is multiplied by the number of tests still open.
    # np.argsort gives the test indices in ascending p-order, e.g. p = [0.04, 0.01] -> [1, 0].
    # running_maximum keeps adjusted p-values monotone non-decreasing along that order.
    for rank, test_index in enumerate(np.argsort(p_values, kind="stable")):
        scaled_p_value = min(1.0, (n_tests - rank) * p_values[test_index])
        running_maximum = max(running_maximum, scaled_p_value)
        adjusted_p_values[test_index] = running_maximum
    return adjusted_p_values
