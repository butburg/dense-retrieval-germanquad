"""Monte-Carlo sign-flip draws: exact pattern count and bit-identity with the previous implementation."""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "pylib"))
import significance  # noqa: E402
from significance import cluster_sign_flip, make_clusters  # noqa: E402

CHUNK = 10_000


def _synthetic_clusters(n_clusters: int = 40, seed: int = 7):
    """Per-query differences with one query per cluster (>16 non-zero clusters -> Monte Carlo)."""
    rng = np.random.default_rng(seed)
    differences = rng.normal(0.05, 1.0, size=n_clusters)
    cluster_indices, n, _ = make_clusters([f"c{i}" for i in range(n_clusters)])
    return differences, cluster_indices, n


def _old_cluster_sign_flip_p(differences, cluster_indices, n_clusters, seed, n_permutations):
    """Monte-Carlo p of the previous implementation (only full 10,000 blocks drawn)."""
    n_queries = len(differences)
    effects = significance.cluster_sums(differences, cluster_indices, n_clusters) / n_queries
    observed = abs(effects.sum())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations // CHUNK):
        signs = rng.choice([-1.0, 1.0], size=(CHUNK, n_clusters))
        hits += int((np.abs(signs @ effects) >= observed - significance.STATISTIC_TOL).sum())
    return (hits + 1) / (n_permutations + 1)


@pytest.mark.parametrize("n_permutations", [15_000, 5_000, 1])
def test_exactly_n_permutations_drawn(monkeypatch, n_permutations):
    differences, cluster_indices, n_clusters = _synthetic_clusters()
    drawn_rows = []

    class CountingRng:
        def __init__(self, seed):
            self._rng = np.random.default_rng(seed)

        def choice(self, *args, **kwargs):
            result = self._rng.choice(*args, **kwargs)
            drawn_rows.append(result.shape[0])
            return result

    fake_random = types.SimpleNamespace(default_rng=CountingRng)
    monkeypatch.setattr(significance, "np", types.SimpleNamespace(**{**vars(np), "random": fake_random}))
    result = cluster_sign_flip(differences, cluster_indices, n_clusters, n_permutations=n_permutations)
    assert result["method"] == "monte_carlo"
    assert sum(drawn_rows) == n_permutations


@pytest.mark.parametrize("n_permutations", [100_000, 20_000])
@pytest.mark.parametrize("seed", [42, 3])
def test_multiples_of_chunk_bit_identical_to_old_implementation(n_permutations, seed):
    differences, cluster_indices, n_clusters = _synthetic_clusters()
    new = cluster_sign_flip(differences, cluster_indices, n_clusters, seed=seed, n_permutations=n_permutations)
    old_p = _old_cluster_sign_flip_p(differences, cluster_indices, n_clusters, seed, n_permutations)
    assert new["p"] == old_p  # exact float equality, no tolerance


def test_old_implementation_from_git_head_matches_reference_copy():
    """The inlined reference copy equals the committed pre-change code (skipped once HEAD contains the fix)."""
    source = subprocess.run(["git", "show", "HEAD:code/pylib/significance.py"], cwd=ROOT,
                            capture_output=True, text=True, check=False).stdout
    if not source or "n_full_blocks" in source:
        pytest.skip("pre-change implementation not available in HEAD")
    namespace = {"__name__": "significance_old"}
    exec(compile(source, "significance_old", "exec"), namespace)
    differences, cluster_indices, n_clusters = _synthetic_clusters()
    for n_permutations in (20_000, 100_000):
        old = namespace["cluster_sign_flip"](differences, cluster_indices, n_clusters, n_permutations=n_permutations)
        new = cluster_sign_flip(differences, cluster_indices, n_clusters, n_permutations=n_permutations)
        assert old == new
