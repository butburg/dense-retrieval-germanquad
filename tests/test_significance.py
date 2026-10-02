"""Cluster significance tools and compare_models on a mini fixture."""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "pylib"))
from significance import cluster_sign_flip, holm, make_clusters  # noqa: E402
import compare_models as cm  # noqa: E402


def test_exact_sign_flip_matches_enumeration():
    cinv, ncl, _ = make_clusters(["a", "b", "c"])
    d = np.array([1.0, 1.0, 1.0])  # all positive: only 2 of 8 patterns reach |sum|=3
    r = cluster_sign_flip(d, cinv, ncl)
    assert r["method"] == "exact" and r["n_nonzero"] == 3 and abs(r["p"] - 2 / 8) < 1e-12


def test_zero_clusters_ignored_and_all_zero():
    cinv, ncl, _ = make_clusters(["a", "a", "b", "c"])
    r = cluster_sign_flip(np.array([1.0, -1.0, 0.0, 0.0]), cinv, ncl)
    assert r["n_nonzero"] == 0 and r["p"] == 1.0


def test_monte_carlo_deterministic_and_floor():
    labels = [f"c{i}" for i in range(40)]
    cinv, ncl, _ = make_clusters(labels)
    d = np.ones(40)
    r1, r2 = cluster_sign_flip(d, cinv, ncl, b=20_000), cluster_sign_flip(d, cinv, ncl, b=20_000)
    assert r1 == r2 and r1["method"] == "monte_carlo" and r1["mc_floor"] and abs(r1["p"] - 1 / 20_001) < 1e-12


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.01], family_size=6), [0.06])


def _ranks(n=60, seed=0):
    rng = np.random.default_rng(seed)
    clusters = [f"c{i % 20}" for i in range(n)]
    base = rng.integers(3, 8, n)
    ranks = {"bm25": base, "bm25_de": base + 1}
    for i, m in enumerate(cm.MODELS):
        ranks[m] = np.maximum(1, base - i % 3)
    return ranks, clusters


def test_compare_55_tests_in_four_families():
    ranks, clusters = _ranks()
    res = cm.compare(ranks, clusters, b_perm=500, b_boot=50)
    sizes = {}
    for r in res["tests"]:
        sizes[r["family"][:2]] = sizes.get(r["family"][:2], 0) + 1
    assert len(res["tests"]) == 55 and sizes == {"F1": 36, "F2": 9, "F3": 9, "F4": 1}
    assert all(r["metric"] == "MRR@10" for r in res["tests"])
    f4 = next(r for r in res["tests"] if r["family"].startswith("F4"))
    assert f4["p_holm"] == f4["p_cluster_signflip"]  # Familie mit einem Test: keine Korrektur
    r = next(t for t in res["tests"] if t["b"] == "jina-v2-base-de" and t["a"] == "bm25")
    assert r["diff_b_minus_a"] > 0 and r["p_holm"] >= r["p_cluster_signflip"]
