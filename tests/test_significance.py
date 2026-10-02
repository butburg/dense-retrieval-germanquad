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


def test_compare_mini_fixture():
    clusters = ["a", "a", "b", "b", "c", "d"]
    ranks = {"bm25": np.array([2, 3, 1, 5, 20, 0]), "e5-large": np.array([1, 1, 1, 2, 1, 3]),
             "bge-m3": np.array([1, 1, 1, 1, 1, 1])}
    res = cm.compare(ranks, clusters, b_perm=10_000, b_boot=500)
    conf = [r for r in res["tests"] if r["family"].startswith("confirmatory")]
    assert [r["b"] for r in conf] == ["e5-large", "bge-m3"]
    r = conf[0]
    assert r["n_nonzero_clusters"] == 4 and r["p_method"] == "exact" and r["not_informative"]
    assert r["p_holm"] >= r["p_cluster_signflip"] and r["diff_b_minus_a"] > 0
    s1 = next(x for x in res["tests"] if x["metric"] == "Success@1" and x["b"] == "bge-m3")
    assert abs(s1["diff_b_minus_a"] - 5 / 6) < 1e-12
    assert any(x["family"].startswith("explorative: embedder pairs") for x in res["tests"])


def test_compare_all_families_one_run():
    """Six confirmatory models, BM25-de and three extension models: 22 families, 103 tests."""
    rng = np.random.default_rng(0)
    n = 60
    clusters = [f"c{i % 20}" for i in range(n)]
    base = rng.integers(3, 8, n)
    ranks = {"bm25": base, "bm25_de": base + 1}
    for i, m in enumerate(cm.EMBEDDERS + cm.EXTENSION):
        ranks[m] = np.maximum(1, base - i % 3)
    res = cm.compare(ranks, clusters, b_perm=500, b_boot=50)
    fams = {r["family"] for r in res["tests"]}
    assert len(res["tests"]) == 103 and len(fams) == 22
    r = next(t for t in res["tests"] if t["b"] == "jina-v2-base-de" and t["a"] == "bge-m3" and t["metric"] == "MRR@10")
    assert r["diff_b_minus_a"] > 0 and "p_holm" in r  # jina-v2-base-de ranks the gold passage higher
