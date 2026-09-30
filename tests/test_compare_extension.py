"""Synthetic checks for compare_extension.run (no data files needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pylib"))
import compare_extension as ce  # noqa: E402


def test_families_and_direction():
    rng = np.random.default_rng(0)
    n = 60
    clusters = [f"c{i % 20}" for i in range(n)]
    base = rng.integers(3, 8, n)
    ranks = {"bm25": base, "bm25_de": base, "bge-m3": base, "arctic-embed-l-v2": np.ones(n, int), "jina-v2-base-de": base + 2}
    rows = ce.run(ranks, clusters, b_perm=2000, b_boot=200)
    assert len(rows) == 3 * 4 * 2  # 3 baselines x 4 metrics x 2 present models
    r = next(t for t in rows if t["b"] == "arctic-embed-l-v2" and t["a"] == "bm25" and t["metric"] == "MRR@10")
    assert r["diff_b_minus_a"] > 0 and "p_holm" in r
    r = next(t for t in rows if t["b"] == "jina-v2-base-de" and t["a"] == "bge-m3" and t["metric"] == "MRR@10")
    assert r["diff_b_minus_a"] < 0
