"""Tests for code/pylib/retrieval_metrics.py with a mini fixture."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pylib"))
from retrieval_metrics import (K_VALUES, build_gold, compute_metrics, first_relevant_ranks, rank_cosine,  # noqa: E402
                               reciprocal_rank_at_k, success_at_k)

DOCS = ["a", "b", "c"]
QRELS = [{"query_id": "q1", "doc_id": "a", "relevance": 1},
         {"query_id": "q2", "doc_id": "c", "relevance": 1},
         {"query_id": "q2", "doc_id": "b", "relevance": 1},
         {"query_id": "q2", "doc_id": "zz", "relevance": 1},   # not in corpus
         {"query_id": "q3", "doc_id": "a", "relevance": 0}]    # below threshold


def test_build_gold_filters():
    assert build_gold(QRELS, set(DOCS)) == {"q1": {"a"}, "q2": {"b", "c"}}


def test_rank_cosine_order():
    d = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert rank_cosine(q, d, 3).tolist() == [[0, 2, 1], [1, 2, 0]]


def test_metrics_values():
    gold = build_gold(QRELS, set(DOCS))
    qids = ["q1", "q2", "q3"]
    ranked = [["b", "a", "c"], ["a", "c", "b"], ["a", "b", "c"]]
    assert first_relevant_ranks(qids, ranked, gold) == [2, 2, None]
    m = {r["k"]: r for r in compute_metrics(qids, ranked, gold, "t", cutoff_values=(1, 2))}
    assert m[1]["success_at_k"] == 0 and m[1]["mrr_at_k"] == 0
    assert m[2]["success_at_k"] == 2 / 3
    assert m[2]["recall_at_k"] == (1 + 0.5) / 3
    assert m[2]["mrr_at_k"] == (0.5 + 0.5) / 3
    assert m[2]["precision_at_k"] == (0.5 + 0.5) / 3


def test_unequal_lengths_raise():
    gold = build_gold(QRELS, set(DOCS))
    with pytest.raises(ValueError, match="differ in length"):
        first_relevant_ranks(["q1", "q2"], [["a"]], gold)
    with pytest.raises(ValueError, match="differ in length"):
        compute_metrics(["q1"], [["a"], ["b"]], gold, "t", cutoff_values=(1,))


def test_duplicate_qids_raise():
    gold = build_gold(QRELS, set(DOCS))
    ranked = [["a", "b"], ["b", "a"]]
    with pytest.raises(ValueError, match="contain duplicates"):
        first_relevant_ranks(["q1", "q1"], ranked, gold)
    with pytest.raises(ValueError, match="contain duplicates"):
        compute_metrics(["q1", "q1"], ranked, gold, "t", cutoff_values=(1,))


def test_empty_queries_give_zero_metrics():
    assert all(m["recall_at_k"] == 0 and m["num_queries"] == 0
               for m in compute_metrics([], [], {}, "t", cutoff_values=(1,)))


def test_run_bm25_metrics_match_compute_metrics(tmp_path, monkeypatch):
    """run_bm25.evaluate liefert dieselben Metriken wie compute_metrics auf den Top-k-Listen."""
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "scripts"))
    import run_bm25
    rows = {"queries": [{"query_id": "q1", "query_text": "Katze Maus"}, {"query_id": "q2", "query_text": "Hund Ball"},
                        {"query_id": "q3", "query_text": "Vogel"}],
            "docs": [{"doc_id": "a", "title": "", "text": "Die Katze jagt die Maus"},
                     {"doc_id": "b", "title": "", "text": "Der Hund spielt mit dem Ball"},
                     {"doc_id": "c", "title": "", "text": "Ein Vogel singt im Baum"}],
            "qrels": [{"query_id": "q1", "doc_id": "a", "relevance": 1}, {"query_id": "q2", "doc_id": "b", "relevance": 1},
                      {"query_id": "q3", "doc_id": "zz", "relevance": 1}]}
    paths = {}
    for name, data in rows.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    metrics, overview, per_q = run_bm25.evaluate("t", paths)
    gold = build_gold(rows["qrels"], {"a", "b", "c"})
    expected = compute_metrics([r["query_id"] for r in per_q], [r["top_doc_ids"] for r in per_q], gold, "t",
                               K_VALUES)
    assert metrics == [{k: v for k, v in m.items() if k != "precision_at_k"} for m in expected]
    assert overview["queries_without_relevant_docs"] == 1 and overview["queries_without_qrels"] == 0
    assert [r["first_rel_rank"] for r in per_q] == [1, 1, None]


# --- Rank-array metrics (success_at_k, reciprocal_rank_at_k) ---
FIRST_RANKS = np.array([1, 2, 5, 6, 10, 11, 12, 0, 3, 0])  # includes rank 0 (none) and ranks beyond k=10


def test_success_at_k_examples():
    assert success_at_k(np.array([1, 7, 0]), 5).tolist() == [1.0, 0.0, 0.0]
    assert success_at_k(FIRST_RANKS, 5).tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    assert success_at_k(FIRST_RANKS, 5).dtype == np.float64


def test_reciprocal_rank_at_k_examples():
    assert reciprocal_rank_at_k(np.array([1, 7, 12, 0]), 10).tolist() == [1.0, 1 / 7, 0.0, 0.0]
    assert reciprocal_rank_at_k(FIRST_RANKS, 5).tolist() == [1, 1 / 2, 1 / 5, 0, 0, 0, 0, 0, 1 / 3, 0]
    assert reciprocal_rank_at_k(FIRST_RANKS, 10).dtype == np.float64


def test_rank_array_metrics_match_compute_metrics():
    """Means of the vectorised per-query values equal the list-based compute_metrics on identical data."""
    n_docs = 15
    qids = [f"q{i}" for i in range(len(FIRST_RANKS))]
    gold = {qid: {"gold"} for qid in qids}
    # Ranked list per query: the gold document sits at its rank, rank 0 means the list holds no gold document.
    ranked = []
    for rank in FIRST_RANKS:
        documents = [f"other{i}" for i in range(n_docs)]
        if rank > 0:
            documents[rank - 1] = "gold"
        ranked.append(documents)
    assert first_relevant_ranks(qids, ranked, gold) == [int(r) or None for r in FIRST_RANKS]
    for row in compute_metrics(qids, ranked, gold, "t", cutoff_values=(1, 5, 10)):
        k = row["k"]
        assert success_at_k(FIRST_RANKS, k).mean() == pytest.approx(row["success_at_k"], abs=1e-12)
        assert reciprocal_rank_at_k(FIRST_RANKS, k).mean() == pytest.approx(row["mrr_at_k"], abs=1e-12)


def test_mrr_at_5_integer_form_is_bit_identical():
    """The former inline MRR@5 (``1 /`` and ``0`` instead of ``1.0 /`` and ``0.0``) gives the same bits."""
    rng = np.random.default_rng(0)
    ranks = rng.integers(0, 40, size=5000)
    inline = np.where((ranks > 0) & (ranks <= 5), 1 / np.maximum(ranks, 1), 0)
    assert inline.dtype == np.float64
    assert np.array_equal(inline, reciprocal_rank_at_k(ranks, 5))
    assert inline.mean() == reciprocal_rank_at_k(ranks, 5).mean()
