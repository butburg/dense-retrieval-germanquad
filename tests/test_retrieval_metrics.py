"""Tests for code/pylib/retrieval_metrics.py with a mini fixture."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pylib"))
from retrieval_metrics import build_gold, compute_metrics, first_relevant_ranks, rank_cosine  # noqa: E402

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
    m = {r["k"]: r for r in compute_metrics(qids, ranked, gold, "t", k_values=(1, 2))}
    assert m[1]["success_at_k"] == 0 and m[1]["mrr_at_k"] == 0
    assert m[2]["success_at_k"] == 2 / 3
    assert m[2]["recall_at_k"] == (1 + 0.5) / 3
    assert m[2]["mrr_at_k"] == (0.5 + 0.5) / 3
    assert m[2]["precision_at_k"] == (0.5 + 0.5) / 3
