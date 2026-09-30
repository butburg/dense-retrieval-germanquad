"""Retrieval-only ranking and metric logic (same definitions as repro_germanquad.py)."""
from __future__ import annotations

import numpy as np

K_VALUES = (1, 5, 10)


def build_gold(qrels: list[dict], doc_ids: set[str], threshold: int = 1) -> dict[str, set[str]]:
    """Map query_id to the set of relevant doc_ids (relevance >= threshold, doc in corpus)."""
    gold: dict[str, set[str]] = {}
    for r in qrels:
        if int(r["relevance"]) >= threshold and str(r["doc_id"]) in doc_ids:
            gold.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))
    return gold


def rank_cosine(q_emb: np.ndarray, d_emb: np.ndarray, max_k: int = 10) -> np.ndarray:
    """Top-``max_k`` document indices per query by dot product.

    Embeddings must be L2-normalised so the dot product equals cosine similarity.
    Returns an int array of shape (n_queries, max_k), best first.
    """
    sims = q_emb.astype(np.float32) @ d_emb.astype(np.float32).T
    part = np.argpartition(-sims, max_k - 1, axis=1)[:, :max_k]
    order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def first_relevant_ranks(qids: list[str], ranked: list[list[str]], gold: dict[str, set[str]]) -> list[int | None]:
    """1-based rank of the first relevant doc within the ranked list, else None."""
    out = []
    for qid, top in zip(qids, ranked):
        g = gold.get(qid, set())
        out.append(next((i for i, d in enumerate(top, 1) if d in g), None))
    return out


def compute_metrics(qids: list[str], ranked: list[list[str]], gold: dict[str, set[str]],
                    dataset: str, k_values=K_VALUES) -> list[dict]:
    """Recall/Success/MRR/Precision@k averaged over all queries.

    Queries without relevant docs count as 0 (denominator = all queries).
    ``ranked`` must contain at least ``max(k_values)`` docs per query.
    """
    rows = []
    n = len(qids)
    for k in k_values:
        rec = suc = mrr = prec = 0.0
        for qid, top in zip(qids, ranked):
            g = gold.get(qid, set())
            hits = sum(d in g for d in top[:k])
            if g:
                rec += hits / len(g)
            suc += hits > 0
            prec += hits / k
            fr = next((i for i, d in enumerate(top, 1) if d in g), None)
            if fr is not None and fr <= k:
                mrr += 1 / fr
        rows.append({"dataset": dataset, "k": k, "recall_at_k": rec / n, "success_at_k": suc / n,
                     "mrr_at_k": mrr / n, "num_queries": n, "precision_at_k": prec / n})
    return rows
