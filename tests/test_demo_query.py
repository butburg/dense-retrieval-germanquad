"""demo_query.py: Gold-Ränge müssen mit den gespeicherten per-query-CSVs übereinstimmen."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))

DS = CODE / "output" / "germanquad"
BM25_CSV = CODE / "output" / "bm25_v2" / "per_query_ranks.csv"
E5_CSV = CODE / "output" / "harness" / "e5-large" / "per_query_germanquad.csv"
CACHE = CODE / "output" / "harness" / "cache"
N = 25
pytest.importorskip("rank_bm25")
pytestmark = pytest.mark.skipif(not (DS / "docs.normalized.jsonl").exists() or not BM25_CSV.exists(),
                                reason="normalisierte Daten oder bm25_v2-CSV fehlen")


def _sample_qids(ids: list[str]) -> list[str]:
    rest = [q for q in ids if q != "q40369"]
    return ["q40369"] + rest[:: max(1, len(rest) // (N - 1))][: N - 1]


@pytest.fixture(scope="module")
def data():
    import demo_query as dq
    from embed_eval import doc_text
    from retrieval_metrics import build_gold
    docs, queries, qrels = dq.load_dataset(DS)
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    return dq, docs_by_id, doc_ids, {str(q["query_id"]): str(q["query_text"]) for q in queries}, \
        build_gold(qrels, set(doc_ids), 1), doc_text


def test_bm25_gold_rank_and_top10_match_csv(data):
    dq, docs_by_id, doc_ids, qtext, gold, _ = data
    ref = pd.read_csv(BM25_CSV, dtype={"query_id": str}).query("dataset == 'germanquad'").set_index("query_id")
    bm25, tok = dq.build_bm25(doc_ids, docs_by_id)
    qids = _sample_qids(list(qtext))
    assert len(qids) >= 20
    for qid in qids:
        rk = dq.bm25_ranking(bm25, tok, doc_ids, qtext[qid])
        assert dq.gold_rank(rk, gold[qid]) == ref.loc[qid, "best_gold_rank_full"]
        assert [d for d, _ in rk[:10]] == ref.loc[qid, "top_doc_ids"].split(" | ")


def test_e5_gold_rank_matches_csv(data):
    dq, docs_by_id, doc_ids, qtext, gold, doc_text = data
    if not list(CACHE.glob("e5-large__germanquad_docs__*.npy")):
        pytest.skip("Dokument-Embedding-Cache fehlt (python code/demo_query.py --warm-cache)")
    from embed_eval import cached_encode
    from embedders import Embedder, get_config
    emb = Embedder(get_config("e5-large"))
    d_emb, hit, _ = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, CACHE, "germanquad_docs")
    assert hit
    ref = pd.read_csv(E5_CSV, dtype={"query_id": str}).set_index("query_id")
    qids = _sample_qids(list(qtext))
    try:
        q_emb = emb.encode([qtext[q] for q in qids], True)
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"Modell nicht ladbar: {type(ex).__name__}")
    for qid, qv in zip(qids, q_emb):
        rk = dq.dense_ranking(qv[None, :], d_emb, doc_ids)
        assert dq.gold_rank(rk, gold[qid]) == ref.loc[qid, "rank_first_relevant"]
