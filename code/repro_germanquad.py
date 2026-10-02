"""Reproduce BM25 / E5 pilot metrics on GermanQuAD (logic of notebooks 03 and 04).

Usage: python repro_germanquad.py {bm25|e5}. Writes only to code/output/repro/.
"""
from __future__ import annotations
import json, platform, sys, time
from datetime import datetime, timezone
from importlib.metadata import version as pv
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
from german_text import TOKEN_PATTERN, tokenize_de  # noqa: E402
from local_dataset_io import doc_text, load_jsonl  # noqa: E402
from retrieval_metrics import build_gold, compute_metrics  # noqa: E402

D = CODE / "output" / "germanquad"
OUT = CODE / "output" / "repro"
K_VALUES, THR = [1, 5, 10], 1
queries, docs, qrels = (load_jsonl(D / f"{n}.normalized.jsonl") for n in ("queries", "docs", "qrels"))
docs_by_id = {str(d["doc_id"]): d for d in docs}
doc_ids = list(docs_by_id)
gold = build_gold(qrels, set(docs_by_id), THR)
qids = [str(q["query_id"]) for q in queries]
# doc_id type check: raw types in docs / qrels, and that all qrels doc_ids resolve after str() casting
DOC_ID_TYPES = {"docs": sorted({type(d["doc_id"]).__name__ for d in docs}),
                "qrels": sorted({type(r["doc_id"]).__name__ for r in qrels}),
                "qrels_doc_ids_not_in_docs": sum(str(r["doc_id"]) not in docs_by_id for r in qrels)}


def env() -> dict:
    e = {"python": platform.python_version(), "platform": platform.platform(),
         "numpy": pv("numpy"), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    return e


mode = sys.argv[1]
t0 = time.time()
if mode == "bm25":
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([tokenize_de(doc_text(docs_by_id[i])) for i in doc_ids], k1=1.5, b=0.75, epsilon=0.25)
    ranked = []
    for q in queries:
        sc = bm.get_scores(tokenize_de(str(q["query_text"])))
        ranked.append([d for d, _ in sorted(zip(doc_ids, sc), key=lambda x: (-float(x[1]), x[0]))[:10]])
    payload = {"experiment": {"retriever": "BM25Okapi", "rank_bm25": pv("rank-bm25"),
               "params": {"k1": 1.5, "b": 0.75, "epsilon": 0.25}, "tokenizer": TOKEN_PATTERN.pattern + " + lowercase",
               "doc_text": "title\\ntext", "k_values": K_VALUES, "relevance_threshold": THR},
               "metrics": compute_metrics(qids, ranked, gold, "germanquad", K_VALUES)}
    fn = "bm25_germanquad_repro.json"
else:
    import os, torch
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import HfApi, snapshot_download
    name = "intfloat/multilingual-e5-large"
    model = SentenceTransformer(name)
    rev = None
    try:
        rev = HfApi().model_info(name).sha
    except Exception as ex:  # noqa: BLE001
        rev = f"unavailable: {ex}"
    from pathlib import Path as P
    snap = getattr(model, "model_card_data", None)
    enc = dict(batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_tensor=True)
    de = model.encode([f"passage: {str(docs_by_id[i].get('text') or '').strip()}" for i in doc_ids], **enc)
    qe = model.encode([f"query: {str(q['query_text']).strip()}" for q in queries], **enc)
    _, idx = (qe @ de.T).topk(k=10, dim=1)
    ranked = [[doc_ids[int(i)] for i in row] for row in idx.tolist()]
    payload = {"experiment": {"retriever": "dense-embedding", "model_name": name, "model_revision_hub_head": rev,
               "sentence_transformers": pv("sentence-transformers"), "torch": torch.__version__,
               "transformers": pv("transformers"), "device": "cpu", "batch_size": 32,
               "query_prefix": "query: ", "document_prefix": "passage: ", "normalize_embeddings": True,
               "similarity": "cosine (dot of normalized embeddings)", "k_values": K_VALUES,
               "relevance_threshold": THR, "hf_home": os.environ.get("HF_HOME")},
               "metrics": compute_metrics(qids, ranked, gold, "germanquad", K_VALUES)}
    fn = "dense_e5_germanquad_repro.json"
payload["environment"] = env()
payload["environment"]["runtime_seconds"] = round(time.time() - t0, 1)
DOC_ID_TYPES["ranking"] = sorted({type(d).__name__ for top in ranked for d in top})
payload["doc_id_types"] = DOC_ID_TYPES
payload["overview"] = {"queries": len(queries), "docs": len(docs), "qrels": len(qrels),
                       "queries_without_relevant_docs": sum(1 for q in qids if q not in gold)}
(OUT / fn).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
print(fn, payload["metrics"], payload["environment"]["runtime_seconds"])
