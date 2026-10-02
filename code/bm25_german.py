"""BM25 baseline with German preprocessing on GermanQuAD (test) -- same corpus, qrels, parameters and ranking as scripts/run_bm25.py.

Usage: python bm25_german.py [--out-dir output/bm25_de]
Inputs: output/germanquad/{queries,docs,qrels}.normalized.jsonl
Outputs: output/bm25_de/{bm25_results.json,bm25_results.csv,per_query_ranks.csv} (format of output/bm25_v2/).
Difference to bm25_v2: tokens are stopword-filtered and Snowball-stemmed (pylib/german_text.py); queries and docs alike.
"""
import argparse, json, platform, sys
from datetime import datetime, timezone
from importlib.metadata import version as pv
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))
from local_dataset_io import doc_text, load_jsonl  # noqa: E402
from retrieval_metrics import build_gold, compute_metrics, first_relevant_ranks  # noqa: E402
from german_text import preprocess_de, preprocessing_info  # noqa: E402
import run_bm25 as base  # noqa: E402  (reuse parameters and dataset paths)

OUT = CODE / "output" / "bm25_de"
NAME = "germanquad"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    out = ap.parse_args(argv).out_dir
    paths = base.DATASETS[NAME]
    queries, docs, qrels = (load_jsonl(paths[r]) for r in ("queries", "docs", "qrels"))
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    bm25 = BM25Okapi([preprocess_de(doc_text(docs_by_id[i])) for i in doc_ids],
                     k1=base.BM25_K1, b=base.BM25_B, epsilon=base.BM25_EPSILON)
    gold = build_gold(qrels, set(doc_ids), base.RELEVANCE_THRESHOLD)
    qids, ranked, rows = [], [], []
    for q in queries:
        qid = str(q["query_id"])
        scores = bm25.get_scores(preprocess_de(str(q["query_text"])))
        full = sorted(zip(doc_ids, scores), key=lambda it: (-float(it[1]), it[0]))
        top, g = [d for d, _ in full[:max(base.K_VALUES)]], gold.get(qid, set())
        gr = [i for i, (d, _) in enumerate(full, 1) if d in g]
        qids.append(qid); ranked.append(top)
        rows.append({"dataset": NAME, "query_id": qid, "gold_doc_ids": " | ".join(sorted(g)),
                     "top_doc_ids": " | ".join(top), "first_rel_rank": first_relevant_ranks([qid], [top], gold)[0],
                     "best_gold_rank_full": min(gr) if gr else None})
    # bm25_v2 format has no precision column
    metrics = [{k: v for k, v in m.items() if k != "precision_at_k"}
               for m in compute_metrics(qids, ranked, gold, NAME, base.K_VALUES)]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "bm25_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "per_query_ranks.csv", index=False, encoding="utf-8")
    payload = {"run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": {"retriever": "BM25Okapi + German preprocessing", "python_version": platform.python_version(),
                              "package": "rank_bm25", "package_version": pv("rank-bm25"),
                              "datasets": {NAME: {r: str(p.relative_to(CODE)) for r, p in paths.items()}},
                              "split": "normalized:local (same files as bm25_v2)",
                              "preprocessing": preprocessing_info(),
                              "bm25_params": {"k1": base.BM25_K1, "b": base.BM25_B, "epsilon": base.BM25_EPSILON},
                              "tie_break": "(-score, doc_id)", "k_values": base.K_VALUES,
                              "relevance_threshold": base.RELEVANCE_THRESHOLD,
                              "seed": "not-applicable (BM25 deterministic)"},
               "overview": {"queries": len(queries), "docs": len(docs_by_id), "qrels": len(qrels),
                            "queries_without_relevant_docs": sum(not r["gold_doc_ids"] for r in rows)},
               "metrics": metrics}
    (out / "bm25_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string())


if __name__ == "__main__":
    main()
