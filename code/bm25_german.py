"""BM25 baseline with German preprocessing on GermanQuAD (test) -- same corpus, qrels, parameters and ranking as scripts/run_bm25.py.

Usage: python bm25_german.py [--out-dir output/bm25_de]
Inputs: output/germanquad/{queries,docs,qrels}.normalized.jsonl
Outputs: output/bm25_de/{bm25_results.json,bm25_results.csv,per_query_ranks.csv} (format of output/bm25_v2/).
Difference to bm25_v2: tokens are stopword-filtered and Snowball-stemmed (pylib/german_text.py); queries and docs alike.

Role in the experiment: second lexical baseline "BM25-de" (family F3: dense models vs. BM25-de; F4: BM25-de vs.
BM25). Run after scripts/run_bm25.py, whose dataset paths and BM25 parameters are imported (``base``); k values and
relevance threshold come from pylib/retrieval_metrics.py, so both baselines differ only in preprocessing. compare_models.py and make_results.py read the
outputs. Deviations from the bm25_v2 files: the JSON ``overview`` is a single dict (bm25_v2: list with one dict per
dataset) and the JSON is written with ``ensure_ascii=False``.
"""
import argparse, json, platform, sys
from datetime import datetime, timezone
from importlib.metadata import version as pv
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))
from local_dataset_io import load_jsonl  # noqa: E402
from retrieval_metrics import (K_VALUES, RELEVANCE_THRESHOLD, best_gold_rank, build_gold, compute_metrics,  # noqa: E402
                               first_relevant_ranks)
from german_text import preprocess_de, preprocessing_info  # noqa: E402
import run_bm25 as base  # noqa: E402  (reuse parameters and dataset paths)

OUT = CODE / "output" / "bm25_de"  # default output directory
NAME = "germanquad"  # dataset key in run_bm25.DATASETS


def main(argv=None) -> None:
    """Rank GermanQuAD with BM25 on stemmed, stopword-free tokens and write metrics and per-query ranks.

    Writes ``bm25_results.csv``, ``bm25_results.json`` and ``per_query_ranks.csv`` into ``--out-dir``
    (existing files are overwritten) and prints the metrics table.

    Args:
        argv: Command-line arguments, e.g. ``["--out-dir", "output/bm25_de"]``; ``None`` reads ``sys.argv``.
    """
    # --- Load data and build the BM25 index on preprocessed documents ---
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    out = ap.parse_args(argv).out_dir
    paths = base.DATASETS[NAME]
    # Generator over the three roles: load each normalized JSONL file; unpacking gives the three record lists.
    queries, docs, qrels = (load_jsonl(paths[r]) for r in ("queries", "docs", "qrels"))
    # Dict comprehension: doc ID -> doc record (lookup table; a repeated ID would keep the last record).
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    # Index over preprocessed (stopword-free, stemmed) tokens; row i belongs to doc_ids[i].
    bm25 = base.build_index(doc_ids, docs_by_id, preprocess_de)
    # gold: query_id -> set of relevant doc_ids (qrels filtered by threshold and corpus membership)
    gold = build_gold(qrels, set(doc_ids), RELEVANCE_THRESHOLD)
    # qids: query IDs in input order; ranked: top-max(k) doc IDs per query (same order as qids);
    # rows: one per-query record for per_query_ranks.csv
    qids, ranked, rows = [], [], []
    # --- Rank the full corpus per query (query gets the same preprocessing as the documents) ---
    # For every query: BM25-score the whole corpus, sort all documents by score, keep the top max(k) doc IDs
    # and the best rank of any gold document (used for Recall/MRR@k and the bootstrap tests).
    for q in queries:
        qid = str(q["query_id"])
        full = base.rank_corpus(bm25, preprocess_de(str(q["query_text"])), doc_ids)
        # top: top-max(k) doc IDs; g: gold doc IDs of this query
        # List comprehension: doc IDs of the first max(k) pairs of the ranking.
        top, g = [d for d, _ in full[:max(K_VALUES)]], gold.get(qid, set())
        qids.append(qid); ranked.append(top)
        rows.append({"dataset": NAME, "query_id": qid, "gold_doc_ids": " | ".join(sorted(g)),
                     "top_doc_ids": " | ".join(top), "first_rel_rank": first_relevant_ranks([qid], [top], gold)[0],
                     "best_gold_rank_full": best_gold_rank(full, g)})
    # --- Metrics and output files ---
    # bm25_v2 format has no precision column
    # Per metric row (one per k): dict comprehension copies all columns except precision_at_k.
    metrics = [{k: v for k, v in m.items() if k != "precision_at_k"}
               for m in compute_metrics(qids, ranked, gold, NAME, K_VALUES)]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "bm25_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "per_query_ranks.csv", index=False, encoding="utf-8")
    payload = {"run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": {"retriever": "BM25Okapi + German preprocessing", "python_version": platform.python_version(),
                              "package": "rank_bm25", "package_version": pv("rank-bm25"),
                              # Dict comprehension: for each role the file path relative to code/, stored as
                              # provenance in the JSON.
                              "datasets": {NAME: {r: str(p.relative_to(CODE)) for r, p in paths.items()}},
                              "split": "normalized:local (same files as bm25_v2)",
                              "preprocessing": preprocessing_info(),
                              "bm25_params": {"k1": base.BM25_K1, "b": base.BM25_B, "epsilon": base.BM25_EPSILON},
                              "tie_break": "(-score, doc_id)", "k_values": list(K_VALUES),
                              "relevance_threshold": RELEVANCE_THRESHOLD,
                              "seed": "not-applicable (BM25 deterministic)"},
               "overview": {"queries": len(queries), "docs": len(docs_by_id), "qrels": len(qrels),
                            # Counts per-query rows whose gold-ID string is empty (sum over booleans: True counts as
                            # 1).
                            "queries_without_relevant_docs": sum(not r["gold_doc_ids"] for r in rows)},
               "metrics": metrics}
    (out / "bm25_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string())


if __name__ == "__main__":
    main()
