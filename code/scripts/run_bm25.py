"""BM25 baseline (same logic as code/03_bm25_retrieval_poc.ipynb) on normalized datasets.

Usage: python code/scripts/run_bm25.py [--out-dir code/output/bm25_v2]
Inputs: code/output/germanquad/{queries,docs,qrels}.normalized.jsonl
Outputs: <out-dir>/{bm25_results.csv,bm25_results.json,per_query_ranks.csv}.
per_query_ranks.csv holds the full-ranking rank of the best-ranked gold doc per query
(for bootstrap); first_rel_rank is the notebook definition (top-10 only, else empty).

Role in the experiment: lexical baseline "BM25" (regex tokenization + lowercase, no stopwords, no stemming).
Runs before bm25_german.py, which imports DATASETS and the BM25 parameters from this module
(K_VALUES and RELEVANCE_THRESHOLD come from pylib/retrieval_metrics.py).
compare_models.py and make_results.py read the three output files (F2 comparisons against BM25).
Qrels: a document is gold for a query if its relevance grade is >= RELEVANCE_THRESHOLD and it is in the corpus.
"""
import argparse, json, platform, sys
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "pylib"))
from german_text import TOKEN_PATTERN, tokenize_de  # noqa: E402
from local_dataset_io import doc_text, load_jsonl  # noqa: E402
from retrieval_metrics import (K_VALUES, RELEVANCE_THRESHOLD, best_gold_rank, build_gold, compute_metrics,  # noqa: E402
                               first_relevant_ranks)

BM25_K1, BM25_B, BM25_EPSILON = 1.5, 0.75, 0.25  # rank_bm25 BM25Okapi parameters (library defaults)
OUT = CODE / "output" / "bm25_v2"  # default output directory
# Dataset name -> paths of the normalized files (written by notebook 02_dataset_parsing_inspection).
# Dict comprehension per role (queries/docs/qrels): role name -> path of its normalized JSONL file.
DATASETS = {"germanquad": {r: CODE / "output" / "germanquad" / f"{r}.normalized.jsonl"
                           for r in ("queries", "docs", "qrels")}}


def rank_corpus(bm25: BM25Okapi, query_tokens: list[str], doc_ids: list[str]) -> list[tuple[str, float]]:
    """Rank the whole corpus for one tokenized query; the single definition of the BM25 ranking and its tie rule.

    Args:
        bm25: Index whose row ``i`` belongs to ``doc_ids[i]``.
        query_tokens: Query tokens produced by the same tokenizer as the indexed documents,
            e.g. ``["wer", "entwickelte", "aufzug"]``.
        doc_ids: Document IDs in index order, e.g. ``["c1", "c2"]``.

    Returns:
        ``(doc_id, score)`` for every document, best first. Equal scores are ordered by ``doc_id`` ascending,
        so the ranking is deterministic. Scores are plain floats.
    """
    scores = bm25.get_scores(query_tokens)
    # The generator turns each NumPy score into a plain float; zip pairs it with its doc ID, e.g.
    # [("a", 2.1), ("b", 7.3)]; sorting by (-score, doc_id) yields the ranking, ties broken by doc ID.
    return sorted(zip(doc_ids, (float(score) for score in scores)), key=lambda it: (-it[1], it[0]))


def evaluate(name, paths):
    """Rank the whole corpus with BM25 for every query and compute metrics and per-query ranks.

    Args:
        name: Dataset label written into every result row, e.g. ``"germanquad"``.
        paths: Mapping ``{"queries": Path, "docs": Path, "qrels": Path}`` to normalized JSONL files.

    Returns:
        Tuple ``(metrics, overview, rows)``. ``metrics`` is one dict per k in ``K_VALUES``
        (without the precision column). ``overview`` holds corpus and query counts.
        ``rows`` has one dict per query with gold IDs, top-``max(K_VALUES)`` IDs,
        ``first_rel_rank`` (top-10 only) and ``best_gold_rank_full`` (rank over the full corpus).
    """
    # --- Load data and build the BM25 index (one token list per document) ---
    # Generator over the three roles: load each normalized JSONL file; unpacking gives the three record lists.
    queries, docs, qrels = (load_jsonl(paths[r]) for r in ("queries", "docs", "qrels"))
    # Dict comprehension: doc ID -> doc record (lookup table; a repeated ID would keep the last record).
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    # List comprehension over doc_ids: one token list per document (query-independent BM25 index input);
    # row i of the index belongs to doc_ids[i].
    bm25 = BM25Okapi([tokenize_de(doc_text(docs_by_id[i])) for i in doc_ids],
                     k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)
    # gold: query_id -> set of relevant doc_ids (qrels filtered by threshold and corpus membership)
    gold = build_gold(qrels, set(doc_ids), RELEVANCE_THRESHOLD)
    # Set comprehension: IDs of all queries that appear in qrels at all (to flag queries without any judgment).
    q_with_qrels = {str(r["query_id"]) for r in qrels}
    # qids: query IDs in input order; ranked: top-max_k doc IDs per query (same order as qids);
    # best_full: best gold rank over the full corpus per query (None if the query has no gold doc)
    max_k, qids, ranked, best_full = max(K_VALUES), [], [], []
    # --- Rank the full corpus per query ---
    # For every query: BM25-score the whole corpus, sort all documents by score, keep the top max_k doc IDs
    # and the best rank of any gold document (used for Recall/MRR@k and the bootstrap tests).
    for q in queries:
        qid = str(q["query_id"])
        full = rank_corpus(bm25, tokenize_de(str(q["query_text"])), doc_ids)
        # Remember this query ID and its top-max_k doc IDs (list comprehension takes the ID of each of the first max_k
        # pairs).
        qids.append(qid); ranked.append([d for d, _ in full[:max_k]])
        # Best (smallest) gold rank over the full corpus, None if the query has no gold document in the corpus.
        best_full.append(best_gold_rank(full, gold.get(qid, set())))
    # --- Per-query rows and metrics ---
    # first_rel_rank: Notebook-Definition (nur Top-10, sonst leer)
    first = first_relevant_ranks(qids, ranked, gold)
    # List comprehension over the parallel lists (one element per query, zip strict=True checks equal length):
    # builds one per-query record with gold IDs, top-k IDs, first relevant rank in the top 10 and best full-corpus
    # gold rank.
    rows = [{"dataset": name, "query_id": qid, "gold_doc_ids": sorted(gold.get(qid, set())), "top_doc_ids": top,
             "first_rel_rank": fr, "best_gold_rank_full": bf, "has_qrels": qid in q_with_qrels}
            for qid, top, fr, bf in zip(qids, ranked, first, best_full, strict=True)]
    # das bm25_v2-Format hat keine Precision-Spalte (precision_at_k is dropped for format compatibility)
    # Per metric row (one per k): dict comprehension copies all columns except precision_at_k.
    metrics = [{k: v for k, v in m.items() if k != "precision_at_k"}
               for m in compute_metrics(qids, ranked, gold, name, K_VALUES)]
    overview = {"dataset": name, "queries": len(queries), "docs": len(docs_by_id), "qrels": len(qrels),
                # Counts per-query rows without any qrels entry (sum over booleans: True counts as 1).
                "queries_without_qrels": sum(not r["has_qrels"] for r in rows),
                # Counts queries whose gold set is empty after threshold and corpus filter.
                "queries_without_relevant_docs": sum(not r["gold_doc_ids"] for r in rows)}
    return metrics, overview, rows


def main(argv=None):
    """Run the BM25 baseline for all datasets in ``DATASETS`` and write CSV/JSON results.

    Writes ``bm25_results.csv``, ``bm25_results.json`` and ``per_query_ranks.csv`` into ``--out-dir``
    (existing files are overwritten) and prints overview and metrics.

    Args:
        argv: Command-line arguments, e.g. ``["--out-dir", "output/bm25_v2"]``; ``None`` reads ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args(argv)
    out, selected = a.out_dir, DATASETS
    out.mkdir(parents=True, exist_ok=True)
    # --- Evaluate each dataset and collect results ---
    all_metrics, overviews, all_rows = [], [], []
    # For every dataset (currently only GermanQuAD): run the BM25 evaluation and collect metrics, overview and
    # per-query rows.
    for name, paths in selected.items():
        m, o, r = evaluate(name, paths)
        all_metrics += m; overviews.append(o); all_rows += r
        print(name, o)
    # --- Write results ---
    # One table of all metric rows, sorted by dataset and k for stable CSV output.
    mdf = pd.DataFrame(all_metrics).sort_values(["dataset", "k"]).reset_index(drop=True)
    mdf.to_csv(out / "bm25_results.csv", index=False, encoding="utf-8")
    pq = pd.DataFrame(all_rows)  # pq: per-query table; ID lists become " | "-joined strings for CSV
    # .map(" | ".join) applies join to every cell, turning a list into a string, e.g. ["a", "b"] -> "a | b" (CSV
    # cannot store lists).
    pq["gold_doc_ids"] = pq["gold_doc_ids"].map(" | ".join)
    # Same conversion for the top-k doc ID lists.
    pq["top_doc_ids"] = pq["top_doc_ids"].map(" | ".join)
    pq.drop(columns="has_qrels").to_csv(out / "per_query_ranks.csv", index=False, encoding="utf-8")
    # JSON file: provenance (versions, parameters, tie rule) + overview + metrics
    payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "retriever": "BM25Okapi", "python_version": platform.python_version(),
            "package": "rank_bm25", "package_version": package_version("rank-bm25"),
            "pandas_version": package_version("pandas"),
            # Dict comprehension: for each dataset and role the file path relative to code/, stored as provenance in
            # the JSON.
            "datasets": {n: {r: str(p.relative_to(CODE)) for r, p in ps.items()} for n, ps in selected.items()},
            "split": "normalized:local", "preprocessing": "regex-tokenization + lowercase",
            "tokenizer_regex": TOKEN_PATTERN.pattern,
            "bm25_params": {"k1": BM25_K1, "b": BM25_B, "epsilon": BM25_EPSILON},
            "tie_break": "(-score, doc_id)", "k_values": list(K_VALUES), "relevance_threshold": RELEVANCE_THRESHOLD,
            "seed": "not-applicable (BM25 deterministic)", "metrics": ["Recall@k", "Success@k", "MRR@k"],
        },
        "overview": overviews, "metrics": all_metrics,
    }
    (out / "bm25_results.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(mdf.to_string())


if __name__ == "__main__":
    main()
