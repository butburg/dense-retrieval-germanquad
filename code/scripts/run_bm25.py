"""BM25 baseline (same logic as code/03_bm25_retrieval_poc.ipynb) on normalized datasets.

Usage: python code/scripts/run_bm25.py [--datasets germanquad [gerlerb_v2]] [--out-dir code/output/bm25_v2]
Inputs: code/output/germanquad/ (default); code/output/gerlerb_v2/ only when requested with --datasets (normalized jsonl)
Outputs: <out-dir>/{bm25_results.csv,bm25_results.json,per_query_ranks.csv} with one block of rows per selected
dataset (a run without gerlerb_v2 therefore writes no GerLeRB rows).
per_query_ranks.csv holds the full-ranking rank of the best-ranked gold doc per query
(for bootstrap); first_rel_rank is the notebook definition (top-10 only, else empty).
"""
import argparse, json, platform, re, sys
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "pylib"))
from local_dataset_io import load_jsonl  # noqa: E402

K_VALUES = [1, 5, 10]
RELEVANCE_THRESHOLD = 1
BM25_K1, BM25_B, BM25_EPSILON = 1.5, 0.75, 0.25
OUT = CODE / "output" / "bm25_v2"
DATASETS = {name: {r: CODE / "output" / name / f"{r}.normalized.jsonl" for r in ("queries", "docs", "qrels")}
            for name in ("germanquad", "gerlerb_v2")}
TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÄÖÜäöüß]+")


def tokenize_de(text):
    return TOKEN_PATTERN.findall((text or "").lower())


def compose_doc_text(doc):
    title, text = str(doc.get("title") or "").strip(), str(doc.get("text") or "").strip()
    return f"{title}\n{text}" if title and text else title or text


def evaluate(name, paths):
    queries, docs, qrels = (load_jsonl(paths[r]) for r in ("queries", "docs", "qrels"))
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    bm25 = BM25Okapi([tokenize_de(compose_doc_text(docs_by_id[i])) for i in doc_ids],
                     k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)
    gold_by_q, q_with_qrels = {}, set()
    for r in qrels:
        q_with_qrels.add(str(r["query_id"]))
        if int(r["relevance"]) >= RELEVANCE_THRESHOLD and str(r["doc_id"]) in docs_by_id:
            gold_by_q.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))
    max_k, rows = max(K_VALUES), []
    for q in queries:
        qid = str(q["query_id"])
        scores = bm25.get_scores(tokenize_de(str(q["query_text"])))
        ranked = sorted(zip(doc_ids, scores), key=lambda it: (-float(it[1]), it[0]))
        top = [d for d, _ in ranked[:max_k]]
        gold = gold_by_q.get(qid, set())
        first = next((i for i, d in enumerate(top, 1) if d in gold), None)
        gold_ranks = [i for i, (d, _) in enumerate(ranked, 1) if d in gold]
        rows.append({"dataset": name, "query_id": qid, "gold_doc_ids": sorted(gold), "top_doc_ids": top,
                     "first_rel_rank": first, "best_gold_rank_full": min(gold_ranks) if gold_ranks else None,
                     "has_qrels": qid in q_with_qrels})
    n = len(rows)
    metrics = []
    for k in K_VALUES:
        rec = suc = mrr = 0.0
        for r in rows:
            g = set(r["gold_doc_ids"])
            hits = sum(d in g for d in r["top_doc_ids"][:k])
            rec += hits / len(g) if g else 0
            suc += hits > 0
            if r["first_rel_rank"] is not None and r["first_rel_rank"] <= k:
                mrr += 1.0 / r["first_rel_rank"]
        d = float(n) if n else 1.0
        metrics.append({"dataset": name, "k": k, "recall_at_k": rec / d, "success_at_k": suc / d,
                        "mrr_at_k": mrr / d, "num_queries": n})
    overview = {"dataset": name, "queries": len(queries), "docs": len(docs_by_id), "qrels": len(qrels),
                "queries_without_qrels": sum(not r["has_qrels"] for r in rows),
                "queries_without_relevant_docs": sum(not r["gold_doc_ids"] for r in rows)}
    return metrics, overview, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["germanquad"],
                    help="normalized datasets to rank (default: germanquad only)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args(argv)
    out, selected = a.out_dir, {n: DATASETS[n] for n in a.datasets}
    out.mkdir(parents=True, exist_ok=True)
    all_metrics, overviews, all_rows = [], [], []
    for name, paths in selected.items():
        m, o, r = evaluate(name, paths)
        all_metrics += m; overviews.append(o); all_rows += r
        print(name, o)
    mdf = pd.DataFrame(all_metrics).sort_values(["dataset", "k"]).reset_index(drop=True)
    mdf.to_csv(out / "bm25_results.csv", index=False, encoding="utf-8")
    pq = pd.DataFrame(all_rows)
    pq["gold_doc_ids"] = pq["gold_doc_ids"].map(" | ".join)
    pq["top_doc_ids"] = pq["top_doc_ids"].map(" | ".join)
    pq.drop(columns="has_qrels").to_csv(out / "per_query_ranks.csv", index=False, encoding="utf-8")
    payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "retriever": "BM25Okapi", "python_version": platform.python_version(),
            "package": "rank_bm25", "package_version": package_version("rank-bm25"),
            "pandas_version": package_version("pandas"),
            "datasets": {n: {r: str(p.relative_to(CODE)) for r, p in ps.items()} for n, ps in selected.items()},
            "split": "normalized:local", "preprocessing": "regex-tokenization + lowercase",
            "tokenizer_regex": TOKEN_PATTERN.pattern,
            "bm25_params": {"k1": BM25_K1, "b": BM25_B, "epsilon": BM25_EPSILON},
            "tie_break": "(-score, doc_id)", "k_values": K_VALUES, "relevance_threshold": RELEVANCE_THRESHOLD,
            "seed": "not-applicable (BM25 deterministic)", "metrics": ["Recall@k", "Success@k", "MRR@k"],
        },
        "overview": overviews, "metrics": all_metrics,
    }
    (out / "bm25_results.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(mdf.to_string())


if __name__ == "__main__":
    main()
