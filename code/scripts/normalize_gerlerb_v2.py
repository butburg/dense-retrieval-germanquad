"""Re-normalize GerLeRB with the fixed TREC parser into code/output/gerlerb_v2/.

Usage: python code/scripts/normalize_gerlerb_v2.py
Inputs: code/input/gerlerb/{corpus.trec.gz,qrels.txt,topics.txt}
Outputs: code/output/gerlerb_v2/{queries,docs,qrels}.normalized*.jsonl, summary.json
"""
import json, sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "pylib"))
from dataset_normalization import check_id_consistency, dedupe_normalized, normalize_docs, normalize_qrels, normalize_queries  # noqa: E402
from local_dataset_io import parse_trec_corpus, parse_tsv, write_jsonl  # noqa: E402

IN, OUT = CODE / "input" / "gerlerb", CODE / "output" / "gerlerb_v2"
docs_raw, stats = parse_trec_corpus(IN / "corpus.trec.gz")
norm = {
    "queries": normalize_queries(parse_tsv(IN / "topics.txt"), source="gerlerb", split="local",
                                 id_keys=["qid"], text_keys=["query"]),
    "docs": normalize_docs(docs_raw, source="gerlerb", split="local", id_keys=["docno"], text_keys=["text"]),
    "qrels": normalize_qrels(parse_tsv(IN / "qrels.txt"), source="gerlerb", split="local",
                             query_id_keys=["qid"], doc_id_keys=["docno"], score_keys=["label"], default_score=1),
}
for d in norm["docs"]:  # same schema as code/output/gerlerb (title field, empty)
    d["title"] = ""
raw_docs_rows = len(norm["docs"])
norm = dedupe_normalized(norm)
cons = check_id_consistency(norm["queries"], norm["docs"], norm["qrels"])
for role, rows in norm.items():
    write_jsonl(OUT / f"{role}.normalized.jsonl", rows)
summary = {
    "dataset": "gerlerb_v2",
    "note": "parse_trec_corpus fixed for combined '</DOC><DOC>' lines (dls-btn.16)",
    "parser_differences": stats,
    "parsed_doc_rows_before_dedupe": raw_docs_rows,
    "normalized_counts": {k: len(v) for k, v in norm.items()},
    "duplicate_doc_rows_dropped": raw_docs_rows - len(norm["docs"]),
    "id_consistency": cons,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
print(json.dumps(summary, indent=1))
