"""Audit of GerLeRB TREC corpus parsing (read-only; prints JSON to stdout).

Usage: python code/scripts/audit_gerlerb.py
Inputs: code/input/gerlerb/{corpus.trec.gz,qrels.txt}, code/output/gerlerb/docs.normalized*.jsonl
"""
import collections, gzip, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pylib.local_dataset_io import load_jsonl, parse_trec_corpus  # noqa: E402

raw = ROOT / "input/gerlerb/corpus.trec.gz"
docs, stats = parse_trec_corpus(raw)

# Independent block-based scan: split raw text on <DOC> ... </DOC>
txt = gzip.open(raw, "rt", encoding="utf-8", errors="ignore").read()
blocks = re.findall(r"<DOC>(.*?)</DOC>", txt, flags=re.S)
docno_re = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.S)
text_re = re.compile(r"<TEXT[^>]*>(.*?)</TEXT>", re.S)
per_block = collections.Counter(len(docno_re.findall(b)) for b in blocks)
multi = [(docno_re.findall(b), len(text_re.findall(b))) for b in blocks if len(docno_re.findall(b)) > 1]
raw_docno_lines = len(re.findall(r"^<DOCNO>", txt, flags=re.M))
raw_docno_any = txt.count("<DOCNO>")
raw_doc_open = len(re.findall(r"^<DOC>$", txt, flags=re.M))

ids = [d["docno"] for d in docs]
dup = {k: v for k, v in collections.Counter(ids).items() if v > 1}
empty = [d["docno"] for d in docs if not d["text"]]
dash = [d["docno"] for d in docs if d["text"].strip() == "-"]
norm = load_jsonl(ROOT / "output/gerlerb/docs.normalized.jsonl")
norm_ids = {d["doc_id"] for d in norm}
qrel_docs = {l.split("\t")[2] for l in (ROOT / "input/gerlerb/qrels.txt").read_text().splitlines()[1:] if l.strip()}
missing_from_norm = sorted(set(ids) - norm_ids)
dropped_dup = sum(v - 1 for v in dup.values())

# Reference parse: regex over complete <DOC>...</DOC> blocks
ref = []
for b in blocks:
    m = docno_re.search(b)
    ref.append((m.group(1).strip(), " ".join(" ".join(t.split()) for t in text_re.findall(b)).strip()))
ref_ids = [r[0] for r in ref]
ref_dup = {k: v for k, v in collections.Counter(ref_ids).items() if v > 1}
ref_first = {}
for i, t in ref:
    ref_first.setdefault(i, t)
lines_joined = len(re.findall(r"^</DOC><DOC>$", txt, flags=re.M))
parsed_text = {}
for d in docs:
    parsed_text.setdefault(d["docno"], " ".join(d["text"].split()))
text_differs = sorted(i for i in parsed_text if parsed_text[i] != ref_first.get(i))
qrel_text_differs = sorted(qrel_docs & set(text_differs))
ref_empty = [i for i, t in ref if not t]
lost_ids = sorted(set(ref_ids) - set(ids))
ex = qrel_text_differs[0] if qrel_text_differs else None
print(json.dumps({
    "ref_ids_lost_entirely_in_parser": len(lost_ids), "lost_examples": lost_ids[:5],
    "qrels_docs_lost_in_parser": len(qrel_docs & set(lost_ids)),
    "example_qrel_doc": ex, "example_len_parser": len(parsed_text.get(ex, "")), "example_len_ref": len(ref_first.get(ex, "")),
    "blocks_lost_by_merge_total": len(ref) - len(ids),
    "empty_ref_blocks_all_text_empty_in_parser_excerpt": len([i for i in set(ref_empty) if parsed_text.get(i) == ""]),
}, indent=1))
print(json.dumps({
    "joined_close_open_lines": lines_joined,
    "ref_blocks": len(ref), "ref_unique_ids": len(set(ref_ids)), "ref_duplicate_ids": len(ref_dup),
    "ref_empty_text_blocks": len(ref_empty),
    "ref_empty_text_ids_in_qrels": len(qrel_docs & set(ref_empty)),
    "first_occurrence_text_differs_parser_vs_ref": len(text_differs),
    "qrels_docs_text_differs": len(qrel_text_differs), "qrels_text_differs_examples": qrel_text_differs[:5],
    "qrels_docs_with_duplicate_ref_ids": sorted(qrel_docs & set(ref_dup))[:10],
    "n_qrels_docs_with_duplicate_ref_ids": len(qrel_docs & set(ref_dup)),
}, indent=1, ensure_ascii=False))
print(json.dumps({
    "parser_stats": stats,
    "raw_docno_at_line_start": raw_docno_lines, "raw_docno_substring": raw_docno_any, "raw_doc_open_lines": raw_doc_open,
    "regex_blocks": len(blocks), "docnos_per_block_histogram": dict(per_block),
    "blocks_with_multiple_docno": len(multi), "multi_examples": multi[:3],
    "parsed_unique_ids": len(set(ids)), "duplicate_ids": len(dup), "duplicate_extra_rows": dropped_dup,
    "duplicate_examples": dict(list(dup.items())[:5]),
    "empty_text_docs": len(empty), "empty_examples": empty[:10],
    "dash_only_text_docs": len(dash),
    "normalized_docs": len(norm), "normalized_unique_ids": len(norm_ids),
    "parsed_ids_missing_in_normalized": len(missing_from_norm), "missing_examples": missing_from_norm[:5],
    "qrels_doc_ids": len(qrel_docs), "qrels_docs_in_parsed": len(qrel_docs & set(ids)),
    "qrels_docs_in_normalized": len(qrel_docs & norm_ids),
    "qrels_docs_empty_text": len(qrel_docs & set(empty)),
    "qrels_docs_duplicate_ids": sorted(qrel_docs & set(dup)),
}, indent=1, ensure_ascii=False))
