"""Regression tests for code/pylib/local_dataset_io.py (synthetic data only)."""

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pylib import local_dataset_io as io  # noqa: E402


def _write(path, ids):
    path.write_text("".join(f'{{"id": {i}}}\n' for i in ids), encoding="utf-8")


def test_load_jsonl_part_order_numeric_not_lexicographic(tmp_path):
    base = tmp_path / "docs.jsonl"
    _write(base, [0])
    for n in (2, 9, 10, 11):
        _write(tmp_path / f"docs.part{n}.jsonl", [n])
    assert [r["id"] for r in io.load_jsonl(base)] == [0, 2, 9, 10, 11]


def test_write_jsonl_split_threshold_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "JSONL_SPLIT_THRESHOLD", 2)
    base = tmp_path / "out.jsonl"
    n = io.write_jsonl(base, ({"id": i} for i in range(23)))
    assert n == 23
    assert (tmp_path / "out.part10.jsonl").exists()
    assert not (tmp_path / "out.part13.jsonl").exists()
    assert [r["id"] for r in io.load_jsonl(base)] == list(range(23))


def test_write_jsonl_removes_stale_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "JSONL_SPLIT_THRESHOLD", 2)
    base = tmp_path / "out.jsonl"
    io.write_jsonl(base, ({"id": i} for i in range(7)))
    assert (tmp_path / "out.part4.jsonl").exists()
    io.write_jsonl(base, ({"id": i} for i in range(3)))
    assert not (tmp_path / "out.part3.jsonl").exists()
    assert [r["id"] for r in io.load_jsonl(base)] == [0, 1, 2]


def test_parse_trec_inline_and_multipart_text(tmp_path):
    content = (
        "<DOC>\n<DOCNO>A1</DOCNO>\n<TEXT>inline text</TEXT>\n</DOC>\n"
        "<DOC>\n<DOCNO>B2</DOCNO>\n<TEXT>\nline one\nline two\n</TEXT>\n</DOC>\n"
        "<DOC>\n<DOCNO>C3</DOCNO>\n<TEXT>part one</TEXT>\n<TEXT>part two</TEXT>\n</DOC>\n"
    )
    p = tmp_path / "c.trec.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(content)
    docs, stats = io.parse_trec_corpus(p)
    by = {d["docno"]: d["text"] for d in docs}
    assert by["A1"] == "inline text"
    assert by["B2"] == "line one line two"
    # Multiple <TEXT> blocks of one document are concatenated.
    assert by["C3"] == "part one part two"
    assert stats["doc_open_tags"] == 3 and stats["parsed_docs"] == 3


def test_parse_trec_corpus_combined_close_open_line(tmp_path):
    raw = (
        "<DOC>\n<DOCNO>a</DOCNO>\n<TEXT>\ntext a\n</TEXT>\n</DOC><DOC>\n"
        "<DOCNO>b</DOCNO>\n<TEXT>text b</TEXT>\n</DOC>\n"
    )
    path = tmp_path / "c.trec.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(raw)
    docs, stats = io.parse_trec_corpus(path)
    assert docs == [{"docno": "a", "text": "text a"}, {"docno": "b", "text": "text b"}]
    assert stats["line_count"] == 9
    assert stats["doc_open_tags"] == stats["doc_close_tags"] == 2
