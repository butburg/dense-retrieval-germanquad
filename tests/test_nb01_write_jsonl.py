"""Equivalence of the former NB01 write_jsonl and pylib's local_dataset_io.write_jsonl.

Criterion: content read back with load_jsonl is identical (bytes need not be).
"""

import json
import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from pylib import local_dataset_io as io  # noqa: E402

NOTEBOOK = CODE_DIR / "01_dataset_download.ipynb"

# Verbatim copy of the function removed from NB01 (threshold passed in for testability).
def old_nb01_write_jsonl(path, records, jsonl_split_threshold=57_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_parts = sorted(path.parent.glob(f"{path.stem}.part*{path.suffix}"))
    for stale_part in stale_parts:
        stale_part.unlink()

    written = 0
    handle = path.open("w", encoding="utf-8")
    try:
        for row in records:
            if written > 0 and written % jsonl_split_threshold == 0:
                handle.close()
                part_number = written // jsonl_split_threshold + 1
                part_path = path.with_name(f"{path.stem}.part{part_number}{path.suffix}")
                handle = part_path.open("w", encoding="utf-8")
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1
    finally:
        handle.close()
    return written


def _records(count):
    return [{"id": str(i), "text": f"Größe, Übung, ß, „Zitat“, é, 日本, \U0001F600 {i}\nZeile", "n": i} for i in range(count)]


def _file_names(directory):
    return sorted(p.name for p in directory.iterdir())


@pytest.mark.parametrize("count, threshold", [(0, 3), (1, 3), (3, 3), (4, 3), (10, 3), (7, 1)])
def test_same_content_names_and_return_value(tmp_path, monkeypatch, count, threshold):
    monkeypatch.setattr(io, "JSONL_SPLIT_THRESHOLD", threshold)
    records = _records(count)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_count = old_nb01_write_jsonl(old_dir / "corpus.jsonl", records, threshold)
    new_count = io.write_jsonl(new_dir / "corpus.jsonl", iter(records))
    assert old_count == new_count == count
    assert _file_names(old_dir) == _file_names(new_dir)
    assert io.load_jsonl(old_dir / "corpus.jsonl") == io.load_jsonl(new_dir / "corpus.jsonl") == records


def test_same_content_above_real_threshold(tmp_path):
    assert io.JSONL_SPLIT_THRESHOLD == 57_000
    records = [{"id": i} for i in range(57_000 * 2 + 5)]
    old_count = old_nb01_write_jsonl(tmp_path / "old" / "c.jsonl", records)
    new_count = io.write_jsonl(tmp_path / "new" / "c.jsonl", records)
    assert old_count == new_count == len(records)
    assert _file_names(tmp_path / "old") == _file_names(tmp_path / "new") == ["c.jsonl", "c.part2.jsonl", "c.part3.jsonl"]
    assert io.load_jsonl(tmp_path / "old" / "c.jsonl") == io.load_jsonl(tmp_path / "new" / "c.jsonl") == records


def test_stale_parts_removed_in_both(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "JSONL_SPLIT_THRESHOLD", 2)
    for name, writer in (("old", lambda p, r: old_nb01_write_jsonl(p, r, 2)), ("new", io.write_jsonl)):
        target = tmp_path / name / "q.jsonl"
        writer(target, _records(7))
        writer(target, _records(3))
        assert [r["id"] for r in io.load_jsonl(target)] == ["0", "1", "2"]
        assert _file_names(tmp_path / name) == ["q.jsonl", "q.part2.jsonl"]


def test_nb01_uses_pylib_write_jsonl():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "def write_jsonl" not in source
    assert "57_000" not in source
    assert "from local_dataset_io import write_jsonl" in source
