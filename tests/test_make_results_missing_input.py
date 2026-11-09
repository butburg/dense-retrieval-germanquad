"""make_results aborts with a clear German message when a BM25 input (rank file or stored metrics) is missing."""
import json
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "pylib"))
import compare_models as cm  # noqa: E402
import make_results  # noqa: E402


def run_main_with_path(monkeypatch, tmp_path, attribute, path):
    """Point ``cm.<attribute>`` to ``path`` and run make_results.main() with a temporary output directory."""
    monkeypatch.setattr(cm, attribute, path)
    monkeypatch.setattr(sys, "argv", ["make_results.py", "--out-dir", str(tmp_path / "results")])
    if not (CODE / "output" / cm.DS / "queries.normalized.jsonl").exists():
        pytest.skip("raw dataset not available in code/output")
    make_results.main()


@pytest.mark.parametrize("attribute, filename, command", [
    ("BM25_CSV", "per_query_ranks.csv", "python scripts/run_bm25.py"),
    ("BM25_JSON", "bm25_results.json", "python scripts/run_bm25.py"),
    ("BM25_DE_CSV", "per_query_ranks.csv", "python bm25_german.py"),
    ("BM25_DE_JSON", "bm25_results.json", "python bm25_german.py")])
def test_missing_bm25_input_exits_with_clear_message(monkeypatch, tmp_path, attribute, filename, command):
    missing = tmp_path / attribute / filename

    with pytest.raises(SystemExit) as excinfo:
        run_main_with_path(monkeypatch, tmp_path, attribute, missing)

    message = str(excinfo.value)
    assert str(missing) in message
    assert command in message
    assert not (tmp_path / "results" / "table_main.md").exists()


@pytest.mark.parametrize("attribute, original", [("BM25_JSON", cm.BM25_JSON), ("BM25_DE_JSON", cm.BM25_DE_JSON)])
def test_tampered_bm25_metrics_fail_consistency_check(monkeypatch, tmp_path, attribute, original):
    if not original.exists():
        pytest.skip("BM25 results not available in code/output")
    stored = json.loads(original.read_text())
    for row in stored["metrics"]:
        row["mrr_at_k"] += 0.01  # far above cm.CONSISTENCY_TOL
    tampered = tmp_path / "bm25_results.json"
    tampered.write_text(json.dumps(stored))

    with pytest.raises(AssertionError, match="metrics json differs"):
        run_main_with_path(monkeypatch, tmp_path, attribute, tampered)
