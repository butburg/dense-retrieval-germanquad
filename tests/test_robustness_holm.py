"""Robustheits-Holm über neun Modelle: die zwei kippenden Urteile (Variante B)."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "pylib"))
from robustness_holm import add_holm, load_rows  # noqa: E402


def _rows():
    return {(r["model"], r["metric"]): r for r in add_holm(load_rows())}


def test_only_two_verdicts_flip_in_b_none_in_a():
    rows = _rows().values()
    assert not any(r["urteil_aendert_sich_A"] for r in rows)
    flips = {(r["model"], r["metric"]) for r in rows if r["urteil_aendert_sich_B"]}
    assert flips == {("jina-embeddings-v3", "Success@5"), ("snowflake-arctic-embed-l-v2.0", "Success@1")}


def test_flipped_values():
    j = _rows()[("jina-embeddings-v3", "Success@5")]
    s = _rows()[("snowflake-arctic-embed-l-v2.0", "Success@1")]
    assert round(j["p_holm_old"], 4) == 0.0091 and round(j["p_holm_B"], 4) == 0.0512
    assert round(s["p_holm_old"], 4) == 0.0058 and round(s["p_holm_B"], 4) == 0.0521
