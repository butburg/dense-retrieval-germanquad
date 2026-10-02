"""QS des Testdesigns: Testzahl, Familiengrößen, Holm-Werte, Wortwahl und Übereinstimmung mit dem Vorgängerstand."""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "pylib"))
from significance import holm  # noqa: E402

JSON = CODE / "output" / "harness" / "comparison_germanquad.json"
SIZES = {"F1": 36, "F2": 9, "F3": 9, "F4": 1}


@pytest.fixture(scope="module")
def tests():
    return json.loads(JSON.read_text())["tests"]


def test_test_count_and_family_sizes(tests):
    assert len(tests) == 55
    assert Counter(t["family"][:2] for t in tests) == SIZES
    assert {t["metric"] for t in tests} == {"MRR@10"}


def test_holm_per_family_uses_actual_size(tests):
    for fam in SIZES:
        g = [t for t in tests if t["family"].startswith(fam)]
        assert np.allclose(holm([t["p_cluster_signflip"] for t in g]), [t["p_holm"] for t in g]), fam
    f4 = next(t for t in tests if t["family"].startswith("F4"))
    assert f4["p_holm"] == f4["p_cluster_signflip"]


def test_family_contents(tests):
    models = {t["b"] for t in tests if t["family"].startswith("F2")}
    assert len(models) == 9
    assert {t["b"] for t in tests if t["family"].startswith("F3")} == models
    assert all(t["a"] == "bm25" for t in tests if t["family"].startswith("F2"))
    assert all(t["a"] == "bm25_de" for t in tests if t["family"].startswith("F3"))
    f1 = {frozenset((t["a"], t["b"])) for t in tests if t["family"].startswith("F1")}
    assert len(f1) == 36 and all(p <= models for p in f1)


def test_removed_terms_absent():
    pat = re.compile(r"confirmatory|extension|konfirmatorisch|Erweiterung", re.IGNORECASE)
    hits = [f"{f.name}: {m.group(0)}" for f in [*CODE.glob("*.py"), CODE / "README.md"]
            for m in pat.finditer(f.read_text(encoding="utf-8"))]
    assert not hits, hits


def test_p_values_identical_to_previous_design(tests):
    """Tests, die auch im Vorgängerdesign (HEAD vor E33) vorkamen, haben identische p-Werte und CI."""
    rev = "a831c4d"  # letzter Stand mit 103 Tests
    try:
        old = json.loads(subprocess.check_output(
            ["git", "show", f"{rev}:code/output/harness/comparison_germanquad.json"], cwd=ROOT, stderr=subprocess.DEVNULL))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Git-Historie nicht verfügbar")
    ref = {(t["a"], t["b"]): t for t in old["tests"] if t["metric"] == "MRR@10"}
    common = [t for t in tests if (t["a"], t["b"]) in ref]
    assert len(common) == 37
    for t in common:
        o = ref[(t["a"], t["b"])]
        for k in ("mean_a", "mean_b", "diff_b_minus_a", "p_cluster_signflip", "n_nonzero_clusters", "ci95_cluster_bootstrap"):
            assert t[k] == o[k], (t["a"], t["b"], k)
