"""p-value formatting in make_results (Holm bounds at the Monte-Carlo floor)."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "pylib"))
from make_results import dep  # noqa: E402

B = 100000


def test_floor_bounds_keep_family_factor():
    assert dep(1 / (B + 1), True) == "≤ 1,0e-05"
    assert dep(6 / (B + 1), True) == "≤ 6,0e-05"
    assert dep(15 / (B + 1), True) == "≤ 1,5e-04"  # not rounded to 0,0001


def test_non_floor_unchanged():
    assert dep(0.0123, False) == "0,0123"
    assert dep(0.00005, False) == "5,0e-05"
    assert dep(1.2e-10, False) == "1,2e-10"
    assert dep(6.3e-6, False) == "6,3e-06"
    assert dep(0.0005, False) == "0,0005"
    assert dep(0.02, True) == "≤ 0,0200"
