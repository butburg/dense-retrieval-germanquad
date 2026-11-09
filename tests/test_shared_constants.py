"""Shared experiment constants have a single source (pylib) and keep their serialized form."""
import inspect
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
for sub_dir in (CODE, CODE / "pylib", CODE / "scripts"):
    sys.path.insert(0, str(sub_dir))
import bm25_german  # noqa: E402
import embed_eval  # noqa: E402
import run_bm25  # noqa: E402
import retrieval_metrics  # noqa: E402
import significance  # noqa: E402


def test_values_unchanged():
    assert retrieval_metrics.RELEVANCE_THRESHOLD == 1
    assert list(retrieval_metrics.K_VALUES) == [1, 5, 10]  # JSON "k_values" is written as this list
    assert significance.EXACT_MAX_N == 16


def test_scripts_use_pylib_objects():
    assert run_bm25.K_VALUES is retrieval_metrics.K_VALUES
    assert run_bm25.RELEVANCE_THRESHOLD == retrieval_metrics.RELEVANCE_THRESHOLD
    assert embed_eval.K_VALUES is retrieval_metrics.K_VALUES
    assert embed_eval.RELEVANCE_THRESHOLD == retrieval_metrics.RELEVANCE_THRESHOLD
    assert bm25_german.K_VALUES is retrieval_metrics.K_VALUES


def test_build_gold_default_threshold_is_shared_constant():
    default = inspect.signature(retrieval_metrics.build_gold).parameters["relevance_threshold"].default
    assert default == retrieval_metrics.RELEVANCE_THRESHOLD


def test_no_second_definition_in_scripts():
    for module in (run_bm25, embed_eval, bm25_german):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "\nK_VALUES =" not in source and "\nRELEVANCE_THRESHOLD =" not in source and "\nTHR =" not in source


# --- Model keys, display names and the 512-token subset have one source each ---
EXPECTED_LABEL = {"bm25": "BM25", "e5-large": "multilingual-e5-large", "bge-m3": "bge-m3",
                  "gte-multilingual-base": "gte-multilingual-base", "jina-v3": "jina-embeddings-v3",
                  "openai-3-small": "text-embedding-3-small", "openai-3-large": "text-embedding-3-large",
                  "bm25_de": "BM25-de", "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B",
                  "arctic-embed-l-v2": "snowflake-arctic-embed-l-v2.0", "jina-v2-base-de": "jina-embeddings-v2-base-de"}


def test_label_values_unchanged():
    import make_results
    assert make_results.LABEL == EXPECTED_LABEL
    assert list(make_results.LABEL)[:2] == ["bm25", "bm25_de"]


def test_every_selected_model_is_registered_with_display_name():
    import compare_models
    from embedders import REGISTRY
    for key in compare_models.MODELS:
        assert REGISTRY[key].display_name, key


def test_len512_models_have_one_source():
    import compare_models
    import make_results
    assert compare_models.LEN512_MODELS == ["bge-m3", "gte-multilingual-base", "jina-v3"]
    assert make_results.LEN512_MODELS is compare_models.LEN512_MODELS
    assert set(compare_models.LEN512_MODELS) <= set(compare_models.MODELS)


def test_no_second_literal_in_make_results_and_notebook():
    for path in (CODE / "make_results.py", CODE / "05_hauptexperiment_germanquad.ipynb"):
        source = path.read_text(encoding="utf-8")
        assert '"bge-m3", "gte-multilingual-base", "jina-v3"' not in source.replace('\\"', '"'), path.name
        assert '"e5-large": "' not in source.replace('\\"', '"'), path.name  # no key -> display name literal
    notebook = (CODE / "05_hauptexperiment_germanquad.ipynb").read_text(encoding="utf-8")
    assert "REGISTRY[key].display_name for key in cm.MODELS" in notebook
    assert "LEN512 = cm.LEN512_MODELS" in notebook


# --- Numerical tolerances have one definition per meaning ---
def test_tolerance_values_unchanged():
    import compare_models
    assert significance.TOL == 1e-9
    assert significance.STATISTIC_TOL == 1e-12
    assert significance.MC_FLOOR_REL_TOL == 1e-9
    assert compare_models.CONSISTENCY_TOL == 1e-9


def test_no_tolerance_literals_outside_definitions():
    significance_source = Path(significance.__file__).read_text(encoding="utf-8")
    assert "1e-12" not in significance_source.replace("STATISTIC_TOL = 1e-12", "")
    assert significance_source.count("1e-9") == 2  # TOL and MC_FLOOR_REL_TOL definitions only
    notebook = (CODE / "05_hauptexperiment_germanquad.ipynb").read_text(encoding="utf-8")
    assert "1e-12" not in notebook and "< 1e-9" not in notebook
    assert "> TOL).sum()" in notebook  # same zero-cluster rule as cluster_sign_flip
    assert "< 1e-9" not in (CODE / "make_results.py").read_text(encoding="utf-8")


# --- Rank metrics have one definition (pylib/retrieval_metrics.py) ---


def test_rank_metrics_have_one_definition():
    import compare_models
    import make_results
    assert not hasattr(compare_models, "succ") and not hasattr(compare_models, "rr10")
    assert compare_models.reciprocal_rank_at_k is retrieval_metrics.reciprocal_rank_at_k
    assert make_results.reciprocal_rank_at_k is retrieval_metrics.reciprocal_rank_at_k
    assert make_results.success_at_k is retrieval_metrics.success_at_k
    assert list(compare_models.METRICS) == ["MRR@10", "Success@1", "Success@5", "Success@10"]
    for path in (CODE / "compare_models.py", CODE / "make_results.py", CODE / "05_hauptexperiment_germanquad.ipynb"):
        source = path.read_text(encoding="utf-8")
        assert "def succ(" not in source and "def rr10(" not in source, path.name
        assert "cm.rr10" not in source and "cm.succ" not in source, path.name
        assert "np.where((r > 0)" not in source, path.name  # no inline reciprocal-rank formula


# --- The BM25 ranking over the whole corpus and the best gold rank have one definition (run_bm25.py) ---
class _FakeBm25:
    """Stand-in index returning fixed scores, one per document row."""

    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, query_tokens):
        return self.scores


def test_rank_corpus_orders_by_score_then_doc_id():
    import numpy as np
    doc_ids = ["d3", "d1", "d2", "d4"]
    bm25 = _FakeBm25(np.array([1.0, 2.0, 2.0, 0.5]))
    ranking = run_bm25.rank_corpus(bm25, ["a"], doc_ids)
    assert ranking == [("d1", 2.0), ("d2", 2.0), ("d3", 1.0), ("d4", 0.5)]  # tie d1/d2 -> doc_id ascending
    assert all(type(score) is float for _, score in ranking)


def test_best_gold_rank_first_gold_or_none():
    ranking = [("d1", 2.0), ("d2", 2.0), ("d3", 1.0)]
    assert retrieval_metrics.best_gold_rank(ranking, {"d3", "d2"}) == 2
    assert retrieval_metrics.best_gold_rank(ranking, {"d9"}) is None
    assert retrieval_metrics.best_gold_rank(ranking, set()) is None


def test_best_gold_rank_defined_only_in_retrieval_metrics():
    assert not hasattr(run_bm25, "best_gold_rank") or run_bm25.best_gold_rank is retrieval_metrics.best_gold_rank
    counts = {name: (CODE / name).read_text(encoding="utf-8").count("def best_gold_rank(")
              for name in ("pylib/retrieval_metrics.py", "scripts/run_bm25.py", "bm25_german.py", "demo_query.py")}
    assert counts == {"pylib/retrieval_metrics.py": 1, "scripts/run_bm25.py": 0, "bm25_german.py": 0,
                      "demo_query.py": 0}


def test_bm25_sort_key_defined_only_in_run_bm25():
    counts = {name: (CODE / name).read_text(encoding="utf-8").count("key=lambda it: (-")
              for name in ("scripts/run_bm25.py", "bm25_german.py", "demo_query.py")}
    assert counts == {"scripts/run_bm25.py": 1, "bm25_german.py": 0, "demo_query.py": 0}
