"""Registry checks for embedders (no model download)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pylib"))
from embedders import REGISTRY, get_config  # noqa: E402
import pytest  # noqa: E402


def test_all_protocol_models_registered():
    assert {"e5-large", "bge-m3", "gte-multilingual-base", "jina-v3",
            "openai-3-small", "openai-3-large"} <= set(REGISTRY)


def test_e5_unchanged():
    c = get_config("e5-large")
    assert (c.query_prefix, c.doc_prefix, c.revision, c.query_task, c.trust_remote_code) == ("query: ", "passage: ", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3", None, False)


def test_new_models_follow_model_cards():
    assert get_config("bge-m3").query_prefix == get_config("bge-m3").doc_prefix == ""
    g = get_config("gte-multilingual-base")
    assert g.trust_remote_code and g.query_prefix == g.doc_prefix == ""
    j = get_config("jina-v3")
    assert j.trust_remote_code and (j.query_task, j.doc_task) == ("retrieval.query", "retrieval.passage")
    for k in ("bge-m3", "gte-multilingual-base", "jina-v3"):
        assert len(get_config(k).revision) == 40


def test_unknown_key():
    with pytest.raises(KeyError):
        get_config("nope")


def test_extension_models_follow_model_cards():
    q = get_config("qwen3-embedding-0.6b")
    assert (q.query_prompt_name, q.query_prefix, q.doc_prefix, q.max_seq_length) == ("query", "", "", 8192)
    a = get_config("arctic-embed-l-v2")
    assert (a.query_prefix, a.doc_prefix) == ("query: ", "")
    j = get_config("jina-v2-base-de")
    assert j.trust_remote_code and j.code_repo and len(j.code_revision) == 40
    for k in ("qwen3-embedding-0.6b", "arctic-embed-l-v2", "jina-v2-base-de"):
        assert len(get_config(k).revision) == 40


def test_prompt_name_changes_cache_key():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from embed_eval import cache_key_parts
    q = get_config("qwen3-embedding-0.6b")
    assert "query" in cache_key_parts(q, True)
    assert "query" not in cache_key_parts(get_config("bge-m3"), True)
