"""Cache key and --max-seq-length CLI of embed_eval (no model download)."""
import dataclasses, json, sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "pylib"))
import embed_eval  # noqa: E402
from embedders import REGISTRY, get_config  # noqa: E402


def test_cache_key_contains_max_seq_length():
    c = get_config("bge-m3")
    assert cache_parts(c) != cache_parts(dataclasses.replace(c, max_seq_length=512))
    assert 512 in cache_parts(dataclasses.replace(c, max_seq_length=512))


def cache_parts(c):
    return embed_eval.cache_key_parts(c, False)


def test_registry_pins():
    assert get_config("e5-large").revision.startswith("3d7cfbda") and len(get_config("e5-large").revision) == 40
    assert get_config("jina-v3").batch_size == 4
    for k in ("gte-multilingual-base", "jina-v3"):
        assert len(get_config(k).code_revision) == 40 and get_config(k).code_repo


class FakeEmb:
    made = []

    def __init__(self, cfg):
        self.cfg, self.api_meta = cfg, {}
        FakeEmb.made.append(cfg)

    def info(self):
        return {"max_seq_length_config": self.cfg.max_seq_length}

    def encode(self, texts, is_query):
        rng = np.random.default_rng(len(texts) + int(is_query))
        x = rng.normal(size=(len(texts), 4)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_cli_override_suffix_and_cache(tmp_path, monkeypatch):
    ds = tmp_path / "toy"; ds.mkdir()
    (ds / "docs.normalized.jsonl").write_text("\n".join(json.dumps({"doc_id": f"d{i}", "text": f"t{i}"}) for i in range(12)))
    (ds / "queries.normalized.jsonl").write_text("\n".join(json.dumps({"query_id": f"q{i}", "query_text": "x"}) for i in range(2)))
    (ds / "qrels.normalized.jsonl").write_text("\n".join(json.dumps({"query_id": f"q{i}", "doc_id": f"d{i}", "relevance": 1}) for i in range(2)))
    monkeypatch.setattr(embed_eval, "Embedder", FakeEmb)
    out = tmp_path / "out"
    base = ["embed_eval.py", "--dataset-dir", str(ds), "--model", "bge-m3", "--out-dir", str(out)]
    for extra in ([], ["--max-seq-length", "512"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        embed_eval.main()
    assert (out / "bge-m3" / "metrics_toy.json").exists() and (out / "bge-m3__len512" / "metrics_toy.json").exists()
    assert FakeEmb.made[0].max_seq_length is None and FakeEmb.made[1].max_seq_length == 512
    assert len(list((out / "cache").glob("*.npy"))) == 4  # docs+queries per setting, no key collision
