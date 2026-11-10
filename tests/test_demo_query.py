"""demo_query.py: Gold-Ränge müssen mit den gespeicherten per-query-CSVs übereinstimmen; Spaltenansicht mit drei Spalten."""
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))

DS = CODE / "output" / "germanquad"
BM25_CSV = CODE / "output" / "bm25_v2" / "per_query_ranks.csv"
BM25_DE_CSV = CODE / "output" / "bm25_de" / "per_query_ranks.csv"
HARNESS = CODE / "output" / "harness"
CACHE = HARNESS / "cache"
N = 25
pytest.importorskip("rank_bm25")
from retrieval_metrics import best_gold_rank  # noqa: E402  (single definition of the best gold rank)
DATA_MISSING = not (DS / "docs.normalized.jsonl").exists() or not BM25_CSV.exists() or not BM25_DE_CSV.exists()


@pytest.fixture(autouse=True)
def _skip_without_data(request):
    """Skip tests that need the normalized data and BM25 CSVs; the offline-decision tests (``test_offline_*``) do not."""
    if DATA_MISSING and not request.node.name.startswith("test_offline_"):
        pytest.skip("normalisierte Daten oder BM25-CSVs fehlen")


def _sample_qids(ids: list[str]) -> list[str]:
    rest = [q for q in ids if q != "q40369"]
    return ["q40369", "q40321"] + rest[:: max(1, len(rest) // (N - 1))][: N - 2]


@pytest.fixture(scope="module")
def data():
    import demo_query as dq
    from embed_eval import doc_text
    from retrieval_metrics import build_gold
    docs, queries, qrels = dq.load_dataset(DS)
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    return dq, docs_by_id, doc_ids, {str(q["query_id"]): str(q["query_text"]) for q in queries}, \
        build_gold(qrels, set(doc_ids), 1), doc_text


@pytest.mark.parametrize("variant, csv_path", [("BM25", BM25_CSV), ("BM25-de", BM25_DE_CSV)])
def test_lexical_gold_rank_and_top10_match_csv(data, variant, csv_path):
    dq, docs_by_id, doc_ids, qtext, gold, _ = data
    ref = pd.read_csv(csv_path, dtype={"query_id": str}).query("dataset == 'germanquad'").set_index("query_id")
    bm25, tokenize = dq.build_bm25(doc_ids, docs_by_id, variant)
    qids = _sample_qids(list(qtext))
    assert len(qids) >= 20
    for qid in qids:
        ranking = dq.bm25_ranking(bm25, tokenize, doc_ids, qtext[qid])
        assert best_gold_rank(ranking, gold[qid]) == ref.loc[qid, "best_gold_rank_full"], (variant, qid)
        assert [d for d, _ in ranking[:10]] == ref.loc[qid, "top_doc_ids"].split(" | "), (variant, qid)


def test_unknown_lexical_variant_is_rejected(data):
    dq, docs_by_id, doc_ids, *_ = data
    with pytest.raises(ValueError):
        dq.build_bm25(doc_ids, docs_by_id, "BM25-xx")


def test_bm25_index_has_single_definition():
    """Index construction (parameters) lives in run_bm25.build_index; demo and bm25_german only call it."""
    counts = {name: (CODE / name).read_text(encoding="utf-8").count("BM25Okapi(")
              for name in ("scripts/run_bm25.py", "bm25_german.py", "demo_query.py")}
    assert counts == {"scripts/run_bm25.py": 1, "bm25_german.py": 0, "demo_query.py": 0}


@pytest.mark.parametrize("model", ["e5-large", "bge-m3"])
def test_dense_gold_rank_matches_csv(data, model):
    dq, docs_by_id, doc_ids, qtext, gold, doc_text = data
    if not list(CACHE.glob(f"{model}__germanquad_docs__*.npy")):
        pytest.skip(f"Dokument-Embedding-Cache für {model} fehlt (python code/demo_query.py --warm-cache --model {model})")
    from embed_eval import cached_encode
    from embedders import Embedder, get_config
    emb = Embedder(get_config(model))
    d_emb, hit, _ = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, CACHE, "germanquad_docs")
    assert hit
    ref = pd.read_csv(HARNESS / model / "per_query_germanquad.csv", dtype={"query_id": str}).set_index("query_id")
    qids = _sample_qids(list(qtext))
    try:
        q_emb = emb.encode([qtext[q] for q in qids], True)
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"Modell nicht ladbar: {type(ex).__name__}")
    for qid, qv in zip(qids, q_emb):
        ranking = dq.dense_ranking(qv[None, :], d_emb, doc_ids)
        assert best_gold_rank(ranking, gold[qid]) == ref.loc[qid, "rank_first_relevant"], (model, qid)


# --- Column view (no model needed) ---
def _toy_view(dq, gold: set[str], width: int, capsys) -> list[str]:
    docs_by_id = {f"c{i}": {"doc_id": f"c{i}", "title": "", "text": f"Titel{i}\nText {i} " + "wort " * 60} for i in (1, 2, 3, 4)}
    rankings = [("BM25", [("c1", 3.0), ("c2", 2.0), ("c3", 1.0), ("c4", 0.5)]),
                ("BM25-de", [("c2", 4.0), ("c1", 2.0), ("c3", 1.0), ("c4", 0.5)]),
                ("bge-m3", [("c4", 0.9), ("c3", 0.8), ("c2", 0.7), ("c1", 0.1)])]
    dq.show("Testfrage?", rankings, docs_by_id, gold, 3, width)
    return capsys.readouterr().out.splitlines()


def test_show_has_three_columns_with_gold_ranks(data, capsys):
    dq = data[0]
    lines = _toy_view(dq, {"c1"}, 170, capsys)
    heading = next(line for line in lines if line.startswith("BM25"))
    assert heading.split(" | ") == ["BM25".ljust(len(heading.split(" | ")[0])),
                                    "BM25-de".ljust(len(heading.split(" | ")[1])), "bge-m3"]
    assert max(len(line) for line in lines) <= 170
    rank_line = next(line for line in lines if line.startswith("Gold-Rang"))
    assert [part.strip() for part in rank_line.split(" | ")] == [
        "Gold-Rang: 1 (volle Rangliste)", "Gold-Rang: 2 (volle Rangliste)", "Gold-Rang: 4 (volle Rangliste)"]
    marks = sum(line.count("[GOLD]") for line in lines if "Score" in line)  # marks on result lines only
    assert marks == 2  # c1 is in the BM25 and BM25-de top 3, not in bge-m3's


def test_show_free_text_has_no_gold_marking(data, capsys):
    dq = data[0]
    lines = _toy_view(dq, set(), 170, capsys)
    assert not any("[GOLD]" in line or "Gold-Rang" in line or "Gold-Passage" in line for line in lines)
    assert len(next(line for line in lines if line.startswith("BM25")).split(" | ")) == 3


def test_show_respects_width_option(data, capsys):
    dq = data[0]
    assert max(len(line) for line in _toy_view(dq, {"c1"}, 130, capsys)) <= 130


def test_snippet_lines_are_short_and_indented(data):
    dq, docs_by_id, *_ = data
    doc = next(iter(docs_by_id.values()))
    lines = dq.snippet_lines(doc, 50)
    assert 1 <= len(lines) <= dq.SNIPPET_LINES and all(len(line) <= 50 for line in lines)
    assert all(line.startswith("   ") for line in lines)


# --- Target passage (--target-id) ---
def test_show_target_rank_footer_and_marks(data, capsys):
    dq = data[0]
    docs_by_id = {f"c{i}": {"doc_id": f"c{i}", "title": f"Titel{i}", "text": "wort " * 60} for i in (1, 2, 3, 4)}
    rankings = [("BM25", [("c1", 3.0), ("c2", 2.0), ("c3", 1.0), ("c4", 0.5)]),
                ("BM25-de", [("c2", 4.0), ("c1", 2.0), ("c3", 1.0), ("c4", 0.5)]),
                ("bge-m3", [("c4", 0.9), ("c3", 0.8), ("c2", 0.7), ("c1", 0.1)])]
    dq.show("Testfrage?", rankings, docs_by_id, set(), 3, 170, "c3")
    lines = capsys.readouterr().out.splitlines()
    rank_line = next(line for line in lines if line.startswith("Ziel-Rang"))
    assert [part.strip() for part in rank_line.split(" | ")] == [
        "Ziel-Rang: 3 (volle Rangliste)", "Ziel-Rang: 3 (volle Rangliste)", "Ziel-Rang: 2 (volle Rangliste)"]
    assert "Ziel-Passage: c3  Titel3" in lines
    assert sum(line.count("[ZIEL]") for line in lines if "Score" in line) == 3  # c3 in every top 3
    assert not any("Gold-Rang" in line for line in lines)  # free text: no gold footer


def test_show_without_target_has_no_target_output(data, capsys):
    dq = data[0]
    lines = _toy_view(dq, {"c1"}, 170, capsys)
    assert not any("Ziel" in line for line in lines)


def test_show_target_that_is_gold_gets_both_marks(data, capsys):
    dq = data[0]
    docs_by_id = {f"c{i}": {"doc_id": f"c{i}", "title": f"Titel{i}", "text": "wort " * 60} for i in (1, 2)}
    dq.show("Testfrage?", [("BM25", [("c1", 2.0), ("c2", 1.0)])] * 3, docs_by_id, {"c1"}, 1, 170, "c1")
    first_result = next(line for line in capsys.readouterr().out.splitlines() if "Score" in line)
    assert "[GOLD]  [ZIEL]" in first_result


def test_target_rank_equals_gold_rank_for_gold_target(data):
    """Target = gold of a dataset query gives the gold rank, via the same rank function in column()."""
    dq, docs_by_id, doc_ids, qtext, gold, _ = data
    qid = "q40321"
    target = sorted(gold[qid])[0]
    bm25, tokenize = dq.build_bm25(doc_ids, docs_by_id, "BM25")
    ranking = dq.bm25_ranking(bm25, tokenize, doc_ids, qtext[qid])
    lines = dq.column("BM25", ranking, docs_by_id, gold[qid], 3, 120, target)
    gold_line = next(line for line in lines if line.startswith("Gold-Rang")).strip()
    target_line = next(line for line in lines if line.startswith("Ziel-Rang")).strip()
    assert gold_line.split(": ")[1] == target_line.split(": ")[1]
    assert f"Gold-Rang: {best_gold_rank(ranking, gold[qid])} " in gold_line


def test_unknown_target_id_is_rejected_with_clear_message():
    import subprocess
    result = subprocess.run([sys.executable, str(CODE / "demo_query.py"), "--query", "Testfrage?",
                             "--target-id", "c_gibt_es_nicht"], capture_output=True, text=True, timeout=300)
    assert result.returncode == 1
    assert "Ziel-ID 'c_gibt_es_nicht' nicht in" in result.stderr


# --- Offline decision (cache check without network) ---
@pytest.fixture
def no_network(monkeypatch):
    """Any socket connection fails the test: the decision must come from the cache folders alone."""
    import socket

    def refuse(self, address):
        raise AssertionError(f"unexpected network access to {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def _fake_snapshot(hub_cache: Path, repo_id: str, revision: str, files: tuple[str, ...]) -> Path:
    """Create ``models--<org>--<name>/snapshots/<revision>`` with empty files (as the hub cache lays them out)."""
    snapshot = hub_cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("x", encoding="utf-8")
    return snapshot


def test_offline_complete_cache_switches_both_variables_on(tmp_path, no_network):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("bge-m3")
    _fake_snapshot(tmp_path, cfg.model_name, cfg.revision, ("config.json", "model.safetensors", "tokenizer.json"))
    environ: dict[str, str] = {}
    assert dq.choose_offline_mode(cfg, environ, tmp_path) is True
    assert environ == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_offline_pytorch_bin_counts_as_weights(tmp_path):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("bge-m3")
    _fake_snapshot(tmp_path, cfg.model_name, cfg.revision, ("config.json", "pytorch_model.bin"))
    assert dq.is_model_cached(cfg, tmp_path)


def test_offline_missing_or_incomplete_cache_stays_online_with_note(tmp_path, capsys, no_network):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("bge-m3")
    environ: dict[str, str] = {}
    assert dq.choose_offline_mode(cfg, environ, tmp_path) is False  # empty cache
    # Tokenizer-only snapshot (no config, no weights) and a config without weights are not usable offline.
    _fake_snapshot(tmp_path, cfg.model_name, cfg.revision, ("tokenizer.json", "config.json"))
    assert dq.choose_offline_mode(cfg, environ, tmp_path) is False
    assert environ == {}
    assert capsys.readouterr().err.count("Hinweis: BAAI/bge-m3") == 2


def test_offline_other_revision_does_not_count(tmp_path):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("bge-m3")
    _fake_snapshot(tmp_path, cfg.model_name, "0" * 40, ("config.json", "model.safetensors"))
    assert not dq.is_model_cached(cfg, tmp_path)


def test_offline_dangling_symlink_is_not_a_cached_file(tmp_path):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("bge-m3")
    snapshot = _fake_snapshot(tmp_path, cfg.model_name, cfg.revision, ("config.json",))
    (snapshot / "model.safetensors").symlink_to(tmp_path / "blobs" / "missing")  # blob was never downloaded
    assert not dq.is_model_cached(cfg, tmp_path)


@pytest.mark.parametrize("explicit", ["0", "1"])
def test_offline_explicit_environment_value_wins(tmp_path, explicit):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("bge-m3")
    _fake_snapshot(tmp_path, cfg.model_name, cfg.revision, ("config.json", "model.safetensors"))
    environ = {"HF_HUB_OFFLINE": explicit}
    assert dq.choose_offline_mode(cfg, environ, tmp_path) is (explicit == "1")
    assert environ == {"HF_HUB_OFFLINE": explicit}  # TRANSFORMERS_OFFLINE is not added next to an explicit choice


def test_offline_remote_code_model_needs_code_repo_in_cache(tmp_path):
    import demo_query as dq
    from embedders import get_config
    cfg = get_config("jina-v3")
    _fake_snapshot(tmp_path, cfg.model_name, cfg.revision, ("config.json", "model.safetensors"))
    assert not dq.is_model_cached(cfg, tmp_path)  # remote code repo missing
    _fake_snapshot(tmp_path, cfg.code_repo, cfg.code_revision, ("README.md",))
    assert not dq.is_model_cached(cfg, tmp_path)  # code repo present but without Python files
    (tmp_path / ("models--" + cfg.code_repo.replace("/", "--")) / "snapshots" / cfg.code_revision / "modeling.py").write_text("x")
    assert dq.is_model_cached(cfg, tmp_path)


def test_offline_hub_cache_dir_follows_environment(tmp_path):
    import demo_query as dq
    assert dq.hub_cache_dir({"HF_HUB_CACHE": "/a", "HF_HOME": "/b"}) == Path("/a")
    assert dq.hub_cache_dir({"HF_HOME": "/b"}) == Path("/b/hub")
    assert dq.hub_cache_dir({"XDG_CACHE_HOME": "/c"}) == Path("/c/huggingface/hub")
    assert dq.hub_cache_dir({}) == Path.home() / ".cache" / "huggingface" / "hub"
