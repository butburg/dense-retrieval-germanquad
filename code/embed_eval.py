"""Dataset-agnostic dense-retrieval harness (retrieval only, cosine, k=1,5,10).

Usage: python embed_eval.py --dataset-dir output/germanquad --model e5-large [--out-dir output/harness]
       [--stage {all,encode,score}]
Input: {docs,queries,qrels}.normalized.jsonl in --dataset-dir.
Stages: ``encode`` turns passages and queries into vectors (slow, hours on CPU) and fills the cache; ``score`` reads the
cached vectors, ranks by cosine and writes the metrics (seconds, needs no model and no network); ``all`` (default) does both.
Output: <out-dir>/cache/*.npy (embedding cache), <out-dir>/<run>/encode_<ds>.json (provenance and duration of the
real encoding, written only when something was encoded), <out-dir>/<run>/{metrics_<ds>.json,per_query_<ds>.csv}.
"""
from __future__ import annotations

import argparse, csv, dataclasses, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
from embedders import Embedder, get_config, REGISTRY  # noqa: E402
from local_dataset_io import doc_text, load_jsonl  # noqa: E402
from retrieval_metrics import K_VALUES, build_gold, compute_metrics, first_relevant_ranks, rank_cosine  # noqa: E402

THR = 1


def cache_key_parts(cfg, is_query: bool) -> list:
    """Config fields that determine the embeddings; max_seq_length is always part of the key."""
    return ([cfg.key, cfg.model_name, cfg.query_prefix, cfg.doc_prefix, cfg.normalize, is_query, cfg.max_seq_length]
            + ([cfg.revision, cfg.query_task, cfg.doc_task] if cfg.revision or cfg.query_task else [])
            + ([cfg.query_prompt_name] if cfg.query_prompt_name else []))


def merge_api_meta(metas: list[dict]) -> dict:
    """Combine per-encode OpenAI request metadata (docs + queries) into one record."""
    metas = [m for m in metas if m]
    if not metas:
        return {}
    return {"requested_model": metas[0]["requested_model"],
            "returned_models": sorted({x for m in metas for x in m["returned_models"]}, key=str),
            "first_request_utc": min(m["first_request_utc"] for m in metas),
            "last_request_utc": max(m["last_request_utc"] for m in metas),
            "n_requests": sum(m["n_requests"] for m in metas)}


def cached_encode(emb: Embedder, texts: list[str], is_query: bool, cache: Path, tag: str,
                  require_cache: bool = False) -> tuple[np.ndarray, bool, dict]:
    """Encode with .npy cache keyed by model config (incl. max_seq_length) + text hash.

    Args:
        require_cache: If True, the model is never used; a missing cache file raises instead of encoding.

    Returns:
        Embeddings, cache-hit flag and the OpenAI request metadata (sidecar ``.meta.json``, empty for local models).

    Raises:
        FileNotFoundError: If ``require_cache`` is set and no cache file matches model config and texts.
    """
    h = hashlib.sha256()
    h.update(json.dumps(cache_key_parts(emb.cfg, is_query)).encode())
    for t in texts:
        h.update(t.encode("utf-8")); h.update(b"\0")
    f = cache / f"{emb.cfg.key}__{tag}__{h.hexdigest()[:16]}.npy"
    meta_f = f.with_suffix(".meta.json")
    if f.exists():
        return np.load(f), True, json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if require_cache:
        raise FileNotFoundError(f"no embedding cache for {emb.cfg.key} ({tag}): {f.name} missing in {cache}; run --stage encode first")
    cache.mkdir(parents=True, exist_ok=True)
    emb.api_meta = {}
    arr = emb.encode(texts, is_query)
    meta = dict(emb.api_meta)
    np.save(f, arr)
    if meta:
        meta_f.write_text(json.dumps(meta))
    return arr, False, meta


_NON_INFO = {"dataset", "dataset_dir", "model_key", "run_name", "similarity", "k_values", "relevance_threshold", "doc_text",
             "openai_api"}


def earlier_encode(run_dir: Path, ds: str) -> tuple[dict, dict] | None:
    """Provenance (model info, environment) of the last real encoding of a run, if recorded.

    Reads ``encode_<ds>.json``; for runs that predate it, falls back to ``metrics_<ds>.json`` provided that file
    stems from a real encoding (not from a cache read). Returns None if neither exists.
    """
    enc, met = run_dir / f"encode_{ds}.json", run_dir / f"metrics_{ds}.json"
    if enc.exists():
        d = json.loads(enc.read_text())
        return d["experiment"], d["environment"]
    if met.exists():
        d = json.loads(met.read_text())
        env = d["environment"]
        if not (env.get("cache_hit_docs") and env.get("cache_hit_queries")):
            return {k: v for k, v in d["experiment"].items() if k not in _NON_INFO}, env
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="override the model default (sensitivity run); results go to <model>__len<N>")
    ap.add_argument("--out-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--stage", choices=("all", "encode", "score"), default="all",
                    help="encode: vectors into the cache only; score: rank and metrics from the cache only; all: both")
    a = ap.parse_args()
    ds = a.dataset_dir.resolve().name
    queries, docs, qrels = (load_jsonl(a.dataset_dir / f"{n}.normalized.jsonl") for n in ("queries", "docs", "qrels"))
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    qids = [str(q["query_id"]) for q in queries]
    gold = build_gold(qrels, set(doc_ids), THR)

    cfg = get_config(a.model)
    if a.max_seq_length is not None:
        if a.max_seq_length < 1 or cfg.backend != "sentence-transformers":
            ap.error("--max-seq-length needs a positive value and a sentence-transformers model")
        cfg = dataclasses.replace(cfg, max_seq_length=a.max_seq_length)
    run_name = f"{a.model}__len{a.max_seq_length}" if a.max_seq_length is not None else a.model
    emb = Embedder(cfg)
    cache = a.out_dir / "cache"
    out = a.out_dir / run_name
    cache_only = a.stage == "score"  # score: cache only, the model is never loaded
    t0 = time.time()
    d_emb, d_hit, d_meta = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, cache, f"{ds}_docs", cache_only)
    q_emb, q_hit, q_meta = cached_encode(emb, [str(q["query_text"]).strip() for q in queries], True, cache, f"{ds}_queries", cache_only)
    t_enc = time.time() - t0
    prev = earlier_encode(out, ds)
    out.mkdir(parents=True, exist_ok=True)
    if d_hit and q_hit and prev:  # no model loaded in this run: keep provenance and duration of the real encoding
        info, enc_env = prev[0], {"encode_seconds": prev[1]["encode_seconds"], "encode_source": "earlier encoding (encode_*.json)"}
    else:
        info = {k: v for k, v in emb.info().items() if k not in _NON_INFO}
        enc_env = {"encode_seconds": round(t_enc, 1), "encode_source": "measured in this run"}
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
           "cpu_count": __import__("os").cpu_count(), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    enc_file = out / f"encode_{ds}.json"
    if not (d_hit and q_hit):  # something was really encoded: persist its provenance and duration
        enc_file.write_text(json.dumps({"experiment": info, "environment": {
            **env, "encode_seconds": round(t_enc, 1), "cache_hit_docs": d_hit, "cache_hit_queries": q_hit}}, indent=2), encoding="utf-8")
    elif prev and not enc_file.exists():  # older run without encode record: keep its provenance before metrics are rewritten
        enc_file.write_text(json.dumps({"experiment": prev[0], "environment": {
            **prev[1], "note": "seeded from metrics_*.json of the original encoding"}}, indent=2), encoding="utf-8")
    if a.stage == "encode":
        print(f"encoded {run_name}: encode_seconds={t_enc:.1f} cache_hit_docs={d_hit} cache_hit_queries={q_hit}")
        return
    t1 = time.time()
    idx = rank_cosine(q_emb, d_emb, max(K_VALUES))
    ranked = [[doc_ids[int(i)] for i in row] for row in idx]
    metrics = compute_metrics(qids, ranked, gold, ds)

    # Rank of the best relevant doc over the full ranking (None-equivalent: empty if no gold).
    sims = q_emb @ d_emb.T
    gidx = {i: n for n, i in enumerate(doc_ids)}
    per_q, ties = [], 0  # ties: queries where a non-gold doc has exactly the gold similarity (gold counts as ahead)
    for qi, qid in enumerate(qids):
        gs = [gidx[x] for x in gold.get(qid, ())]
        rank = int((sims[qi] > sims[qi][gs].max()).sum()) + 1 if gs else ""
        ties += bool(gs) and int((sims[qi] == sims[qi][gs].max()).sum()) > len(gs)
        per_q.append((qid, rank, len(gs)))

    score_seconds = round(time.time() - t1, 2)
    with (out / f"per_query_{ds}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query_id", "rank_first_relevant", "num_relevant"]); w.writerows(per_q)
    payload = {"experiment": {"dataset": ds, "dataset_dir": str(a.dataset_dir), "model_key": a.model, "run_name": run_name,
                              **info,
                              **({"openai_api": merge_api_meta([d_meta, q_meta])} if cfg.backend == "openai" else {}), "similarity": "cosine (dot of L2-normalised embeddings)",
                              "k_values": list(K_VALUES), "relevance_threshold": THR,
                              "doc_text": "title\\ntext if both, else either"},
               "metrics": metrics,
               "overview": {"queries": len(queries), "docs": len(docs), "qrels": len(qrels), "embedding_dim": int(d_emb.shape[1]),
                            "queries_without_relevant_docs": sum(q not in gold for q in qids),
                            "queries_with_gold_ties": ties},
               "environment": {**env, **enc_env, "score_seconds": score_seconds, "cache_hit_docs": d_hit, "cache_hit_queries": q_hit}}
    (out / f"metrics_{ds}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(metrics, indent=1), f"encode_seconds={enc_env['encode_seconds']} score_seconds={score_seconds}")


if __name__ == "__main__":
    main()
