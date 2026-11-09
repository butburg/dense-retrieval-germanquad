"""Dataset-agnostic dense-retrieval harness (retrieval only, cosine, k=1,5,10).

Usage: python embed_eval.py --dataset-dir output/germanquad --model e5-large [--out-dir output/harness]
       [--stage {all,encode,score}]
Input: {docs,queries,qrels}.normalized.jsonl in --dataset-dir.
Stages: ``encode`` turns passages and queries into vectors (slow, hours on CPU) and fills the cache; ``score`` reads the
cached vectors, ranks by cosine and writes the metrics (seconds, needs no model and no network); ``all`` (default) does both.
Output: <out-dir>/cache/*.npy (embedding cache), <out-dir>/<run>/encode_<ds>.json (provenance and duration of the
real encoding, written only when something was encoded), <out-dir>/<run>/{metrics_<ds>.json,per_query_<ds>.csv}.

Role in the experiment: core of the dense side. One call = one embedder (registry key from pylib/embedders.py) on one
dataset. Upstream: the normalized dataset files (notebook 02). Downstream: compare_models.py and make_results.py read
<run>/per_query_<ds>.csv and <run>/metrics_<ds>.json; notebook 05 and demo_query.py call this module (cached_encode,
doc_text) as well. ``<run>`` is the model key, or ``<model>__len<N>`` for the max-seq-length sensitivity run.
Ranking is exact (all passages, dot product of L2-normalised vectors = cosine), the metrics use the shared logic in
pylib/retrieval_metrics.py (identical for all retrievers). Running it overwrites metrics_<ds>.json and per_query_<ds>.csv
of the run; the cache is keyed by model config and text hash, so it is reused across runs and never overwritten.
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
from retrieval_metrics import K_VALUES, RELEVANCE_THRESHOLD, build_gold, compute_metrics, first_relevant_ranks, rank_cosine  # noqa: E402


def cache_key_parts(cfg, is_query: bool) -> list:
    """Config fields that determine the embeddings; max_seq_length is always part of the key.

    Args:
        cfg: Embedder configuration (``EmbedderConfig`` from pylib/embedders.py).
        is_query: True for the query side, False for passages; prefixes and prompts differ by side, so the
            cache must not be shared between them.

    Returns:
        List of JSON-serialisable values hashed into the cache file name. Revision and task fields enter only if set;
        changing any listed field therefore leads to a new cache file instead of a stale hit.
    """
    return ([cfg.key, cfg.model_name, cfg.query_prefix, cfg.doc_prefix, cfg.normalize, is_query, cfg.max_seq_length]
            + ([cfg.revision, cfg.query_task, cfg.doc_task] if cfg.revision or cfg.query_task else [])
            + ([cfg.query_prompt_name] if cfg.query_prompt_name else []))


def merge_api_meta(metas: list[dict]) -> dict:
    """Combine per-encode OpenAI request metadata (docs + queries) into one record.

    Args:
        metas: Metadata dicts of the document and query encoding (``requested_model``, ``returned_models``,
            ``first_request_utc``, ``last_request_utc``, ``n_requests``); empty dicts (local models or cache
            without sidecar) are skipped.

    Returns:
        One dict spanning all requests (union of returned models, earliest/latest request time, summed request
        count), or ``{}`` if no metadata exists.
    """
    # Drop empty metadata dicts (local models, cache without sidecar) so only real API request records are merged.
    metas = [m for m in metas if m]
    if not metas:
        return {}
    return {"requested_model": metas[0]["requested_model"],
            # Set comprehension over all records and their returned model strings: union without duplicates, sorted
            # for stable output.
            "returned_models": sorted({x for m in metas for x in m["returned_models"]}, key=str),
            # Generator over the records: earliest first-request time (ISO strings sort chronologically).
            "first_request_utc": min(m["first_request_utc"] for m in metas),
            # Generator over the records: latest last-request time.
            "last_request_utc": max(m["last_request_utc"] for m in metas),
            # Generator over the records: total number of API requests across docs and queries.
            "n_requests": sum(m["n_requests"] for m in metas)}


def cached_encode(emb: Embedder, texts: list[str], is_query: bool, cache: Path, tag: str,
                  require_cache: bool = False) -> tuple[np.ndarray, bool, dict]:
    """Encode with .npy cache keyed by model config (incl. max_seq_length) + text hash.

    Args:
        emb: Embedder wrapping the model configuration.
        texts: Texts to encode in a fixed order (passages in ``doc_ids`` order or queries in file order); the order
            is part of the cache key, the returned rows follow it.
        is_query: True applies the query prefix/prompt, False the passage prefix.
        cache: Cache directory (``output/harness/cache``); created on a real encoding.
        tag: Label in the cache file name, e.g. ``"germanquad_docs"``.
        require_cache: If True, the model is never used; a missing cache file raises instead of encoding.

    Returns:
        Embeddings, cache-hit flag and the OpenAI request metadata (sidecar ``.meta.json``, empty for local models).

    Raises:
        FileNotFoundError: If ``require_cache`` is set and no cache file matches model config and texts.
    """
    # --- Cache lookup: file name = model key + tag + hash of (config fields, all texts in order) ---
    h = hashlib.sha256()
    h.update(json.dumps(cache_key_parts(emb.cfg, is_query)).encode())
    # For every text in order: feed its bytes (plus a separator) into the hash, so the cache key changes
    # whenever any text or the order changes (prevents stale cache hits).
    for t in texts:
        h.update(t.encode("utf-8")); h.update(b"\0")
    f = cache / f"{emb.cfg.key}__{tag}__{h.hexdigest()[:16]}.npy"  # f: cache file for this config and text list
    meta_f = f.with_suffix(".meta.json")  # sidecar with OpenAI request metadata (only for API models)
    if f.exists():
        return np.load(f), True, json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if require_cache:
        raise FileNotFoundError(f"no embedding cache for {emb.cfg.key} ({tag}): {f.name} missing in {cache}; run --stage encode first")
    # --- Cache miss: encode for real and store ---
    cache.mkdir(parents=True, exist_ok=True)
    emb.api_meta = {}
    arr = emb.encode(texts, is_query)
    meta = dict(emb.api_meta)
    np.save(f, arr)
    if meta:
        meta_f.write_text(json.dumps(meta))
    return arr, False, meta


# Keys of ``experiment`` that describe a particular scoring run (not the encoding); they are removed when the
# provenance of an earlier encoding is carried over.
_NON_INFO = {"dataset", "dataset_dir", "model_key", "run_name", "similarity", "k_values", "relevance_threshold", "doc_text",
             "openai_api"}


def earlier_encode(run_dir: Path, ds: str) -> tuple[dict, dict] | None:
    """Provenance (model info, environment) of the last real encoding of a run, if recorded.

    Reads ``encode_<ds>.json``; for runs that predate it, falls back to ``metrics_<ds>.json`` provided that file
    stems from a real encoding (not from a cache read). Returns None if neither exists.

    Args:
        run_dir: Result directory of the run, e.g. ``output/harness/e5-large``.
        ds: Dataset name, e.g. ``"germanquad"``.

    Returns:
        Tuple ``(model_info, environment)`` of the earlier encoding, or ``None``.
    """
    enc, met = run_dir / f"encode_{ds}.json", run_dir / f"metrics_{ds}.json"
    if enc.exists():
        d = json.loads(enc.read_text())
        return d["experiment"], d["environment"]
    if met.exists():
        d = json.loads(met.read_text())
        env = d["environment"]
        if not (env.get("cache_hit_docs") and env.get("cache_hit_queries")):
            # Dict comprehension: copy the earlier experiment record without the run-specific keys listed in
            # _NON_INFO.
            return {k: v for k, v in d["experiment"].items() if k not in _NON_INFO}, env
    return None


def main() -> None:
    """Encode (or read from cache), rank by cosine, compute metrics and write the run's result files.

    Steps depend on ``--stage``: ``encode`` stops after the cache is filled and the provenance written;
    ``score`` never loads the model and fails if the cache is incomplete; ``all`` does both.
    Existing ``metrics_<ds>.json`` and ``per_query_<ds>.csv`` of the run are overwritten.
    """
    # --- Parse arguments ---
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="override the model default (sensitivity run); results go to <model>__len<N>")
    ap.add_argument("--out-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--stage", choices=("all", "encode", "score"), default="all",
                    help="encode: vectors into the cache only; score: rank and metrics from the cache only; all: both")
    a = ap.parse_args()
    # --- Load data: ds = dataset name from the folder name; gold = query_id -> relevant doc_ids ---
    ds = a.dataset_dir.resolve().name
    # Generator over the three roles: load each normalized JSONL file; unpacking gives the three record lists.
    queries, docs, qrels = (load_jsonl(a.dataset_dir / f"{n}.normalized.jsonl") for n in ("queries", "docs", "qrels"))
    # Dict comprehension: doc ID -> doc record (lookup table; a repeated ID would keep the last record).
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)  # doc_ids: passage order; embedding row i belongs to doc_ids[i]
    # List comprehension: query IDs as strings in file order.
    qids = [str(q["query_id"]) for q in queries]  # qids: query order; query embedding row i belongs to qids[i]
    gold = build_gold(qrels, set(doc_ids), RELEVANCE_THRESHOLD)

    # --- Model configuration (optional max_seq_length override = sensitivity run, separate result folder) ---
    cfg = get_config(a.model)
    if a.max_seq_length is not None:
        if a.max_seq_length < 1 or cfg.backend != "sentence-transformers":
            ap.error("--max-seq-length needs a positive value and a sentence-transformers model")
        cfg = dataclasses.replace(cfg, max_seq_length=a.max_seq_length)
    run_name = f"{a.model}__len{a.max_seq_length}" if a.max_seq_length is not None else a.model
    emb = Embedder(cfg)  # the model is loaded lazily on the first real encoding
    cache = a.out_dir / "cache"
    out = a.out_dir / run_name
    cache_only = a.stage == "score"  # score: cache only, the model is never loaded
    # --- Encode passages and queries (d_*: passages, q_*: queries; *_hit: read from cache; *_meta: API metadata) ---
    t0 = time.time()
    # Inner list comprehension: indexed text (title + text) of every passage in doc_ids order, so embedding row i =
    # doc_ids[i].
    d_emb, d_hit, d_meta = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, cache, f"{ds}_docs", cache_only)
    # Inner list comprehension: stripped query text of every query in file order, so embedding row i = qids[i].
    q_emb, q_hit, q_meta = cached_encode(emb, [str(q["query_text"]).strip() for q in queries], True, cache, f"{ds}_queries", cache_only)
    t_enc = time.time() - t0
    # --- Provenance: record how and how long the embeddings were really produced ---
    prev = earlier_encode(out, ds)  # prev: provenance of an earlier real encoding of this run, if any
    out.mkdir(parents=True, exist_ok=True)
    if d_hit and q_hit and prev:  # no model loaded in this run: keep provenance and duration of the real encoding
        info, enc_env = prev[0], {"encode_seconds": prev[1]["encode_seconds"], "encode_source": "earlier encoding (encode_*.json)"}
    else:
        # Dict comprehension: keep the model provenance (name, revision, versions) without run-specific keys.
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
    # --- Rank and score: exact cosine ranking of all passages, shared metric code ---
    t1 = time.time()
    # idx: top-max(k) passage indices per query (best first); ranked: the same as doc IDs. The summary metrics
    # (Recall/Success/MRR@k) use only these top lists; the full-ranking gold rank below feeds the significance tests.
    idx = rank_cosine(q_emb, d_emb, max(K_VALUES))
    # Nested list comprehension: for each query row of idx (top-k passage indices, best first)
    # map every index to its doc ID, giving the ranked doc-ID lists that compute_metrics expects.
    ranked = [[doc_ids[int(i)] for i in row] for row in idx]
    metrics = compute_metrics(qids, ranked, gold, ds)

    # Rank of the best relevant doc over the full ranking (None-equivalent: empty if no gold).
    # Tie rule: rank = 1 + number of passages with a strictly higher similarity than the best gold passage, so gold
    # passages win ties; `ties` counts queries where a non-gold passage has exactly the same similarity.
    # Matrix product (n_queries, dim) @ (dim, n_docs) = all query-passage cosine similarities;
    # sims[i, j] is the similarity of query i and passage j.
    sims = q_emb @ d_emb.T  # sims: (n_queries, n_docs) cosine similarities
    # Dict comprehension: doc ID -> column index in sims/d_emb (translates a query's gold doc IDs into matrix
    # columns).
    gidx = {i: n for n, i in enumerate(doc_ids)}  # gidx: doc_id -> row in d_emb
    per_q, ties = [], 0  # per_q: (query_id, rank, number of gold docs) rows for the CSV; ties: queries where a non-gold doc has exactly the gold similarity (gold counts as ahead)
    # For every query (qi = row in sims): rank of its best gold passage within the full ranking of all passages
    # (the input of the significance tests), plus a count of ties with non-gold passages.
    for qi, qid in enumerate(qids):
        # List comprehension: row indices of this query's gold passages (empty if the query has no gold passage).
        gs = [gidx[x] for x in gold.get(qid, ())]  # gs: row indices of this query's gold passages
        # sims[qi][gs] picks the similarities of the gold passages (list indexing), .max() the best one.
        # sims[qi] > best gives a boolean mask over all passages; .sum() counts the strictly better ones,
        # e.g. sims [0.9, 0.8, 0.7], best gold 0.8 -> mask [True, False, False] -> rank 1 + 1 = 2 (empty if no gold).
        rank = int((sims[qi] > sims[qi][gs].max()).sum()) + 1 if gs else ""
        # Counts passages with exactly the best gold similarity (== mask, .sum()); more of them than gold passages
        # means a non-gold passage ties with the gold one.
        ties += bool(gs) and int((sims[qi] == sims[qi][gs].max()).sum()) > len(gs)
        per_q.append((qid, rank, len(gs)))

    # --- Write per-query ranks and the metrics JSON (metrics + overview + provenance + environment) ---
    score_seconds = round(time.time() - t1, 2)
    with (out / f"per_query_{ds}.csv").open("w", newline="", encoding="utf-8") as f:
        # writerows writes one CSV line per (query_id, rank, num_relevant) tuple in per_q.
        w = csv.writer(f); w.writerow(["query_id", "rank_first_relevant", "num_relevant"]); w.writerows(per_q)
    payload = {"experiment": {"dataset": ds, "dataset_dir": str(a.dataset_dir), "model_key": a.model, "run_name": run_name,
                              **info,
                              **({"openai_api": merge_api_meta([d_meta, q_meta])} if cfg.backend == "openai" else {}), "similarity": "cosine (dot of L2-normalised embeddings)",
                              "k_values": list(K_VALUES), "relevance_threshold": RELEVANCE_THRESHOLD,
                              "doc_text": "title\\ntext if both, else either"},
               "metrics": metrics,
               "overview": {"queries": len(queries), "docs": len(docs), "qrels": len(qrels), "embedding_dim": int(d_emb.shape[1]),
                            # Generator over all query IDs: counts those without any gold document (booleans sum as
                            # 0/1).
                            "queries_without_relevant_docs": sum(q not in gold for q in qids),
                            "queries_with_gold_ties": ties},
               "environment": {**env, **enc_env, "score_seconds": score_seconds, "cache_hit_docs": d_hit, "cache_hit_queries": q_hit}}
    (out / f"metrics_{ds}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(metrics, indent=1), f"encode_seconds={enc_env['encode_seconds']} score_seconds={score_seconds}")


if __name__ == "__main__":
    main()
