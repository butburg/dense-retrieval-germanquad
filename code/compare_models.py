"""Paired comparison of all eleven retrievers on GermanQuAD per docs/evaluation_protocol.md (one run, 103 tests).

Usage: python compare_models.py [--harness-dir output/harness] [--out output/harness/comparison_germanquad.json]
Input: per_query_germanquad.csv of the nine embedders in --harness-dir (rank of first relevant doc over the full
ranking), BM25 (output/bm25_v2/per_query_ranks.csv) and BM25-de (output/bm25_de/per_query_ranks.csv),
qrels/queries from output/germanquad (cluster = first sorted gold doc).
Families (Holm within each family, 22 families):
  confirmatory: six embedders vs BM25 on MRR@10 (planned family size 6);
  explorative: pairs of the six on MRR@10; six vs BM25 per Success@k; BM25-de vs BM25 (4 metrics);
  six vs BM25-de per metric; three extension models vs BM25, BM25-de and bge-m3 per metric.
Output: JSON with p (raw, Holm), n_nonzero and cluster-bootstrap CI (B=10000), all seeded with 42 per test.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent


def rel(p: Path) -> str:
    """Repo-relative POSIX path (falls back to the given path outside the repo)."""
    try:
        return Path(p).resolve().relative_to(CODE.parent).as_posix()
    except ValueError:
        return str(p)

sys.path.insert(0, str(CODE / "pylib"))
from local_dataset_io import load_jsonl  # noqa: E402
from retrieval_metrics import build_gold  # noqa: E402
from significance import B_BOOT, B_PERM, SEED, cluster_bootstrap_ci, cluster_sign_flip, holm, make_clusters  # noqa: E402

EMBEDDERS = ["e5-large", "bge-m3", "gte-multilingual-base", "jina-v3", "openai-3-small", "openai-3-large"]  # confirmatory
EXTENSION = ["qwen3-embedding-0.6b", "arctic-embed-l-v2", "jina-v2-base-de"]  # added later, explorative only
THRESHOLD = 20  # n_nonzero below this: "nicht aussagekräftig" (protocol, Detailfestlegungen)
DS = "germanquad"
BM25_CSV = CODE / "output" / "bm25_v2" / "per_query_ranks.csv"
BM25_DE_CSV = CODE / "output" / "bm25_de" / "per_query_ranks.csv"


def succ(r: np.ndarray, k: int) -> np.ndarray:
    """Success@k indicator from first-relevant ranks (0 = no relevant doc)."""
    return ((r > 0) & (r <= k)).astype(float)


def rr10(r: np.ndarray) -> np.ndarray:
    """Reciprocal rank at cutoff 10 from first-relevant ranks (0 = no relevant doc)."""
    return np.where((r > 0) & (r <= 10), 1.0 / np.maximum(r, 1), 0.0)


METRICS = {"MRR@10": rr10, "Success@1": lambda r: succ(r, 1), "Success@5": lambda r: succ(r, 5),
           "Success@10": lambda r: succ(r, 10)}


def load_ranks(path: Path, qids: list[str]) -> np.ndarray:
    """Read first-relevant ranks aligned to ``qids`` (harness or bm25_v2 per-query format).

    Raises:
        ValueError: If the query ids do not match ``qids`` (after filtering bm25_v2 to the dataset).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "dataset" in df:
        df = df[df["dataset"] == DS]
    col = "rank_first_relevant" if "rank_first_relevant" in df else "best_gold_rank_full"
    if df["query_id"].tolist() != qids:
        raise ValueError(f"{path}: query ids/order differ from queries.normalized.jsonl")
    return np.array([int(float(x)) if x != "" else 0 for x in df[col]], dtype=int)


def apply_holm(rows: list[dict], size_of=lambda family: None) -> None:
    """Add Holm-adjusted ``p_holm`` to ``rows`` in place, corrected within each ``family`` label.

    Args:
        rows: Result rows of ``paired_test`` with ``family`` and ``p_cluster_signflip``.
        size_of: Maps a family label to a planned family size (``None`` = number of rows present).
    """
    for fam in {r["family"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        for i, h in zip(idx, holm([rows[i]["p_cluster_signflip"] for i in idx], size_of(fam))):
            rows[i]["p_holm"] = h


def load_queries_clusters(dataset_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Read query ids and cluster labels (first sorted gold passage per query) of the normalized dataset.

    Args:
        dataset_dir: Directory with the ``*.normalized.jsonl`` files (default ``output/germanquad``).

    Returns:
        Query ids in file order and the cluster label per query.

    Raises:
        AssertionError: If a query has no gold passage.
    """
    ds = dataset_dir or CODE / "output" / DS
    queries, docs, qrels = (load_jsonl(ds / f"{x}.normalized.jsonl") for x in ("queries", "docs", "qrels"))
    qids = [str(q["query_id"]) for q in queries]
    gold = build_gold(qrels, {str(d["doc_id"]) for d in docs}, 1)
    assert all(q in gold for q in qids), "queries without gold are not supported"
    return qids, [sorted(gold[q])[0] for q in qids]


def make_meta(sources: dict[str, Path | str], note: str, **extra) -> dict:
    """Metadata block of a comparison JSON.

    Args:
        sources: Retriever key to input file; stored repo-relative.
        note: Free-text note stored at the end of the block.
        **extra: Additional entries placed after the resample counts (e.g. ``status``).
    """
    return {"dataset": DS, "seed": SEED, "B_permutation": B_PERM, "B_bootstrap": B_BOOT, **extra,
            "sources": {k: rel(p) for k, p in sources.items()}, "numpy": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "note": note}


def paired_test(ranks: dict[str, np.ndarray], clusters: list[str], a: str, b: str, metric: str, family: str = "",
              b_perm: int = B_PERM, b_boot: int = B_BOOT) -> dict:
    """Paired test of retriever ``b`` against ``a`` on one metric (one result row, no Holm yet).

    Args:
        ranks: First-relevant ranks per retriever key, aligned to the queries (0 = no relevant doc).
        clusters: Cluster label (gold passage) per query.
        a: Reference retriever key.
        b: Compared retriever key; positive ``diff_b_minus_a`` means ``b`` is better.
        metric: Key of ``METRICS``.
        family: Label stored in the row; Holm is applied by the caller per family.
        b_perm: Sign-flip resamples (Monte-Carlo case).
        b_boot: Bootstrap resamples for the 95 % CI.

    Returns:
        Row with means, difference, cluster sign-flip p, n_nonzero and cluster-bootstrap CI.
        Every call seeds its generators with ``SEED``.
    """
    cinv, ncl, sizes = make_clusters(clusters)
    fa, fb = METRICS[metric](ranks[a]), METRICS[metric](ranks[b])
    d = fb - fa  # positive = b better
    sf = cluster_sign_flip(d, cinv, ncl, SEED, b_perm)
    return {"family": family, "metric": metric, "a": a, "b": b, "mean_a": float(fa.mean()), "mean_b": float(fb.mean()),
            "diff_b_minus_a": float(d.mean()), "p_cluster_signflip": sf["p"], "p_method": sf["method"],
            "n_nonzero_clusters": sf["n_nonzero"], "p_is_upper_bound": sf["mc_floor"],
            "ci95_cluster_bootstrap": cluster_bootstrap_ci(d, cinv, ncl, sizes, SEED, b_boot),
            "not_informative": sf["n_nonzero"] < THRESHOLD}


def compare(ranks: dict[str, np.ndarray], clusters: list[str], b_perm: int = B_PERM, b_boot: int = B_BOOT) -> dict:
    """Run all comparisons for the retrievers in ``ranks`` (keys: ``bm25``, optional ``bm25_de``, embedders).

    Families whose reference is missing in ``ranks`` are skipped (BM25-de, bge-m3 for the extension models).

    Returns:
        Dict with the result rows (``tests``), number of queries and clusters; Holm is applied per family.
    """
    _, ncl, _ = make_clusters(clusters)
    emb = [m for m in EMBEDDERS if m in ranks]
    ext = [m for m in EXTENSION if m in ranks]
    succ_k = ("Success@1", "Success@5", "Success@10")
    rows = []

    def one(family: str, a: str, b: str, metric: str) -> None:
        rows.append(paired_test(ranks, clusters, a, b, metric, family, b_perm, b_boot))

    for m in emb:
        one("confirmatory: embedder vs BM25, MRR@10", "bm25", m, "MRR@10")
    for a, b in itertools.combinations(emb, 2):
        one("explorative: embedder pairs, MRR@10", a, b, "MRR@10")
    for metric in succ_k:
        for m in emb:
            one(f"explorative: embedder vs BM25, {metric}", "bm25", m, metric)
    if "bm25_de" in ranks:
        for metric in METRICS:
            one("explorative: BM25-de vs BM25 standard (4 metrics)", "bm25", "bm25_de", metric)
        for metric in METRICS:
            for m in emb:
                one(f"explorative: embedder vs BM25-de, {metric}", "bm25_de", m, metric)
    for tag, base, label in (("a", "bm25", "BM25 standard"), ("b", "bm25_de", "BM25-de"), ("c", "bge-m3", "bge-m3")):
        if base in ranks:
            for metric in METRICS:
                for m in ext:
                    one(f"explorative ({tag}): new model vs {label}, {metric}", base, m, metric)
    apply_holm(rows, lambda fam: 6 if fam.startswith("confirmatory") else None)  # planned size 6 even if fewer models exist yet
    return {"tests": rows, "n_queries": len(clusters), "n_clusters": ncl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=CODE / "output" / DS)
    ap.add_argument("--harness-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--out", type=Path, default=CODE / "output" / "harness" / f"comparison_{DS}.json")
    a = ap.parse_args()
    qids, clusters = load_queries_clusters(a.dataset_dir)
    sources = {m: a.harness_dir / m / f"per_query_{DS}.csv" for m in EMBEDDERS + EXTENSION}
    sources = {k: p for k, p in sources.items() if p.exists()}
    sources.update(bm25=BM25_CSV, bm25_de=BM25_DE_CSV)
    if not BM25_CSV.exists():
        sys.exit(f"no BM25 per-query file: {BM25_CSV}")
    ranks = {k: load_ranks(p, qids) for k, p in sources.items() if p.exists()}
    res = compare(ranks, clusters)
    res["meta"] = make_meta({k: p for k, p in sources.items() if k in ranks},
                            "Holm: confirmatory family size 6; explorative families corrected within family. "
                            "diff = b minus a; MC p at floor 1/(B+1) is an upper bound.",
                            exact_if_n_nonzero_le=16, threshold_not_informative=THRESHOLD)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    for r in res["tests"]:
        if r["family"].startswith("confirmatory"):
            print(f"{r['b']:>24} vs {r['a']}: diff={r['diff_b_minus_a']:+.4f} p={r['p_cluster_signflip']:.3g} "
                  f"holm={r['p_holm']:.3g} n!=0={r['n_nonzero_clusters']} CI={r['ci95_cluster_bootstrap']}")


if __name__ == "__main__":
    main()
