"""Paired comparison of all available retrievers on GermanQuAD per docs/evaluation_protocol.md.

Usage: python compare_models.py [--harness-dir output/harness] [--bm25-csv PATH] [--out output/harness/comparison_germanquad.json]
Input: per_query_germanquad.csv of every model directory in --harness-dir (rank of first relevant doc over the
full ranking), BM25 per-query ranks (harness/bm25/per_query_germanquad.csv if present, else
output/bm25_v2/per_query_ranks.csv), qrels/queries from output/germanquad (cluster = first sorted gold doc).
Confirmatory family: each embedder vs BM25 on MRR@10 (Holm, family size 6). Explorative families (Holm within
each): embedder pairs on MRR@10, and Success@1/5/10 vs BM25 (+ exact McNemar with b, c).
Output: JSON with p (raw, Holm), n_nonzero, cluster-bootstrap CI (B=10000), all seeded with 42 per test.
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
from significance import (B_BOOT, B_PERM, SEED, cluster_bootstrap_ci, cluster_sign_flip, cluster_sums,  # noqa: E402
                          holm, make_clusters, mcnemar_exact)

EMBEDDERS = ["e5-large", "bge-m3", "gte-multilingual-base", "jina-v3", "openai-3-small", "openai-3-large"]
THRESHOLD = 20  # n_nonzero (and b+c) below this: "nicht aussagekräftig" (protocol, Detailfestlegungen)
DS = "germanquad"


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


def compare(ranks: dict[str, np.ndarray], clusters: list[str], bm25_key: str = "bm25",
            b_perm: int = B_PERM, b_boot: int = B_BOOT) -> dict:
    """Run all comparisons for the retrievers in ``ranks`` (must contain ``bm25_key``).

    Returns:
        Dict with the result rows (``tests``) and family definitions; Holm is applied per family.
    """
    cinv, ncl, sizes = make_clusters(clusters)
    n = len(clusters)
    emb = [m for m in EMBEDDERS if m in ranks] + sorted(m for m in ranks if m not in EMBEDDERS and m != bm25_key)
    rows = []

    def one(family: str, a: str, b: str, metric: str) -> None:
        fa, fb = METRICS[metric](ranks[a]), METRICS[metric](ranks[b])
        d = fb - fa  # positive = b better
        sf = cluster_sign_flip(d, cinv, ncl, SEED, b_perm)
        row = {"family": family, "metric": metric, "a": a, "b": b, "mean_a": float(fa.mean()), "mean_b": float(fb.mean()),
               "diff_b_minus_a": float(d.mean()), "p_cluster_signflip": sf["p"], "p_method": sf["method"],
               "n_nonzero_clusters": sf["n_nonzero"], "p_is_upper_bound": sf["mc_floor"],
               "ci95_cluster_bootstrap": cluster_bootstrap_ci(d, cinv, ncl, sizes, SEED, b_boot),
               "not_informative": sf["n_nonzero"] < THRESHOLD}
        if metric.startswith("Success"):
            mc = mcnemar_exact(fa, fb)
            row.update(mcnemar_b_a_only=mc["b"], mcnemar_c_b_only=mc["c"], mcnemar_b_plus_c=mc["b_plus_c"],
                       mcnemar_p=mc["p"], mcnemar_note="exact, optimistic (ignores clusters), descriptive",
                       not_informative=row["not_informative"] or mc["b_plus_c"] < THRESHOLD)
        rows.append(row)

    for m in emb:
        one("confirmatory: embedder vs BM25, MRR@10", bm25_key, m, "MRR@10")
    for a, b in itertools.combinations(emb, 2):
        one("explorative: embedder pairs, MRR@10", a, b, "MRR@10")
    for metric in ("Success@1", "Success@5", "Success@10"):
        for m in emb:
            one(f"explorative: embedder vs BM25, {metric}", bm25_key, m, metric)
    for fam in {r["family"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        size = 6 if fam.startswith("confirmatory") else None  # planned family size 6 even if fewer models exist yet
        for i, h in zip(idx, holm([rows[i]["p_cluster_signflip"] for i in idx], size)):
            rows[i]["p_holm"] = h
    return {"tests": rows, "n_queries": n, "n_clusters": ncl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=CODE / "output" / DS)
    ap.add_argument("--harness-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--bm25-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=CODE / "output" / "harness" / f"comparison_{DS}.json")
    a = ap.parse_args()
    queries, docs, qrels = (load_jsonl(a.dataset_dir / f"{x}.normalized.jsonl") for x in ("queries", "docs", "qrels"))
    qids = [str(q["query_id"]) for q in queries]
    gold = build_gold(qrels, {str(d["doc_id"]) for d in docs}, 1)
    assert all(q in gold for q in qids), "queries without gold are not supported"
    clusters = [sorted(gold[q])[0] for q in qids]
    ranks, sources = {}, {}
    for m in EMBEDDERS:
        f = a.harness_dir / m / f"per_query_{DS}.csv"
        if f.exists():
            ranks[m], sources[m] = load_ranks(f, qids), rel(f)
    bm = a.bm25_csv or next((p for p in (a.harness_dir / "bm25" / f"per_query_{DS}.csv",
                                         CODE / "output" / "bm25_v2" / "per_query_ranks.csv") if p.exists()), None)
    if bm is None:
        sys.exit("no BM25 per-query file found")
    ranks["bm25"], sources["bm25"] = load_ranks(bm, qids), rel(bm)
    res = compare(ranks, clusters)
    res["meta"] = {"dataset": DS, "seed": SEED, "B_permutation": B_PERM, "B_bootstrap": B_BOOT, "exact_if_n_nonzero_le": 16,
                   "threshold_not_informative": THRESHOLD, "sources": sources, "numpy": np.__version__,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "note": "Holm: confirmatory family size 6; explorative families corrected within family. "
                           "diff = b minus a; MC p at floor 1/(B+1) is an upper bound."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    for r in res["tests"]:
        if r["family"].startswith("confirmatory"):
            print(f"{r['b']:>24} vs {r['a']}: diff={r['diff_b_minus_a']:+.4f} p={r['p_cluster_signflip']:.3g} "
                  f"holm={r['p_holm']:.3g} n!=0={r['n_nonzero_clusters']} CI={r['ci95_cluster_bootstrap']}")


if __name__ == "__main__":
    main()
