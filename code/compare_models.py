"""Gepaarte Vergleiche aller elf Retriever auf GermanQuAD (docs/evaluation_protocol.md), ein Lauf, 55 Tests.

Aufruf: python compare_models.py [--harness-dir output/harness] [--out output/harness/comparison_germanquad.json]
Eingabe: per_query_germanquad.csv der neun Embedder, BM25 (output/bm25_v2/per_query_ranks.csv) und BM25-de
(output/bm25_de/per_query_ranks.csv) mit dem Rang der ersten relevanten Passage im vollen Ranking; Fragen und
Gold-Passagen aus output/germanquad (Cluster = erste sortierte Gold-Passage).
Alle Tests nutzen die Primärmetrik MRR@10 und denselben Cluster-Randomisierungstest; Holm gilt je Familie, die
Familiengröße ist die tatsächliche Testzahl:
  F1: alle 36 Paare der neun Modelle; F2: je Modell gegen BM25 (9); F3: je Modell gegen BM25-de (9);
  F4: BM25-de gegen BM25 (1, ohne Korrektur).
Ausgabe: JSON mit p (roh, Holm), n_nonzero und Cluster-Bootstrap-CI (B_BOOT); je Test SEED (beide aus pylib/significance.py).
Sekundärmetriken (Success@k, MRR@k) werden nur deskriptiv in Tabellen berichtet (make_results.py).

Role in the experiment (English summary): last computation step before the tables. Upstream: embed_eval.py (one
per_query_germanquad.csv per model), scripts/run_bm25.py and bm25_german.py (per_query_ranks.csv). Downstream:
make_results.py and notebook 05 read comparison_germanquad.json (tests/test_comparison_design.py checks it).
Unit of analysis is the query, but queries sharing the same gold passage are not independent, so tests resample or
sign-flip whole clusters (one cluster = all queries whose first sorted gold passage is the same). Ranks are over the
full corpus ranking (0 = no relevant passage), so MRR@10 is derived here from the rank, not from stored top-10 lists.
Running the script overwrites --out (default output/harness/comparison_germanquad.json).
"""
from __future__ import annotations

import argparse, itertools, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
from local_dataset_io import load_jsonl  # noqa: E402
from retrieval_metrics import build_gold, reciprocal_rank_at_k, success_at_k  # noqa: E402
from significance import B_BOOT, B_PERM, EXACT_MAX_N, SEED, cluster_bootstrap_ci, cluster_sign_flip, holm, make_clusters  # noqa: E402

MODELS = ["e5-large", "bge-m3", "gte-multilingual-base", "jina-v3", "openai-3-small", "openai-3-large",
          "qwen3-embedding-0.6b", "arctic-embed-l-v2", "jina-v2-base-de"]
LEN512_MODELS = ["bge-m3", "gte-multilingual-base", "jina-v3"]  # models that also have a <model>__len512 run
THRESHOLD = 20  # n_nonzero darunter: "nicht aussagekräftig" (Protokoll, Detailfestlegungen)
CONSISTENCY_TOL = 1e-9  # max. absolute deviation between stored JSON values and values recomputed from the ranks
DS = "germanquad"  # dataset name (also the file suffix per_query_<DS>.csv and the "dataset" column filter)
BM25_CSV = CODE / "output" / "bm25_v2" / "per_query_ranks.csv"
BM25_DE_CSV = CODE / "output" / "bm25_de" / "per_query_ranks.csv"
BM25_JSON = BM25_CSV.with_name("bm25_results.json")  # stored metrics of the BM25 run (compute_metrics result)
BM25_DE_JSON = BM25_DE_CSV.with_name("bm25_results.json")  # stored metrics of the BM25-de run


# Per-query metric functions (rank array -> per-query values). The tests use only MRR@10; the Success@k entries are
# kept for descriptive use.
METRICS = {"MRR@10": lambda r: reciprocal_rank_at_k(r, 10), "Success@1": lambda r: success_at_k(r, 1),
           "Success@5": lambda r: success_at_k(r, 5), "Success@10": lambda r: success_at_k(r, 10)}


def load_ranks(path: Path, qids: list[str]) -> np.ndarray:
    """Liest die Ränge der ersten relevanten Passage, ausgerichtet an ``qids`` (Harness- oder bm25_v2-Format).

    Args:
        path: ``per_query_germanquad.csv`` of an embedder (column ``rank_first_relevant``) or ``per_query_ranks.csv``
            of a BM25 run (column ``best_gold_rank_full``; rows filtered to ``DS``).
        qids: Query IDs in the order of ``queries.normalized.jsonl``; the CSV rows must have exactly this order.

    Returns:
        Integer array aligned with ``qids``: rank of the first relevant passage in the full ranking
        (1-based); an empty CSV cell (no gold passage) becomes 0.

    Raises:
        ValueError: Wenn die Frage-IDs nach dem Filtern auf den Datensatz nicht zu ``qids`` passen.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "dataset" in df:
        # Boolean row filter: df["dataset"] == DS is True for GermanQuAD rows only; indexing with it keeps just those
        # rows.
        df = df[df["dataset"] == DS]
    col = "rank_first_relevant" if "rank_first_relevant" in df else "best_gold_rank_full"
    # Compare the query IDs of the CSV (list, file order) with qids: ranks are only meaningful if both orders are
    # identical.
    if df["query_id"].tolist() != qids:
        raise ValueError(f"{path}: query ids/order differ from queries.normalized.jsonl")
    # For every cell of the rank column: empty cell (query without gold passage) -> 0, otherwise the rank as int
    # (the string may look like "3.0", hence int(float(x))).
    return np.array([int(float(x)) if x != "" else 0 for x in df[col]], dtype=int)


def load_queries_clusters(dataset_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Liest Frage-IDs und Cluster-Labels (erste sortierte Gold-Passage je Frage).

    Args:
        dataset_dir: Ordner mit den ``*.normalized.jsonl``-Dateien (Standard ``output/germanquad``).

    Returns:
        Tuple ``(qids, clusters)``: query IDs in file order and, same order, the doc ID of the alphabetically first
        gold passage of each query as cluster label, e.g. ``(["q1", "q2"], ["d7", "d7"])``.

    Raises:
        AssertionError: Wenn eine Frage keine Gold-Passage hat.
    """
    ds = dataset_dir or CODE / "output" / DS  # ds: dataset directory
    # Generator over the three roles: load each normalized JSONL file; unpacking gives the three record lists.
    queries, docs, qrels = (load_jsonl(ds / f"{x}.normalized.jsonl") for x in ("queries", "docs", "qrels"))
    # List comprehension: query IDs as strings in file order.
    qids = [str(q["query_id"]) for q in queries]
    # Set comprehension: IDs of all corpus passages; build_gold keeps only qrels that point to them.
    gold = build_gold(qrels, {str(d["doc_id"]) for d in docs}, 1)
    # Generator over all query IDs: every query must have a gold passage, otherwise its cluster label would be
    # undefined.
    assert all(q in gold for q in qids), "queries without gold are not supported"
    # List comprehension: cluster label of each query = alphabetically first gold passage ID (stable choice if there
    # are several).
    return qids, [sorted(gold[q])[0] for q in qids]


def paired_test(ranks: dict[str, np.ndarray], clusters: list[str], a: str, b: str, family: str,
                b_perm: int = B_PERM, b_boot: int = B_BOOT) -> dict:
    """Gepaarter Test von Retriever ``b`` gegen ``a`` auf MRR@10 (eine Ergebniszeile, noch ohne Holm).

    Args:
        ranks: Rang der ersten relevanten Passage je Retriever (0 = keine), ausgerichtet an den Fragen.
        clusters: Cluster-Label (Gold-Passage) je Frage.
        a: Referenz-Retriever.
        b: Verglichener Retriever; positive ``diff_b_minus_a`` heißt: ``b`` ist besser.
        family: Familienlabel der Zeile; Holm wird in ``compare`` je Familie angewandt.
        b_perm: Anzahl Vorzeichenwechsel (Monte-Carlo-Fall).
        b_boot: Bootstrap-Ziehungen für das 95-%-CI.

    Returns:
        Zeile mit Mittelwerten, Differenz, Cluster-Sign-Flip-p, n_nonzero und Cluster-Bootstrap-CI;
        die Zufallsgeneratoren starten je Aufruf mit ``SEED``.
    """
    # cinv: cluster index per query; ncl: number of clusters; sizes: queries per cluster
    cinv, ncl, sizes = make_clusters(clusters)
    # fa, fb: per-query reciprocal rank (cutoff 10) of retriever a / b; d: per-query difference b - a
    # Per-query reciprocal ranks of both retrievers as arrays (one entry per query, same order).
    fa, fb = reciprocal_rank_at_k(ranks[a], 10), reciprocal_rank_at_k(ranks[b], 10)
    # Element-wise difference per query, e.g. fb = [1.0, 0.5], fa = [0.5, 0.5] -> d = [0.5, 0.0]; positive = b better.
    # The means of fa, fb and d (below) are the MRR@10 values and their difference.
    d = fb - fa
    # sf: sign-flip result (p, method, n_nonzero, mc_floor); the CI is a cluster bootstrap of the mean difference
    sf = cluster_sign_flip(d, cinv, ncl, SEED, b_perm)
    return {"family": family, "metric": "MRR@10", "a": a, "b": b, "mean_a": float(fa.mean()), "mean_b": float(fb.mean()),
            "diff_b_minus_a": float(d.mean()), "p_cluster_signflip": sf["p"], "p_method": sf["method"],
            "n_nonzero_clusters": sf["n_nonzero"], "p_is_upper_bound": sf["mc_floor"],
            "ci95_cluster_bootstrap": cluster_bootstrap_ci(d, cinv, ncl, sizes, SEED, b_boot),
            "not_informative": sf["n_nonzero"] < THRESHOLD}


def compare(ranks: dict[str, np.ndarray], clusters: list[str], b_perm: int = B_PERM, b_boot: int = B_BOOT) -> dict:
    """Rechnet die 55 Tests der vier Familien; ``ranks`` enthält ``bm25``, ``bm25_de`` und alle ``MODELS``.

    Args:
        ranks: Retriever key -> integer array with the rank of the first relevant passage per query (0 = none),
            aligned with ``clusters``, e.g. ``{"bm25": np.array([1, 0]), "e5-large": np.array([2, 1]), ...}``.
        clusters: Cluster label (first sorted gold passage) per query, e.g. ``["d7", "d9"]``.
        b_perm: Number of Monte-Carlo sign flips (only used if more than
            ``significance.EXACT_MAX_N`` clusters have a non-zero effect).
        b_boot: Number of bootstrap resamples for the 95 % CI.

    Returns:
        Dict mit den Ergebniszeilen (``tests``, je Familie Holm mit der tatsächlichen Testzahl),
        Fragenzahl und Clusterzahl.
    """
    # --- Define the 55 comparisons as (family, reference a, compared b) ---
    # Pairs of two different models without repetition and order (36 for 9 models), e.g. combinations([A, B, C], 2) ->
    # (A, B), (A, C), (B, C).
    pairs = [("F1: Modellpaare", a, b) for a, b in itertools.combinations(MODELS, 2)]
    # One test per model: BM25 as reference a, the model as compared b.
    pairs += [("F2: Modell gegen BM25", "bm25", m) for m in MODELS]
    # One test per model: BM25-de as reference a, the model as compared b.
    pairs += [("F3: Modell gegen BM25-de", "bm25_de", m) for m in MODELS]
    pairs += [("F4: BM25-de gegen BM25", "bm25", "bm25_de")]
    # List comprehension: run one paired test per planned comparison (family label, reference a, compared b).
    rows = [paired_test(ranks, clusters, a, b, fam, b_perm, b_boot) for fam, a, b in pairs]
    # --- Holm correction separately per family; family size = number of tests in that family ---
    # (F4 has one test, so Holm leaves its p unchanged.)
    # For every family (set of distinct family labels, so each is handled once): correct its p-values with Holm.
    for fam in {r["family"] for r in rows}:
        # List comprehension: the result rows of this family only.
        grp = [r for r in rows if r["family"] == fam]  # grp: the rows of this family
        # Pair each row of the family with its Holm-adjusted p-value (holm returns them in the same order);
        # the inner list comprehension passes the raw p-values of the family.
        for r, h in zip(grp, holm([r["p_cluster_signflip"] for r in grp])):
            r["p_holm"] = h
    return {"tests": rows, "n_queries": len(clusters), "n_clusters": make_clusters(clusters)[1]}


def main() -> None:
    """Load all per-query ranks, run the 55 tests and write the JSON and a console summary.

    Writes ``--out`` (existing file is overwritten) with ``tests``, ``n_queries``, ``n_clusters`` and ``meta``
    (seed, B values, thresholds, source files, numpy version, timestamp).
    """
    # --- Load data: query IDs, cluster labels and ranks per retriever ---
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=CODE / "output" / DS)
    ap.add_argument("--harness-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--out", type=Path, default=CODE / "output" / "harness" / f"comparison_{DS}.json")
    a = ap.parse_args()
    qids, clusters = load_queries_clusters(a.dataset_dir)
    # sources: retriever key -> CSV with per-query ranks (nine embedders from the harness plus both BM25 variants)
    # Dict comprehension: model key -> path of that model's per-query rank CSV.
    sources = {m: a.harness_dir / m / f"per_query_{DS}.csv" for m in MODELS}
    sources.update(bm25=BM25_CSV, bm25_de=BM25_DE_CSV)
    # --- Run tests and write results ---
    # Dict comprehension: load the rank array of every retriever (aligned with qids) before running the tests.
    res = compare({k: load_ranks(p, qids) for k, p in sources.items()}, clusters)
    # rel: repo-relative POSIX path for the meta record (falls back to the given path outside the repo)
    rel = lambda p: Path(p).resolve().relative_to(CODE.parent).as_posix() if CODE.parent in Path(p).resolve().parents else str(p)
    res["meta"] = {"dataset": DS, "seed": SEED, "B_permutation": B_PERM, "B_bootstrap": B_BOOT,
                   "exact_if_n_nonzero_le": EXACT_MAX_N, "threshold_not_informative": THRESHOLD,
                   # Dict comprehension: retriever key -> repo-relative path of its input CSV (provenance).
                   "sources": {k: rel(p) for k, p in sources.items()}, "numpy": np.__version__,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "note": "Holm je Familie mit Familiengröße = Testzahl (36, 9, 9, 1); F4 ohne Korrektur. "
                           "diff = b minus a; MC-p an der Untergrenze 1/(B+1) ist eine obere Schranke."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    # Console summary; "n!=0" is the number of clusters with a non-zero summed difference.
    # For every test result: print one summary line (retriever pair, difference, raw and Holm p, non-zero clusters).
    for r in res["tests"]:
        print(f"{r['family'][:2]} {r['b']:>22} vs {r['a']:<22} diff={r['diff_b_minus_a']:+.4f} "
              f"p={r['p_cluster_signflip']:.3g} holm={r['p_holm']:.3g} n!=0={r['n_nonzero_clusters']}")


if __name__ == "__main__":
    main()
