"""Gepaarte Vergleiche aller elf Retriever auf GermanQuAD (docs/evaluation_protocol.md), ein Lauf, 55 Tests.

Aufruf: python compare_models.py [--harness-dir output/harness] [--out output/harness/comparison_germanquad.json]
Eingabe: per_query_germanquad.csv der neun Embedder, BM25 (output/bm25_v2/per_query_ranks.csv) und BM25-de
(output/bm25_de/per_query_ranks.csv) mit dem Rang der ersten relevanten Passage im vollen Ranking; Fragen und
Gold-Passagen aus output/germanquad (Cluster = erste sortierte Gold-Passage).
Alle Tests nutzen die Primärmetrik MRR@10 und denselben Cluster-Randomisierungstest; Holm gilt je Familie, die
Familiengröße ist die tatsächliche Testzahl:
  F1: alle 36 Paare der neun Modelle; F2: je Modell gegen BM25 (9); F3: je Modell gegen BM25-de (9);
  F4: BM25-de gegen BM25 (1, ohne Korrektur).
Ausgabe: JSON mit p (roh, Holm), n_nonzero und Cluster-Bootstrap-CI (B=10000); je Test Seed 42.
Sekundärmetriken (Success@k, MRR@k) werden nur deskriptiv in Tabellen berichtet (make_results.py).
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
from retrieval_metrics import build_gold  # noqa: E402
from significance import B_BOOT, B_PERM, SEED, cluster_bootstrap_ci, cluster_sign_flip, holm, make_clusters  # noqa: E402

MODELS = ["e5-large", "bge-m3", "gte-multilingual-base", "jina-v3", "openai-3-small", "openai-3-large",
          "qwen3-embedding-0.6b", "arctic-embed-l-v2", "jina-v2-base-de"]
THRESHOLD = 20  # n_nonzero darunter: "nicht aussagekräftig" (Protokoll, Detailfestlegungen)
DS = "germanquad"
BM25_CSV = CODE / "output" / "bm25_v2" / "per_query_ranks.csv"
BM25_DE_CSV = CODE / "output" / "bm25_de" / "per_query_ranks.csv"


def succ(r: np.ndarray, k: int) -> np.ndarray:
    """Success@k-Indikator aus Rängen der ersten relevanten Passage (0 = keine relevante Passage)."""
    return ((r > 0) & (r <= k)).astype(float)


def rr10(r: np.ndarray) -> np.ndarray:
    """Reziproker Rang mit Cutoff 10 aus Rängen der ersten relevanten Passage (0 = keine relevante Passage)."""
    return np.where((r > 0) & (r <= 10), 1.0 / np.maximum(r, 1), 0.0)


METRICS = {"MRR@10": rr10, "Success@1": lambda r: succ(r, 1), "Success@5": lambda r: succ(r, 5),
           "Success@10": lambda r: succ(r, 10)}


def load_ranks(path: Path, qids: list[str]) -> np.ndarray:
    """Liest die Ränge der ersten relevanten Passage, ausgerichtet an ``qids`` (Harness- oder bm25_v2-Format).

    Raises:
        ValueError: Wenn die Frage-IDs nach dem Filtern auf den Datensatz nicht zu ``qids`` passen.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "dataset" in df:
        df = df[df["dataset"] == DS]
    col = "rank_first_relevant" if "rank_first_relevant" in df else "best_gold_rank_full"
    if df["query_id"].tolist() != qids:
        raise ValueError(f"{path}: query ids/order differ from queries.normalized.jsonl")
    return np.array([int(float(x)) if x != "" else 0 for x in df[col]], dtype=int)


def load_queries_clusters(dataset_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Liest Frage-IDs und Cluster-Labels (erste sortierte Gold-Passage je Frage).

    Args:
        dataset_dir: Ordner mit den ``*.normalized.jsonl``-Dateien (Standard ``output/germanquad``).

    Raises:
        AssertionError: Wenn eine Frage keine Gold-Passage hat.
    """
    ds = dataset_dir or CODE / "output" / DS
    queries, docs, qrels = (load_jsonl(ds / f"{x}.normalized.jsonl") for x in ("queries", "docs", "qrels"))
    qids = [str(q["query_id"]) for q in queries]
    gold = build_gold(qrels, {str(d["doc_id"]) for d in docs}, 1)
    assert all(q in gold for q in qids), "queries without gold are not supported"
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
    cinv, ncl, sizes = make_clusters(clusters)
    fa, fb = rr10(ranks[a]), rr10(ranks[b])
    d = fb - fa
    sf = cluster_sign_flip(d, cinv, ncl, SEED, b_perm)
    return {"family": family, "metric": "MRR@10", "a": a, "b": b, "mean_a": float(fa.mean()), "mean_b": float(fb.mean()),
            "diff_b_minus_a": float(d.mean()), "p_cluster_signflip": sf["p"], "p_method": sf["method"],
            "n_nonzero_clusters": sf["n_nonzero"], "p_is_upper_bound": sf["mc_floor"],
            "ci95_cluster_bootstrap": cluster_bootstrap_ci(d, cinv, ncl, sizes, SEED, b_boot),
            "not_informative": sf["n_nonzero"] < THRESHOLD}


def compare(ranks: dict[str, np.ndarray], clusters: list[str], b_perm: int = B_PERM, b_boot: int = B_BOOT) -> dict:
    """Rechnet die 55 Tests der vier Familien; ``ranks`` enthält ``bm25``, ``bm25_de`` und alle ``MODELS``.

    Returns:
        Dict mit den Ergebniszeilen (``tests``, je Familie Holm mit der tatsächlichen Testzahl),
        Fragenzahl und Clusterzahl.
    """
    pairs = [("F1: Modellpaare", a, b) for a, b in itertools.combinations(MODELS, 2)]
    pairs += [("F2: Modell gegen BM25", "bm25", m) for m in MODELS]
    pairs += [("F3: Modell gegen BM25-de", "bm25_de", m) for m in MODELS]
    pairs += [("F4: BM25-de gegen BM25", "bm25", "bm25_de")]
    rows = [paired_test(ranks, clusters, a, b, fam, b_perm, b_boot) for fam, a, b in pairs]
    for fam in {r["family"] for r in rows}:
        grp = [r for r in rows if r["family"] == fam]
        for r, h in zip(grp, holm([r["p_cluster_signflip"] for r in grp])):
            r["p_holm"] = h
    return {"tests": rows, "n_queries": len(clusters), "n_clusters": make_clusters(clusters)[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=CODE / "output" / DS)
    ap.add_argument("--harness-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--out", type=Path, default=CODE / "output" / "harness" / f"comparison_{DS}.json")
    a = ap.parse_args()
    qids, clusters = load_queries_clusters(a.dataset_dir)
    sources = {m: a.harness_dir / m / f"per_query_{DS}.csv" for m in MODELS}
    sources.update(bm25=BM25_CSV, bm25_de=BM25_DE_CSV)
    res = compare({k: load_ranks(p, qids) for k, p in sources.items()}, clusters)
    rel = lambda p: Path(p).resolve().relative_to(CODE.parent).as_posix() if CODE.parent in Path(p).resolve().parents else str(p)
    res["meta"] = {"dataset": DS, "seed": SEED, "B_permutation": B_PERM, "B_bootstrap": B_BOOT,
                   "exact_if_n_nonzero_le": 16, "threshold_not_informative": THRESHOLD,
                   "sources": {k: rel(p) for k, p in sources.items()}, "numpy": np.__version__,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "note": "Holm je Familie mit Familiengröße = Testzahl (36, 9, 9, 1); F4 ohne Korrektur. "
                           "diff = b minus a; MC-p an der Untergrenze 1/(B+1) ist eine obere Schranke."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for r in res["tests"]:
        print(f"{r['family'][:2]} {r['b']:>22} vs {r['a']:<22} diff={r['diff_b_minus_a']:+.4f} "
              f"p={r['p_cluster_signflip']:.3g} holm={r['p_holm']:.3g} n!=0={r['n_nonzero_clusters']}")


if __name__ == "__main__":
    main()
