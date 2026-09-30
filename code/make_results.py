"""Build the Chapter-4 result tables and the central figure from the harness outputs (GermanQuAD).

Usage: python make_results.py [--harness-dir output/harness] [--out-dir output/harness/results] [--recompute]
Input: <harness>/<model>/metrics_germanquad.json + per_query_germanquad.csv (also <model>__len512), BM25 per-query
ranks (as compare_models.py), <harness>/comparison_germanquad.json (re-created by compare_models.py if missing,
stale w.r.t. the available models, or with --recompute; runtime about 30 s).
Output (out-dir): table_main.md/.csv, table_significance.md, table_len512.md, fig_mrr10_ci.pdf/.png, README-style
notes on skipped models are printed and appended to each table file.
Cell values are read from the inputs; the CI is the 95% cluster-bootstrap CI (B=10000, seed 42) of the mean.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
import compare_models as cm  # noqa: E402
from local_dataset_io import load_jsonl  # noqa: E402
from retrieval_metrics import build_gold  # noqa: E402
from significance import cluster_bootstrap_ci, make_clusters, SEED  # noqa: E402

DS = cm.DS
LABEL = {"bm25": "BM25", "e5-large": "multilingual-e5-large", "bge-m3": "bge-m3",
         "gte-multilingual-base": "gte-multilingual-base", "jina-v3": "jina-embeddings-v3",
         "openai-3-small": "text-embedding-3-small", "openai-3-large": "text-embedding-3-large"}
LEN512_MODELS = ["bge-m3", "gte-multilingual-base", "jina-v3"]


def de(x: float, nd: int = 3, sign: bool = False) -> str:
    """German number format (decimal comma)."""
    return f"{x:+.{nd}f}".replace(".", ",") if sign else f"{x:.{nd}f}".replace(".", ",")


def dep(p: float, floor: bool) -> str:
    """German p-value format; Monte-Carlo p at the floor is reported as upper bound (value = bound)."""
    if p < 0.0001 and not floor:  # exact value in scientific notation (no "< 0,0001")
        return f"{p:.1e}".replace(".", ",")
    # bounds below 1e-3 in scientific notation, so that e.g. 15/(B+1) = 1.5e-04 is not rounded to 0,0001
    s = f"{p:.4f}" if p >= (0.001 if floor else 0.0001) else f"{p:.1e}"
    return ("≤ " if floor else "") + s.replace(".", ",")


def md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a pipe table."""
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def revision(exp: dict) -> str:
    """Model revision (HF commit) or API model name with retrieval date."""
    if exp.get("backend") == "openai" or "openai_api" in exp:
        api = exp["openai_api"]
        return f"{api['returned_models'][0]} (API, abgerufen {api['first_request_utc'][:10]})"
    rev = exp.get("model_revision") or exp.get("model_revision_hub_head") or "n/a"
    return f"{exp['model_name']}@{rev[:12]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--out-dir", type=Path, default=CODE / "output" / "harness" / "results")
    ap.add_argument("--recompute", action="store_true", help="force compare_models.py rerun")
    a = ap.parse_args()
    H, O = a.harness_dir, a.out_dir
    O.mkdir(parents=True, exist_ok=True)

    dd = CODE / "output" / DS
    queries, docs, qrels = (load_jsonl(dd / f"{x}.normalized.jsonl") for x in ("queries", "docs", "qrels"))
    qids = [str(q["query_id"]) for q in queries]
    gold = build_gold(qrels, {str(d["doc_id"]) for d in docs}, 1)
    cinv, ncl, sizes = make_clusters([sorted(gold[q])[0] for q in qids])

    ranks, meta, notes = {}, {}, []
    for m in cm.EMBEDDERS + [f"{x}__len512" for x in LEN512_MODELS]:
        f = H / m / f"per_query_{DS}.csv"
        if f.exists() and (H / m / f"metrics_{DS}.json").exists():
            ranks[m] = cm.load_ranks(f, qids)
            meta[m] = json.loads((H / m / f"metrics_{DS}.json").read_text())
        else:
            if "__len512" not in m:
                notes.append(f"{m}: nicht vorhanden (übersprungen)")
    bm = next((p for p in (H / "bm25" / f"per_query_{DS}.csv", CODE / "output" / "bm25_v2" / "per_query_ranks.csv")
               if p.exists()), None)
    if bm is None:
        sys.exit("no BM25 per-query file found")
    ranks["bm25"] = cm.load_ranks(bm, qids)
    if "e5-large" not in ranks and not any(m in ranks for m in cm.EMBEDDERS):
        sys.exit("no embedder outputs found")

    # main table: embedder values from metrics json (cross-checked against ranks), BM25 from ranks
    rows, ci = [], {}
    for m in ["bm25"] + [m for m in cm.EMBEDDERS if m in ranks]:
        r = ranks[m]
        v = {"MRR@10": cm.rr10(r).mean(), "Success@1": cm.succ(r, 1).mean(), "Success@5": cm.succ(r, 5).mean(),
             "Success@10": cm.succ(r, 10).mean(), "MRR@5": np.where((r > 0) & (r <= 5), 1 / np.maximum(r, 1), 0).mean()}
        if m in meta:
            mm = {x["k"]: x for x in meta[m]["metrics"]}
            j = {"MRR@10": mm[10]["mrr_at_k"], "Success@1": mm[1]["success_at_k"], "Success@5": mm[5]["success_at_k"],
                 "Success@10": mm[10]["success_at_k"], "MRR@5": mm[5]["mrr_at_k"]}
            assert all(abs(j[k] - v[k]) < 1e-9 for k in v), f"{m}: metrics json differs from per-query ranks"
            v = j
        ci[m] = cluster_bootstrap_ci(cm.rr10(r), cinv, ncl, sizes)
        rev = revision(meta[m]["experiment"]) if m in meta else f"rank_bm25 (k1=1.5, b=0.75), {bm.relative_to(CODE)}"
        rows.append({"Retriever": LABEL[m], **v, "Modellrevision / API-Modell": rev})
    t = pd.DataFrame(rows)
    t.to_csv(O / "table_main.csv", index=False)
    tm = t.copy()
    for c in ["MRR@10", "Success@1", "Success@5", "Success@10", "MRR@5"]:
        tm[c] = tm[c].map(de)
    n = len(qids)
    foot = (f"\nGermanQuAD (Test), n = {n} Queries, {ncl} Cluster (Gold-Passagen), Corpus {len(docs)} Passagen; "
            "Primärmetrik MRR@10, alle Werte auf dem vollen Ranking, Cosine-Similarity (BM25: BM25Okapi). "
            "Modelle mit nativer Eingabelänge. Fehlende Läufe: " + ("; ".join(notes) if notes else "keine") + "\n")
    (O / "table_main.md").write_text("# Hauptergebnisse\n\n" + md(tm) + foot)

    # significance
    cj = H / f"comparison_{DS}.json"
    present = {m for m in cm.EMBEDDERS if m in ranks}
    stale = True
    if cj.exists() and not a.recompute:
        seen = {r["b"] for r in json.loads(cj.read_text())["tests"] if r["family"].startswith("confirmatory")}
        stale = seen != present
    if stale:
        print("running compare_models.py ...")
        sys.argv = ["compare_models.py", "--harness-dir", str(H), "--out", str(cj)]
        cm.main()
    res = json.loads(cj.read_text())
    meta_c = res["meta"]

    def srow(r: dict) -> dict:
        return {"Vergleich": f"{LABEL[r['b']]} − {LABEL[r['a']]}", "Metrik": r["metric"],
                "Differenz": de(r["diff_b_minus_a"], 3, True),
                "95%-CI": f"[{de(r['ci95_cluster_bootstrap'][0], 3, True)}; {de(r['ci95_cluster_bootstrap'][1], 3, True)}]",
                "p roh": dep(r["p_cluster_signflip"], r["p_is_upper_bound"]),
                "p Holm": dep(r["p_holm"], r["p_is_upper_bound"]),
                "n≠0": r["n_nonzero_clusters"],
                "Hinweis": "nicht aussagekräftig (n≠0 < 20)" if r["not_informative"] else ""}

    tests = res["tests"]
    conf = [srow(r) for r in tests if r["family"].startswith("confirmatory")]
    L = ["# Signifikanz", "",
         f"Differenz = zweiter minus erster Retriever. Test: gepaarter Vorzeichenwechsel auf Cluster-Ebene, zweiseitig, "
         f"B = {meta_c['B_permutation']}, Seed {meta_c['seed']}; CI: 95-%-Cluster-Bootstrap (B = {meta_c['B_bootstrap']}); "
         f"exakt bei n≠0 ≤ {meta_c['exact_if_n_nonzero_le']}; Monte-Carlo-p an der Untergrenze als Schranke (≤). "
         f"Holm in der konfirmatorischen Familie mit Familiengröße 6 (geplant; aktuell {len(conf)} Tests vorhanden). "
         f"Datensatz GermanQuAD, n = {res['n_queries']} Queries, {res['n_clusters']} Cluster.", "",
         "## Konfirmatorisch: Embedder vs. BM25, MRR@10", "", md(pd.DataFrame(conf))]
    fams = sorted({r["family"] for r in tests if r["family"].startswith("explorative")})
    L += ["## Explorativ (Holm nur innerhalb der jeweiligen Familie, keine konfirmatorische Aussage)", ""]
    for fam in fams:
        L += [f"### {fam.replace('explorative: ', '')}", "", md(pd.DataFrame([srow(r) for r in tests if r["family"] == fam]))]
        if "Success" in fam:
            L += ["McNemar (exakt, optimistisch, ignoriert Cluster; deskriptiv), b = nur erster, c = nur zweiter Retriever: " +
                  "; ".join(f"{LABEL[r['b']]}: b={r['mcnemar_b_a_only']}, c={r['mcnemar_c_b_only']}, p={dep(r['mcnemar_p'], False)}"
                            for r in tests if r["family"] == fam), ""]
    L += ["Quelle: " + str(cj.relative_to(CODE)) + "; Fehlende Läufe: " + ("; ".join(notes) if notes else "keine")]
    (O / "table_significance.md").write_text("\n".join(L) + "\n")

    # len512 sensitivity
    L5 = []
    for m in LEN512_MODELS:
        k = f"{m}__len512"
        if m in ranks and k in ranks:
            for metric, fn in cm.METRICS.items():
                dn, d5 = fn(ranks[m]), fn(ranks[k])
                d = d5 - dn
                c = cluster_bootstrap_ci(d, cinv, ncl, sizes)
                L5.append({"Modell": LABEL[m], "Metrik": metric, "nativ": de(dn.mean()), "512": de(d5.mean()),
                           "Differenz (512 − nativ)": de(d.mean(), 3, True),
                           "95%-CI": f"[{de(c[0], 3, True)}; {de(c[1], 3, True)}]"})
        else:
            notes.append(f"len512 {m}: nicht vollständig vorhanden (übersprungen)")
    body = md(pd.DataFrame(L5)) if L5 else "Keine Paare nativ/512 vorhanden.\n"
    (O / "table_len512.md").write_text(
        "# Sensitivität Eingabelänge (explorativ, ohne Holm)\n\nnativ vs. max_seq_length = 512 (nur Passagen); "
        "CI: 95-%-Cluster-Bootstrap der Differenz (B = 10000, Seed 42); keine p-Werte.\n\n" + body +
        "\nFehlende Läufe: " + ("; ".join(n_ for n_ in notes if n_.startswith("len512")) or "keine") + "\n")

    # figure: sorted dot plot, black-and-white friendly (distinct markers, direct labels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    order = sorted(ci, key=lambda m: cm.rr10(ranks[m]).mean())
    mk = ["o", "s", "^", "D", "v", "P", "X"]
    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.6 * len(order)))
    for i, m in enumerate(order):
        v = cm.rr10(ranks[m]).mean()
        ax.errorbar(v, i, xerr=[[v - ci[m][0]], [ci[m][1] - v]], fmt=mk[i % len(mk)], color="black", mfc="white" if m == "bm25" else "black",
                    capsize=4, markersize=7)
        ax.annotate(de(v), (ci[m][1], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    lo = min(c[0] for c in ci.values())
    ax.set_xlim(max(0, lo - 0.03), min(1.0, max(c[1] for c in ci.values()) + 0.05))
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))  # locale-independent decimal comma
    ax.set_xlabel("MRR@10")
    ax.set_title(f"GermanQuAD: MRR@10 je Retriever (n = {n} Queries)")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.5, 0.005, f"Balken: 95-%-Cluster-Bootstrap-CI (Cluster = Gold-Passage, B = 10000, Seed {SEED}); x-Achse gekürzt",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(O / f"fig_mrr10_ci.{ext}", dpi=200)
    print("written to", O)
    for x in notes:
        print("NOTE:", x)


if __name__ == "__main__":
    main()
