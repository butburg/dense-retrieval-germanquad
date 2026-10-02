"""Build the Chapter-4 result tables and the central figure from the harness outputs (GermanQuAD).

Usage: python make_results.py [--harness-dir output/harness] [--out-dir output/harness/results] [--recompute]
Eingabe: <harness>/<model>/metrics_germanquad.json + per_query_germanquad.csv der neun Embedder (auch <model>__len512),
Ränge von BM25 und BM25-de (Pfade wie in compare_models.py), <harness>/comparison_germanquad.json (wird von
compare_models.py neu erzeugt, wenn sie fehlt oder --recompute gesetzt ist; Laufzeit etwa 1 min).
Ausgabe (out-dir): table_main.md/.csv (deskriptiv, alle Kennzahlen), table_significance.md (55 Tests in vier Familien),
table_len512.md, fig_mrr10_ci.pdf/.png (Abbildung 1), fig_pair_matrix.png (MRR@10-Differenzen der 36 Modellpaare),
fig_test_flow.png (Testablauf).
Zellwerte stammen aus den Eingaben; das CI ist das 95-%-Cluster-Bootstrap-CI (B=10000, Seed 42) des Mittelwerts.
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
         "openai-3-small": "text-embedding-3-small", "openai-3-large": "text-embedding-3-large",
         "bm25_de": "BM25-de", "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B",
         "arctic-embed-l-v2": "snowflake-arctic-embed-l-v2.0", "jina-v2-base-de": "jina-embeddings-v2-base-de"}
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


def draw_pair_matrix(order: list[str], mrr: dict[str, float], not_sig: set[frozenset], path: Path) -> None:
    """Dreiecksmatrix der MRR@10-Differenzen (Spalte minus Zeile) der neun Modelle in Graustufen.

    Args:
        order: Modellschlüssel nach MRR@10 absteigend (links stehende Spalten sind besser als die Zeilen darunter).
        mrr: MRR@10 je Modell; die Differenz der Mittelwerte ist der Mittelwert der Differenzen je Frage.
        not_sig: Paare mit p Holm >= 0,05 in F1; ihre Zellen tragen ein „°“.
        path: Ausgabe-PNG.
    """
    import matplotlib.pyplot as plt
    n = len(order)
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i):
            D[i, j] = mrr[order[j]] - mrr[order[i]]
    D = D[1:, :-1]  # first row and last column are empty
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    im = ax.imshow(np.ma.masked_invalid(D), cmap="Greys", vmin=0, vmax=np.nanmax(D) * 1.6)
    for i in range(n - 1):
        for j in range(i + 1):
            ax.text(j, i, de(D[i, j]) + ("°" if frozenset((order[i + 1], order[j])) in not_sig else ""),
                    ha="center", va="center", fontsize=8)
    lab = [LABEL[k] for k in order]
    ax.set_xticks(range(n - 1), lab[:-1], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n - 1), lab[1:], fontsize=9)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("MRR@10 Spalte minus Zeile", fontsize=9)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    fig.text(0.5, 0.01, "° = nicht trennbar (p Holm ≥ 0,05, Familie F1 mit 36 Paaren)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_flow(path: Path) -> None:
    """Graustufen-Ablaufschema des gepaarten Tests mit Legende."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    steps = ["2.204 Fragen,\n474 Passagen", "11 Retriever:\nRangliste je Frage", "Rang der\nGold-Passage",
             "MRR@10 je Frage", "Differenz je Frage:\nzweiter minus\nerster Retriever",
             "Summe je Cluster,\nVorzeichen zufällig\n(B = 100.000)", "p-Wert und\nCluster-Bootstrap-\nIntervall",
             "Holm-Korrektur\nje Familie", "Urteil:\nH0 verworfen\noder nicht"]
    pos = [(1.15 + 2.2 * i, 3.1) for i in range(5)] + [(1.15 + 2.2 * i, 1.0) for i in range(4, 0, -1)]
    w, h = 1.9, 1.15
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 11.2); ax.set_ylim(-0.7, 3.9); ax.axis("off")
    for (x, y), t in zip(pos, steps):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec="black", lw=1.0, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=10.5, zorder=3)
    kw = dict(arrowstyle="-|>", mutation_scale=14, lw=1.2, color="black", shrinkA=0, shrinkB=0)
    for i in range(8):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 != y1:
            ax.annotate("", (x1, y1 + h / 2), (x0, y0 - h / 2), arrowprops=kw)
        else:
            s_ = 1 if x1 > x0 else -1
            ax.annotate("", (x1 - s_ * w / 2, y1), (x0 + s_ * w / 2, y0), arrowprops=kw)
    ax.text(0.15, -0.3, "Legende: Kasten = Arbeitsschritt, Pfeil = Reihenfolge. Familien (Holm je Familie): F1 Modellpaare (36 Tests),\n"
            "F2 Modell gegen BM25 (9), F3 Modell gegen BM25-de (9), F4 BM25-de gegen BM25 (1, ohne Korrektur); zusammen 55 Tests.",
            fontsize=11, va="center")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    for m in cm.MODELS + [f"{x}__len512" for x in LEN512_MODELS]:
        f = H / m / f"per_query_{DS}.csv"
        if f.exists() and (H / m / f"metrics_{DS}.json").exists():
            ranks[m] = cm.load_ranks(f, qids)
            meta[m] = json.loads((H / m / f"metrics_{DS}.json").read_text())
        else:
            if "__len512" not in m:
                notes.append(f"{m}: nicht vorhanden (übersprungen)")
    bm, bmde = cm.BM25_CSV, cm.BM25_DE_CSV
    if not bm.exists():
        sys.exit("no BM25 per-query file found")
    ranks["bm25"] = cm.load_ranks(bm, qids)
    if bmde.exists():
        ranks["bm25_de"] = cm.load_ranks(bmde, qids)

    # main table: embedder values from metrics json (cross-checked against ranks), BM25 from ranks
    rows, ci = [], {}
    for m in ["bm25", "bm25_de"] + [m for m in cm.MODELS if m in ranks]:
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
        rev = (revision(meta[m]["experiment"]) if m in meta else
               f"rank_bm25 (k1=1.5, b=0.75), {bm.relative_to(CODE)}" if m == "bm25" else
               f"rank_bm25 mit Stoppwörtern und Stemming, {bmde.relative_to(CODE)}")
        rows.append({"Retriever": LABEL[m], **v, "Modellrevision / API-Modell": rev})
    t = pd.DataFrame(rows)
    t.to_csv(O / "table_main.csv", index=False)
    tm = t.copy()
    for c in ["MRR@10", "Success@1", "Success@5", "Success@10", "MRR@5"]:
        tm[c] = tm[c].map(de)
    n = len(qids)
    foot = (f"\nGermanQuAD (Test), n = {n} Queries, {ncl} Cluster (Gold-Passagen), Corpus {len(docs)} Passagen; "
            "Primärmetrik MRR@10, alle Werte auf dem vollen Ranking, Cosine-Similarity (BM25: BM25Okapi). "
            "Modelle mit nativer Eingabelänge; Kennzahlen deskriptiv, getestet wird nur MRR@10 (table_significance.md). Fehlende Läufe: " + ("; ".join(notes) if notes else "keine") + "\n")
    (O / "table_main.md").write_text("# Hauptergebnisse\n\n" + md(tm) + foot)

    # significance
    cj = H / f"comparison_{DS}.json"
    if a.recompute or not cj.exists():
        print("running compare_models.py ...")
        sys.argv = ["compare_models.py", "--harness-dir", str(H), "--out", str(cj)]
        cm.main()
    res = json.loads(cj.read_text())
    meta_c = res["meta"]
    tests = res["tests"]

    def srow(r: dict) -> dict:
        return {"Vergleich": f"{LABEL[r['b']]} − {LABEL[r['a']]}", "Differenz": de(r["diff_b_minus_a"], 3, True),
                "95%-CI": f"[{de(r['ci95_cluster_bootstrap'][0], 3, True)}; {de(r['ci95_cluster_bootstrap'][1], 3, True)}]",
                "p roh": dep(r["p_cluster_signflip"], r["p_is_upper_bound"]),
                "p Holm": dep(r["p_holm"], r["p_is_upper_bound"]),
                "n≠0": r["n_nonzero_clusters"],
                "Hinweis": "nicht aussagekräftig (n≠0 < 20)" if r["not_informative"] else ""}

    def fam(prefix: str) -> list[dict]:
        return [r for r in tests if r["family"].startswith(prefix)]

    f2 = {r["b"]: r for r in fam("F2")}
    f3 = {r["b"]: r for r in fam("F3")}
    core = [{"Modell": LABEL[m], "MRR@10": de(f2[m]["mean_b"]),
             "p Holm gegen BM25": dep(f2[m]["p_holm"], f2[m]["p_is_upper_bound"]),
             "p Holm gegen BM25-de": dep(f3[m]["p_holm"], f3[m]["p_is_upper_bound"])}
            for m in sorted(f2, key=lambda m: -f2[m]["mean_b"])]
    L = ["# Signifikanz", "",
         f"Alle Tests auf der Primärmetrik MRR@10; Differenz = zweiter minus erster Retriever. Test: gepaarter Vorzeichenwechsel auf "
         f"Cluster-Ebene, zweiseitig, B = {meta_c['B_permutation']}, Seed {meta_c['seed']}; CI: 95-%-Cluster-Bootstrap "
         f"(B = {meta_c['B_bootstrap']}); exakt bei n≠0 ≤ {meta_c['exact_if_n_nonzero_le']}; Monte-Carlo-p an der Untergrenze als "
         f"Schranke (≤). Holm je Familie mit der tatsächlichen Testzahl als Familiengröße. Datensatz GermanQuAD, "
         f"n = {res['n_queries']} Queries, {res['n_clusters']} Cluster. Sekundärmetriken stehen deskriptiv in table_main.md.", "",
         f"## Kernübersicht: MRR@10 je Modell mit p Holm gegen die Baselines (F2 und F3, je {len(f2)} Tests)", "",
         md(pd.DataFrame(core)), "BM25-de gegen BM25 (F4, ohne Korrektur): " +
         "; ".join(f"Differenz {de(r['diff_b_minus_a'], 3, True)}, p {dep(r['p_cluster_signflip'], r['p_is_upper_bound'])}" for r in fam("F4")), ""]
    for prefix in ("F1", "F2", "F3", "F4"):
        rows_ = fam(prefix)
        L += [f"## {rows_[0]['family']} ({len(rows_)} Tests, Holm über {len(rows_)})" if len(rows_) > 1 else f"## {rows_[0]['family']} (1 Test, ohne Korrektur)",
              "", md(pd.DataFrame([srow(r) for r in rows_]))]
    L += ["Quelle: " + cj.relative_to(CODE.parent).as_posix() + "; Fehlende Läufe: " + ("; ".join(notes) if notes else "keine")]
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
        "# Sensitivität Eingabelänge (deskriptiv, ohne Test)\n\nnativ vs. max_seq_length = 512 (nur Passagen); "
        "CI: 95-%-Cluster-Bootstrap der Differenz (B = 10000, Seed 42); keine p-Werte.\n\n" + body +
        "\nFehlende Läufe: " + ("; ".join(n_ for n_ in notes if n_.startswith("len512")) or "keine") + "\n")

    # figure: sorted dot plot, black-and-white friendly (distinct markers, direct labels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    order = sorted(ci, key=lambda m: cm.rr10(ranks[m]).mean())  # Abbildung 1: alle elf Retriever
    mk = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h"]
    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 0.5 * len(order)))
    for i, m in enumerate(order):
        v = cm.rr10(ranks[m]).mean()
        base = m in ("bm25", "bm25_de")
        ax.errorbar(v, i, xerr=[[v - ci[m][0]], [ci[m][1] - v]], fmt=mk[i % len(mk)], color="black",
                    mfc="white" if base else "black", capsize=4, markersize=7)
        ax.annotate(de(v), (ci[m][1], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    from matplotlib.lines import Line2D
    mk_of = {m: mk[i % len(mk)] for i, m in enumerate(order)}  # legend uses the markers actually plotted
    ax.legend(handles=[Line2D([], [], marker=mk_of["bm25"], color="black", mfc="white", ls="", label="BM25 (offen)"),
                       Line2D([], [], marker=mk_of["bm25_de"], color="black", mfc="white", ls="", label="BM25-de (offen)"),
                       Line2D([], [], marker="o", color="black", mfc="black", ls="", label="Embedding-Modell (gefüllt)")],
              loc="center left", bbox_to_anchor=(0, 0.64), fontsize=8, frameon=True)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    lo = min(c[0] for c in ci.values())
    ax.set_xlim(max(0, lo - 0.03), min(1.0, max(c[1] for c in ci.values()) + 0.05))
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))  # locale-independent decimal comma
    ax.set_xlabel("MRR@10")
    ax.set_title(f"GermanQuAD: MRR@10 je Retriever (n = {n:,} Queries)".replace(",", "."))
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.5, 0.005, f"Balken: 95-%-Cluster-Bootstrap-CI (Cluster = Gold-Passage, B = 10000, Seed {SEED}); x-Achse gekürzt",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(O / f"fig_mrr10_ci.{ext}", dpi=200)
    plt.close(fig)
    nine = [m for m in reversed(order) if m in cm.MODELS]
    not_sig = {frozenset((r["a"], r["b"])) for r in tests if r["family"].startswith("F1") and r["p_holm"] >= 0.05}
    draw_pair_matrix(nine, {m: cm.rr10(ranks[m]).mean() for m in nine}, not_sig, O / "fig_pair_matrix.png")
    draw_flow(O / "fig_test_flow.png")
    print("written to", O)
    for x in notes:
        print("NOTE:", x)


if __name__ == "__main__":
    main()
