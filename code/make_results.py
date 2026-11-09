"""Build the Chapter-4 result tables and the central figure from the harness outputs (GermanQuAD).

Usage: python make_results.py [--harness-dir output/harness] [--out-dir output/harness/results] [--recompute]
Eingabe: <harness>/<model>/metrics_germanquad.json + per_query_germanquad.csv der neun Embedder (auch <model>__len512),
Ränge und gespeicherte Metriken (bm25_results.json) von BM25 und BM25-de (Pfade wie in compare_models.py),
<harness>/comparison_germanquad.json (wird von compare_models.py neu erzeugt, wenn sie fehlt oder --recompute gesetzt
ist; Laufzeit etwa 1 min).
Ausgabe (out-dir): table_main.md/.csv (deskriptiv, alle Kennzahlen), table_significance.md (55 Tests in vier Familien),
table_len512.md, fig_mrr10_ci.pdf/.png (Abbildung 1), fig_pair_matrix.png (MRR@10-Differenzen der 36 Modellpaare),
fig_test_flow.png (Testablauf).
Zellwerte stammen aus den Eingaben; das CI ist das 95-%-Cluster-Bootstrap-CI (B und Seed aus pylib/significance.py)
des Mittelwerts.

Role in the experiment (English summary): final reporting step. Upstream: embed_eval.py (per-model metrics and
per-query ranks), scripts/run_bm25.py and bm25_german.py (BM25 ranks), compare_models.py (tests; imported here as
``cm`` for MODELS, METRICS, load_ranks and BM25 paths; per-query metrics come from pylib/retrieval_metrics.py). No new
test is computed here; the significance tables only re-format comparison_germanquad.json. Descriptive values of all
eleven retrievers are taken from the stored metrics files (metrics_*.json, bm25_results.json) and cross-checked
against the per-query ranks. Existing files in --out-dir are overwritten. Numbers use German formatting (decimal
comma), because the tables go into the manuscript.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))
import compare_models as cm  # noqa: E402
from embedders import REGISTRY  # noqa: E402
from local_dataset_io import load_jsonl  # noqa: E402
from retrieval_metrics import build_gold, reciprocal_rank_at_k, success_at_k  # noqa: E402
from significance import B_BOOT, B_PERM, SEED, cluster_bootstrap_ci, make_clusters  # noqa: E402
from run_bm25 import BM25_B, BM25_K1  # noqa: E402  (BM25 parameters have one source: scripts/run_bm25.py)

DS = cm.DS  # dataset name, shared with compare_models.py
# Retriever key -> display name used in tables and figures; embedder names come from the registry (display_name).
LABEL = {"bm25": "BM25", "bm25_de": "BM25-de", **{key: REGISTRY[key].display_name for key in cm.MODELS}}
LEN512_MODELS = cm.LEN512_MODELS  # models that also have a <model>__len512 run (one source: compare_models.py)


def de(x: float, nd: int = 3, sign: bool = False) -> str:
    """German number format (decimal comma).

    Args:
        x: Value to format, e.g. ``0.8123``.
        nd: Number of decimals.
        sign: If True, always show the sign (``+0,012``), used for differences.

    Returns:
        String such as ``"0,812"`` or ``"+0,012"``.
    """
    return f"{x:+.{nd}f}".replace(".", ",") if sign else f"{x:.{nd}f}".replace(".", ",")


def dep(p: float, floor: bool) -> str:
    """German p-value format; Monte-Carlo p at the floor is reported as upper bound (value = bound).

    Args:
        p: Raw or Holm-adjusted p-value.
        floor: True if ``p`` is a Monte-Carlo p at its minimum 1/(B+1), i.e. only an upper bound
            (``p_is_upper_bound`` in comparison_germanquad.json); prefixed with "≤".

    Returns:
        Decimal-comma string, e.g. ``"0,0312"``, ``"≤ 1,0e-05"`` or ``"2,3e-07"`` (scientific for very small values).
    """
    if p < 0.0001 and not floor:  # exact value in scientific notation (no "< 0,0001")
        return f"{p:.1e}".replace(".", ",")
    # bounds below 1e-3 in scientific notation, so that e.g. 15/(B+1) = 1.5e-04 is not rounded to 0,0001
    s = f"{p:.4f}" if p >= (0.001 if floor else 0.0001) else f"{p:.1e}"
    return ("≤ " if floor else "") + s.replace(".", ",")


def md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown pipe table.

    Args:
        df: Table to render; cell values are converted with ``str``.

    Returns:
        Markdown string (header, separator, one line per row, trailing newline).
    """
    # Markdown header: column names joined by " | ", then one "---|" per column as separator row.
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    # Outer generator: one Markdown line per table row (itertuples yields each row as a tuple);
    # inner generator: the cell values of that row converted to text and joined by " | ".
    return h + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def revision(exp: dict) -> str:
    """Model revision (HF commit) or API model name with retrieval date.

    Args:
        exp: ``experiment`` block of a ``metrics_germanquad.json`` file.

    Returns:
        ``"<model_name>@<first 12 chars of revision>"`` for local models, or
        ``"<returned API model> (API, abgerufen <date>)"`` for OpenAI models.
    """
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
    # D[i, j] (j < i): MRR@10 of the better model order[j] minus that of order[i]; the upper triangle stays empty.
    D = np.full((n, n), np.nan)
    # Fill the lower triangle: for every pair (row i, column j < i) store the MRR@10 difference
    # "better model order[j] minus model order[i]" (order is sorted best first, so all differences are >= 0).
    for i in range(n):
        # Inner loop over the columns left of the diagonal (j < i), i.e. the models better than order[i].
        for j in range(i):
            D[i, j] = mrr[order[j]] - mrr[order[i]]
    D = D[1:, :-1]  # first row and last column are empty
    # Annotate each cell with the difference; "°" marks pairs that are not separable after Holm correction.
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    im = ax.imshow(np.ma.masked_invalid(D), cmap="Greys", vmin=0, vmax=np.nanmax(D) * 1.6)
    # After cutting D, row i compares order[i + 1] with the columns j <= i; write the difference into each
    # cell of this lower triangle as text ("°" appended if the pair is not significant after Holm).
    for i in range(n - 1):
        # Inner loop over the columns of this row: one text label per cell of the triangle.
        for j in range(i + 1):
            # frozenset((a, b)) is an order-free pair, so the lookup in not_sig works for (a, b) and (b, a).
            ax.text(j, i, de(D[i, j]) + ("°" if frozenset((order[i + 1], order[j])) in not_sig else ""),
                    ha="center", va="center", fontsize=8)
    # List comprehension: display names of the models in plotting order (used as axis tick labels).
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
    """Graustufen-Ablaufschema des gepaarten Tests mit Legende.

    Args:
        path: Output PNG path. The step texts and counts (2.204 queries, 474 passages, 11 retrievers) are hard-coded
            in the function and must be updated manually if the design changes; the permutation count B is formatted from ``significance.B_PERM``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    permutations_text = f"{B_PERM:,}".replace(",", ".")  # German thousands separator, e.g. 100.000
    # Boxes in reading order: first row left to right, second row right to left (snake layout).
    steps = ["2.204 Fragen,\n474 Passagen", "11 Retriever:\nRangliste je Frage", "Rang der\nGold-Passage",
             "MRR@10 je Frage", "Differenz je Frage:\nzweiter minus\nerster Retriever",
             f"Summe je Cluster,\nVorzeichen zufällig\n(B = {permutations_text})", "p-Wert und\nCluster-Bootstrap-\nIntervall",
             "Holm-Korrektur\nje Familie", "Urteil:\nH0 verworfen\noder nicht"]
    # pos: box centres; w, h: box width and height
    # Two list comprehensions for the box centres: first row x = 1.15, 3.35, ... (5 boxes, y = 3.1),
    # second row from right to left (i = 4, 3, 2, 1; y = 1.0) so the arrows form a snake.
    pos = [(1.15 + 2.2 * i, 3.1) for i in range(5)] + [(1.15 + 2.2 * i, 1.0) for i in range(4, 0, -1)]
    w, h = 1.9, 1.15
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 11.2); ax.set_ylim(-0.7, 3.9); ax.axis("off")
    # For every (box centre, step text): draw a rounded box and put the text in its centre.
    for (x, y), t in zip(pos, steps):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec="black", lw=1.0, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=10.5, zorder=3)
    # Arrows connect consecutive boxes: vertical at the row change, horizontal otherwise (direction by sign s_).
    kw = dict(arrowstyle="-|>", mutation_scale=14, lw=1.2, color="black", shrinkA=0, shrinkB=0)
    # For every pair of consecutive boxes (9 boxes -> 8 arrows): draw an arrow from box i to box i + 1.
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
    """Build all result tables and figures from the harness outputs.

    Writes ``table_main.csv/.md``, ``table_significance.md``, ``table_len512.md``, ``fig_mrr10_ci.pdf/.png``,
    ``fig_pair_matrix.png`` and ``fig_test_flow.png`` into ``--out-dir``. If ``comparison_germanquad.json`` is
    missing (or ``--recompute`` is set), compare_models.py is run first via ``cm.main()``.
    """
    # --- Parse arguments and load dataset (cluster structure for the bootstrap CIs) ---
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness-dir", type=Path, default=CODE / "output" / "harness")
    ap.add_argument("--out-dir", type=Path, default=CODE / "output" / "harness" / "results")
    ap.add_argument("--recompute", action="store_true", help="force compare_models.py rerun")
    a = ap.parse_args()
    H, O = a.harness_dir, a.out_dir  # H: harness directory (input); O: output directory
    O.mkdir(parents=True, exist_ok=True)

    dd = CODE / "output" / DS
    # Generator over the three roles: load each normalized JSONL file; unpacking gives the three record lists.
    queries, docs, qrels = (load_jsonl(dd / f"{x}.normalized.jsonl") for x in ("queries", "docs", "qrels"))
    # List comprehension: query IDs as strings in file order.
    qids = [str(q["query_id"]) for q in queries]
    # Set comprehension: IDs of all corpus passages; build_gold keeps only qrels that point to them.
    gold = build_gold(qrels, {str(d["doc_id"]) for d in docs}, 1)
    # Cluster = first sorted gold passage per query (same definition as compare_models.load_queries_clusters);
    # cinv: cluster index per query, ncl: number of clusters, sizes: queries per cluster.
    # List comprehension: cluster label per query = alphabetically first gold passage ID.
    cinv, ncl, sizes = make_clusters([sorted(gold[q])[0] for q in qids])

    # --- Load per-query ranks and metrics of all available runs ---
    # ranks: retriever key -> rank of first relevant passage per query (0 = none); meta: key -> parsed metrics JSON;
    # notes: messages about skipped runs, printed at the end and written into the table footers.
    ranks, meta, notes = {}, {}, []
    # For every embedder run (nine native runs plus the <model>__len512 runs of the three long-context models,
    # built by the inner list comprehension): load ranks and metrics if both files exist, else record a note.
    for m in cm.MODELS + [f"{x}__len512" for x in LEN512_MODELS]:
        f = H / m / f"per_query_{DS}.csv"
        if f.exists() and (H / m / f"metrics_{DS}.json").exists():
            ranks[m] = cm.load_ranks(f, qids)
            meta[m] = json.loads((H / m / f"metrics_{DS}.json").read_text())
        else:
            if "__len512" not in m:
                notes.append(f"{m}: nicht vorhanden (übersprungen)")
    # The four BM25 inputs (ranks and stored metrics of both runs) with the command that creates each of them.
    bm25_inputs = [(cm.BM25_CSV, "python scripts/run_bm25.py"), (cm.BM25_JSON, "python scripts/run_bm25.py"),
                   (cm.BM25_DE_CSV, "python bm25_german.py"), (cm.BM25_DE_JSON, "python bm25_german.py")]
    for path, command in bm25_inputs:
        if not path.exists():
            sys.exit(f"BM25-Eingabe fehlt: {path} existiert nicht. Erzeugen mit `{command}` (im Ordner code/), "
                     "danach make_results.py erneut starten.")
    ranks["bm25"], ranks["bm25_de"] = cm.load_ranks(cm.BM25_CSV, qids), cm.load_ranks(cm.BM25_DE_CSV, qids)
    # stored_metrics: retriever key -> metric rows (one per k); all eleven retrievers are checked against the ranks.
    stored_metrics = {m: meta[m]["metrics"] for m in meta} | {
        "bm25": json.loads(cm.BM25_JSON.read_text())["metrics"],
        "bm25_de": json.loads(cm.BM25_DE_JSON.read_text())["metrics"]}

    # --- Main table (table_main): stored values of all retrievers, cross-checked against the ranks ---
    # rows: table rows; ci: retriever key -> 95 % cluster-bootstrap CI of MRR@10 (for Figure 1)
    rows, ci = [], {}
    # For every retriever (BM25 and BM25-de first, then the embedders that were actually loaded, via the inner
    # comprehension): build one table row with MRR@10, Success@k, MRR@5, bootstrap CI and provenance.
    for m in ["bm25", "bm25_de"] + [m for m in cm.MODELS if m in ranks]:
        r = ranks[m]  # r: rank of first relevant passage per query (0 = none)
        # v: descriptive metrics from the ranks (MRR@5 is not part of cm.METRICS); each is the mean over all queries
        # of a per-query value from the rank array r (e.g. r = [1, 7, 0]: Success@5 per query = [1, 0, 0], mean 0.33).
        v = {"MRR@10": reciprocal_rank_at_k(r, 10).mean(), "Success@1": success_at_k(r, 1).mean(),
             "Success@5": success_at_k(r, 5).mean(), "Success@10": success_at_k(r, 10).mean(),
             "MRR@5": reciprocal_rank_at_k(r, 5).mean()}
        mm = {x["k"]: x for x in stored_metrics[m]}  # stored metric row per cutoff k (1, 5, 10)
        stored = {"MRR@10": mm[10]["mrr_at_k"], "Success@1": mm[1]["success_at_k"], "Success@5": mm[5]["success_at_k"],
                  "Success@10": mm[10]["success_at_k"], "MRR@5": mm[5]["mrr_at_k"]}
        # Each stored value must equal the one recomputed from the ranks (tolerance cm.CONSISTENCY_TOL).
        assert all(abs(stored[k] - v[k]) < cm.CONSISTENCY_TOL for k in v), f"{m}: metrics json differs from per-query ranks"
        ci[m] = cluster_bootstrap_ci(reciprocal_rank_at_k(r, 10), cinv, ncl, sizes)
        # rev: provenance column (model revision / API model; for BM25 the parameters and source file)
        rev = (revision(meta[m]["experiment"]) if m in meta else
               f"rank_bm25 (k1={BM25_K1}, b={BM25_B}), {cm.BM25_CSV.relative_to(CODE)}" if m == "bm25" else
               f"rank_bm25 mit Stoppwörtern und Stemming, {cm.BM25_DE_CSV.relative_to(CODE)}")
        rows.append({"Retriever": LABEL[m], **stored, "Modellrevision / API-Modell": rev})
    t = pd.DataFrame(rows)  # t: numeric table (CSV); tm: German-formatted copy for Markdown
    t.to_csv(O / "table_main.csv", index=False)
    tm = t.copy()
    # German number format per metric column for the Markdown table (e.g. 0.8123 -> "0,812").
    for c in ["MRR@10", "Success@1", "Success@5", "Success@10", "MRR@5"]:
        tm[c] = tm[c].map(de)
    n = len(qids)
    foot = (f"\nGermanQuAD (Test), n = {n} Queries, {ncl} Cluster (Gold-Passagen), Corpus {len(docs)} Passagen; "
            "Primärmetrik MRR@10, alle Werte auf dem vollen Ranking, Cosine-Similarity (BM25: BM25Okapi). "
            "Modelle mit nativer Eingabelänge; Kennzahlen deskriptiv, getestet wird nur MRR@10 (table_significance.md). Fehlende Läufe: " + ("; ".join(notes) if notes else "keine") + "\n")
    (O / "table_main.md").write_text("# Hauptergebnisse\n\n" + md(tm) + foot)

    # --- Significance tables (table_significance): re-format the stored test results, no recomputation ---
    cj = H / f"comparison_{DS}.json"
    if a.recompute or not cj.exists():
        print("running compare_models.py ...")
        sys.argv = ["compare_models.py", "--harness-dir", str(H), "--out", str(cj)]
        cm.main()
    res = json.loads(cj.read_text())
    meta_c = res["meta"]
    tests = res["tests"]

    def srow(r: dict) -> dict:
        """Format one test result (``tests`` entry of the JSON) as a table row."""
        return {"Vergleich": f"{LABEL[r['b']]} − {LABEL[r['a']]}", "Differenz": de(r["diff_b_minus_a"], 3, True),
                "95%-CI": f"[{de(r['ci95_cluster_bootstrap'][0], 3, True)}; {de(r['ci95_cluster_bootstrap'][1], 3, True)}]",
                "p roh": dep(r["p_cluster_signflip"], r["p_is_upper_bound"]),
                "p Holm": dep(r["p_holm"], r["p_is_upper_bound"]),
                "n≠0": r["n_nonzero_clusters"],
                "Hinweis": "nicht aussagekräftig (n≠0 < 20)" if r["not_informative"] else ""}

    def fam(prefix: str) -> list[dict]:
        """Return the tests whose family label starts with ``prefix`` (``"F1"`` to ``"F4"``)."""
        # List comprehension: all test rows whose family label starts with the prefix (e.g. "F2").
        return [r for r in tests if r["family"].startswith(prefix)]

    # f2/f3: model key -> its test against BM25 (F2) / BM25-de (F3); core: compact overview sorted by MRR@10
    # Dict comprehension: compared model key -> its F2 test row.
    f2 = {r["b"]: r for r in fam("F2")}
    # Dict comprehension: compared model key -> its F3 test row.
    f3 = {r["b"]: r for r in fam("F3")}
    # List comprehension: one overview row per model, ordered by descending MRR@10 (sorted with key = -MRR@10),
    # with the Holm-adjusted p against BM25 and BM25-de.
    core = [{"Modell": LABEL[m], "MRR@10": de(f2[m]["mean_b"]),
             "p Holm gegen BM25": dep(f2[m]["p_holm"], f2[m]["p_is_upper_bound"]),
             "p Holm gegen BM25-de": dep(f3[m]["p_holm"], f3[m]["p_is_upper_bound"])}
            for m in sorted(f2, key=lambda m: -f2[m]["mean_b"])]
    # L: Markdown lines of table_significance.md (overview first, then one section per family)
    L = ["# Signifikanz", "",
         f"Alle Tests auf der Primärmetrik MRR@10; Differenz = zweiter minus erster Retriever. Test: gepaarter Vorzeichenwechsel auf "
         f"Cluster-Ebene, zweiseitig, B = {meta_c['B_permutation']}, Seed {meta_c['seed']}; CI: 95-%-Cluster-Bootstrap "
         f"(B = {meta_c['B_bootstrap']}); exakt bei n≠0 ≤ {meta_c['exact_if_n_nonzero_le']}; Monte-Carlo-p an der Untergrenze als "
         f"Schranke (≤). Holm je Familie mit der tatsächlichen Testzahl als Familiengröße. Datensatz GermanQuAD, "
         f"n = {res['n_queries']} Queries, {res['n_clusters']} Cluster. Sekundärmetriken stehen deskriptiv in table_main.md.", "",
         f"## Kernübersicht: MRR@10 je Modell mit p Holm gegen die Baselines (F2 und F3, je {len(f2)} Tests)", "",
         md(pd.DataFrame(core)), "BM25-de gegen BM25 (F4, ohne Korrektur): " +
         # Generator over the F4 rows (one test): formats difference and p, joined by "; ".
         "; ".join(f"Differenz {de(r['diff_b_minus_a'], 3, True)}, p {dep(r['p_cluster_signflip'], r['p_is_upper_bound'])}" for r in fam("F4")), ""]
    # For every family F1..F4: add a heading with the number of tests and the formatted table of its rows.
    for prefix in ("F1", "F2", "F3", "F4"):
        rows_ = fam(prefix)
        L += [f"## {rows_[0]['family']} ({len(rows_)} Tests, Holm über {len(rows_)})" if len(rows_) > 1 else f"## {rows_[0]['family']} (1 Test, ohne Korrektur)",
              # List comprehension: format each test row of the family with srow (difference, CI, p values, hint).
              "", md(pd.DataFrame([srow(r) for r in rows_]))]
    L += ["Quelle: " + cj.relative_to(CODE.parent).as_posix() + "; Fehlende Läufe: " + ("; ".join(notes) if notes else "keine")]
    (O / "table_significance.md").write_text("\n".join(L) + "\n")

    # --- Input-length sensitivity (table_len512): native max_seq_length vs. 512, descriptive, CI only (no test) ---
    # L5: table rows; dn/d5: per-query metric values native/512; d: per-query difference (512 minus native)
    L5 = []
    # For every long-context model: compare its native-length run with the 512-token run, if both exist.
    for m in LEN512_MODELS:
        k = f"{m}__len512"
        if m in ranks and k in ranks:
            # For every per-query metric function (MRR@10, Success@1/5/10): evaluate both runs on all queries,
            # take the per-query difference and bootstrap its CI (descriptive, no test).
            for metric, fn in cm.METRICS.items():
                # dn, d5: arrays with one metric value per query for the native / 512 run.
                dn, d5 = fn(ranks[m]), fn(ranks[k])
                # Element-wise difference per query (512 minus native), e.g. [1.0, 0.5] - [0.5, 0.5] = [0.5, 0.0].
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
        f"CI: 95-%-Cluster-Bootstrap der Differenz (B = {B_BOOT}, Seed {SEED}); keine p-Werte.\n\n" + body +
        # Generator over the notes: only those about incomplete len512 runs.
        "\nFehlende Läufe: " + ("; ".join(n_ for n_ in notes if n_.startswith("len512")) or "keine") + "\n")

    # --- Figure 1: MRR@10 with 95 % CI per retriever, sorted dot plot, black-and-white friendly (distinct markers, direct labels) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    # order: retriever keys by ascending MRR@10 (best retriever at the top of the plot); mk: marker shapes
    # sorted(..., key=lambda) sorts the retriever keys by their MRR@10 (mean of the per-query reciprocal ranks),
    # lowest first.
    order = sorted(ci, key=lambda m: reciprocal_rank_at_k(ranks[m], 10).mean())  # Abbildung 1: alle elf Retriever
    mk = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h"]
    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 0.5 * len(order)))
    # For every retriever (i = vertical position, bottom = worst): draw its MRR@10 as a marker with CI bar and value
    # label.
    for i, m in enumerate(order):
        v = reciprocal_rank_at_k(ranks[m], 10).mean()
        base = m in ("bm25", "bm25_de")
        # xerr=[[left], [right]] gives asymmetric bars: v - lower bound to the left, upper bound - v to the right.
        ax.errorbar(v, i, xerr=[[v - ci[m][0]], [ci[m][1] - v]], fmt=mk[i % len(mk)], color="black",
                    mfc="white" if base else "black", capsize=4, markersize=7)
        ax.annotate(de(v), (ci[m][1], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    from matplotlib.lines import Line2D
    # Dict comprehension: retriever key -> marker shape it was plotted with (same cycling as in the loop above).
    mk_of = {m: mk[i % len(mk)] for i, m in enumerate(order)}  # legend uses the markers actually plotted
    ax.legend(handles=[Line2D([], [], marker=mk_of["bm25"], color="black", mfc="white", ls="", label="BM25 (offen)"),
                       Line2D([], [], marker=mk_of["bm25_de"], color="black", mfc="white", ls="", label="BM25-de (offen)"),
                       Line2D([], [], marker="o", color="black", mfc="black", ls="", label="Embedding-Modell (gefüllt)")],
              loc="center left", bbox_to_anchor=(0, 0.64), fontsize=8, frameon=True)
    # List comprehension: display name per retriever in plotting order.
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    # Generator over all CIs: smallest lower bound (left edge of the x-axis).
    lo = min(c[0] for c in ci.values())
    # Generator over all CIs: largest upper bound plus margin (right edge of the x-axis, capped at 1.0).
    ax.set_xlim(max(0, lo - 0.03), min(1.0, max(c[1] for c in ci.values()) + 0.05))
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))  # locale-independent decimal comma
    ax.set_xlabel("MRR@10")
    ax.set_title(f"GermanQuAD: MRR@10 je Retriever (n = {n:,} Queries)".replace(",", "."))
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.5, 0.005, f"Balken: 95-%-Cluster-Bootstrap-CI (Cluster = Gold-Passage, B = {B_BOOT}, Seed {SEED}); x-Achse gekürzt",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    # Save the same figure as vector PDF and as PNG.
    for ext in ("pdf", "png"):
        fig.savefig(O / f"fig_mrr10_ci.{ext}", dpi=200)
    plt.close(fig)
    # --- Pair matrix and flow chart ---
    # nine: the nine embedders by descending MRR@10; not_sig: F1 pairs without a Holm-significant difference
    # List comprehension: reverse the ascending order to best first and keep only the nine embedders (drop BM25
    # variants).
    nine = [m for m in reversed(order) if m in cm.MODELS]
    # Set comprehension: unordered model pairs of family F1 whose Holm-adjusted p is >= 0.05 (difference not
    # separable).
    not_sig = {frozenset((r["a"], r["b"])) for r in tests if r["family"].startswith("F1") and r["p_holm"] >= 0.05}
    # Dict comprehension: MRR@10 per embedder (mean of per-query reciprocal ranks) for the pair matrix.
    draw_pair_matrix(nine, {m: reciprocal_rank_at_k(ranks[m], 10).mean() for m in nine}, not_sig, O / "fig_pair_matrix.png")
    draw_flow(O / "fig_test_flow.png")
    print("written to", O)
    # For every note about a skipped run: print it so missing inputs are visible in the console.
    for x in notes:
        print("NOTE:", x)


if __name__ == "__main__":
    main()
