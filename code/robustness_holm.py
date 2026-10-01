"""Robustheitsrechnung: Holm-Korrektur über alle neun Modelle gegen BM25-de.

Eingabe sind die gespeicherten rohen p-Werte (gepaarter Cluster-Vorzeichenwechsel) der
Vergleiche „Modell minus BM25-de“ aus ``output/harness/comparison_germanquad.json``
(sechs konfirmatorisch geprüfte und drei weitere Modelle, je vier Metriken). Es wird nichts neu kodiert oder simuliert; nur die Familiengröße der Holm-Korrektur
(``pylib.significance.holm``) variiert:

- p_holm_alt: bisherige Familie (unverändert aus den JSON-Dateien),
- Variante A: Holm je Metrik über die neun Modelle,
- Variante B: eine Familie über alle 36 Vergleiche.

Ausgabe: ``output/harness/robustness_holm_nine.json`` und
``output/harness/results/table_robustness_holm.md``. Bestehende Dateien werden nicht verändert.

Aufruf (im Ordner ``code/``): ``python robustness_holm.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pylib"))
from significance import holm  # noqa: E402

HARNESS = HERE / "output" / "harness"
ALPHA = 0.05
METRICS = ["MRR@10", "Success@1", "Success@5", "Success@10"]
NAMES = {"e5-large": "multilingual-e5-large", "openai-3-small": "text-embedding-3-small",
         "openai-3-large": "text-embedding-3-large", "jina-v3": "jina-embeddings-v3",
         "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B",
         "arctic-embed-l-v2": "snowflake-arctic-embed-l-v2.0",
         "jina-v2-base-de": "jina-embeddings-v2-base-de"}
TABLE = {"MRR@10": "Tabelle 4", "Success@1": "D6/D7", "Success@5": "D6/D7", "Success@10": "D6/D7"}


def load_rows(harness: Path = HARNESS) -> list[dict]:
    """Lade die 36 Roh-p-Werte „Modell minus BM25-de“ (9 Modelle x 4 Metriken).

    Args:
        harness: Ordner ``output/harness`` mit ``comparison_germanquad.json``.

    Returns:
        Zeilen mit ``model``, ``metric``, ``diff``, ``p_raw``, ``p_holm_old``, in der
        Reihenfolge Metrik, dann konfirmatorische Modelle, dann Erweiterungsmodelle.

    Raises:
        ValueError: Wenn nicht genau 36 Vergleiche gefunden werden.
    """
    tests = json.load(open(harness / "comparison_germanquad.json", encoding="utf-8"))["tests"]
    sel = [t for t in tests if t["a"] == "bm25_de" and t["family"].startswith("explorative: embedder vs BM25-de")]
    sel += [t for t in tests if t["a"] == "bm25_de" and "new model vs BM25-de" in t["family"]]
    rows = [{"model": NAMES.get(t["b"], t["b"]), "metric": t["metric"], "diff": t["diff_b_minus_a"],
             "p_raw": t["p_cluster_signflip"], "p_holm_old": t["p_holm"]} for t in sel]
    if len(rows) != 36:
        raise ValueError(f"36 Vergleiche erwartet, {len(rows)} gefunden")
    return rows


def add_holm(rows: list[dict]) -> list[dict]:
    """Ergänze Holm-Varianten A (je Metrik, neun) und B (alle 36) und das Urteil bei 0,05.

    Args:
        rows: Ausgabe von :func:`load_rows`.

    Returns:
        Neue Zeilen (Eingabe bleibt unverändert) mit ``p_holm_A``, ``p_holm_B`` und den
        Urteilen ``signif_alt``, ``signif_A``, ``signif_B`` (Holm-p < 0,05).
    """
    out = [dict(r) for r in rows]
    for i, p in enumerate(holm([r["p_raw"] for r in rows])):
        out[i]["p_holm_B"] = p
    for m in METRICS:
        idx = [i for i, r in enumerate(rows) if r["metric"] == m]
        for i, p in zip(idx, holm([rows[i]["p_raw"] for i in idx])):
            out[i]["p_holm_A"] = p
    for r in out:
        for k, key in (("old", "p_holm_old"), ("A", "p_holm_A"), ("B", "p_holm_B")):
            r[f"signif_{k}"] = r[key] < ALPHA
        r["urteil_aendert_sich_A"] = r["signif_A"] != r["signif_old"]
        r["urteil_aendert_sich_B"] = r["signif_B"] != r["signif_old"]
    return out


def de(x: float, nd: int = 4, sign: bool = False) -> str:
    """Formatiere eine Zahl mit Dezimalkomma."""
    return f"{x:+.{nd}f}".replace(".", ",") if sign else f"{x:.{nd}f}".replace(".", ",")


def to_markdown(rows: list[dict]) -> str:
    """Erzeuge die Markdown-Tabelle (deutsche Dezimalkommas, vier Nachkommastellen)."""
    head = ["# Robustheit: Holm-Korrektur über alle neun Modelle gegen BM25-de", "",
            "Differenz = Embedding-Modell minus BM25-de; p roh und p Holm (alt) unverändert aus den "
            "Ergebnisdateien. A: Holm je Metrik über neun Vergleiche; B: Holm über alle 36 Vergleiche. "
            "Urteil: Holm-p < 0,05. Alle Werte auf vier Nachkommastellen gerundet.", "",
            "| Modell | Metrik | Tabelle | Differenz | p roh | p Holm alt | p Holm A | p Holm B | Urteil alt | Urteil A | Urteil B |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    j = lambda b: "signifikant" if b else "n. s."  # noqa: E731
    for r in rows:
        head.append(f"| {r['model']} | {r['metric']} | {TABLE[r['metric']]} | {de(r['diff'], 4, True)} | "
                    f"{de(r['p_raw'])} | {de(r['p_holm_old'])} | {de(r['p_holm_A'])} | {de(r['p_holm_B'])} | "
                    f"{j(r['signif_old'])} | {j(r['signif_A'])} | {j(r['signif_B'])} |")
    nA = sum(r["urteil_aendert_sich_A"] for r in rows)
    nB = sum(r["urteil_aendert_sich_B"] for r in rows)
    head += ["", f"Urteilswechsel bei 0,05: Variante A {nA} von 36, Variante B {nB} von 36."]
    return "\n".join(head) + "\n"


def main(harness: Path = HARNESS) -> list[dict]:
    """Rechne beide Varianten und schreibe JSON und Markdown-Tabelle."""
    rows = add_holm(load_rows(harness))
    payload = {"meta": {"alpha": ALPHA, "n_comparisons": len(rows),
                        "quelle": "comparison_germanquad.json",
                        "hinweis": "Rohe p-Werte unverändert; nur die Familiengröße der Holm-Korrektur variiert."},
               "rows": rows}
    (harness / "robustness_holm_nine.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (harness / "results" / "table_robustness_holm.md").write_text(to_markdown(rows), encoding="utf-8")
    return rows


if __name__ == "__main__":
    r = main()
    print("Urteilswechsel A:", sum(x["urteil_aendert_sich_A"] for x in r),
          "B:", [(x["model"], x["metric"]) for x in r if x["urteil_aendert_sich_B"]])
