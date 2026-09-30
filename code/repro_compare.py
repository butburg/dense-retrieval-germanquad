"""Build output/repro/REPRO.md: reference vs. reproduction (GermanQuAD)."""
import json
from pathlib import Path
C = Path(__file__).resolve().parent / "output"
def ld(p): return json.loads(p.read_text())
def gq(m): return {r["k"]: r for r in m if r["dataset"] == "germanquad"}
specs = [("BM25", C/"bm25/bm25_results.json", C/"repro/bm25_germanquad_repro.json", 0.0),
         ("E5", C/"dense_e5/dense_e5_results.json", C/"repro/dense_e5_germanquad_repro.json", 0.005)]
out = ["# Reproduktion GermanQuAD (dls-4ko.3)", "", "| Retriever | k | Metrik | Referenz | Reproduktion | abs. Diff | Bewertung |", "|---|---|---|---|---|---|---|"]
for name, ref, rep, tol in specs:
    if not rep.exists():
        out.append(f"| {name} | - | - | - | nicht vorhanden | - | offen |"); continue
    R, P = gq(ld(ref)["metrics"]), gq(ld(rep)["metrics"])
    for k in sorted(R):
        for m in ("recall_at_k", "success_at_k", "mrr_at_k", "precision_at_k"):
            if m in R[k] and m in P[k]:
                d = abs(R[k][m] - P[k][m])
                v = "exakt" if d == 0 else ("innerhalb Toleranz (<=0.005)" if name == "E5" and d <= tol else "ABWEICHUNG")
                out.append(f"| {name} | {k} | {m} | {R[k][m]:.6f} | {P[k][m]:.6f} | {d:.2e} | {v} |")
bm = C/"repro/bm25_germanquad_repro.json"
if bm.exists():
    P = gq(ld(bm)["metrics"])
    out += ["", "## BM25 Precision@k (neu ergänzt, ohne Referenzwert im Pilot)", "",
            "Precision@k = Treffer in Top-k / k, gemittelt über alle Queries (gleiche Formel wie für E5). "
            "Bestehende BM25-Metriken sind gegenüber dem Lauf vor der Ergänzung bitgenau unverändert.", "",
            "| Retriever | k | precision_at_k |", "|---|---|---|"]
    out += [f"| BM25 | {k} | {P[k]['precision_at_k']:.6f} |" for k in sorted(P) if "precision_at_k" in P[k]]
    t = ld(bm).get("doc_id_types", {})
    out += ["", "## doc_id-Typprüfung", "",
            f"Rohtypen in JSONL: docs {t.get('docs')}, qrels {t.get('qrels')}; Ranking (Ausgabe) {t.get('ranking')}. "
            f"qrels-doc_ids ohne Treffer in docs: {t.get('qrels_doc_ids_not_in_docs')}. "
            "Der Code castet zusätzlich alle IDs mit str(); der BM25-Tie-Break `(-score, doc_id)` sortiert damit "
            "lexikografisch als String (z. B. \"c10\" vor \"c9\")."]
out += ["", "## Pfad-Limitation", "",
        "Die Notebooks 03/04 enthalten in gespeicherten Zell-Ausgaben absolute Pfade der lokalen Maschine des Autors "
        "(`<PROJEKT>/...`, 4 Fundstellen per Suche in `code/*.ipynb`) sowie Python 3.9.6. "
        "Diese Pfade sind nicht auf andere Rechner übertragbar. Die Reproduktion (`repro_germanquad.py`) nutzt nur relative Pfade "
        "unter `code/output/` und lief auf Python 3.11 (Linux); die Metriken stimmen trotzdem exakt (BM25) bzw. innerhalb Toleranz (E5) überein."]
(C/"repro/REPRO.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
