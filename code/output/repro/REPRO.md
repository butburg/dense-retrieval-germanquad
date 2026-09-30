# Reproduktion GermanQuAD (dls-4ko.3)

| Retriever | k | Metrik | Referenz | Reproduktion | abs. Diff | Bewertung |
|---|---|---|---|---|---|---|
| BM25 | 1 | recall_at_k | 0.837568 | 0.837568 | 0.00e+00 | exakt |
| BM25 | 1 | success_at_k | 0.837568 | 0.837568 | 0.00e+00 | exakt |
| BM25 | 1 | mrr_at_k | 0.837568 | 0.837568 | 0.00e+00 | exakt |
| BM25 | 5 | recall_at_k | 0.948276 | 0.948276 | 0.00e+00 | exakt |
| BM25 | 5 | success_at_k | 0.948276 | 0.948276 | 0.00e+00 | exakt |
| BM25 | 5 | mrr_at_k | 0.882116 | 0.882116 | 0.00e+00 | exakt |
| BM25 | 10 | recall_at_k | 0.964610 | 0.964610 | 0.00e+00 | exakt |
| BM25 | 10 | success_at_k | 0.964610 | 0.964610 | 0.00e+00 | exakt |
| BM25 | 10 | mrr_at_k | 0.884404 | 0.884404 | 0.00e+00 | exakt |
| E5 | 1 | recall_at_k | 0.898367 | 0.898367 | 0.00e+00 | exakt |
| E5 | 1 | success_at_k | 0.898367 | 0.898367 | 0.00e+00 | exakt |
| E5 | 1 | mrr_at_k | 0.898367 | 0.898367 | 0.00e+00 | exakt |
| E5 | 1 | precision_at_k | 0.898367 | 0.898367 | 0.00e+00 | exakt |
| E5 | 5 | recall_at_k | 0.979129 | 0.979129 | 0.00e+00 | exakt |
| E5 | 5 | success_at_k | 0.979129 | 0.979129 | 0.00e+00 | exakt |
| E5 | 5 | mrr_at_k | 0.932214 | 0.932214 | 0.00e+00 | exakt |
| E5 | 5 | precision_at_k | 0.195826 | 0.195826 | 0.00e+00 | exakt |
| E5 | 10 | recall_at_k | 0.986842 | 0.986842 | 0.00e+00 | exakt |
| E5 | 10 | success_at_k | 0.986842 | 0.986842 | 0.00e+00 | exakt |
| E5 | 10 | mrr_at_k | 0.933254 | 0.933254 | 0.00e+00 | exakt |
| E5 | 10 | precision_at_k | 0.098684 | 0.098684 | 0.00e+00 | exakt |

## BM25 Precision@k (neu ergänzt, ohne Referenzwert im Pilot)

Precision@k = Treffer in Top-k / k, gemittelt über alle Queries (gleiche Formel wie für E5). Bestehende BM25-Metriken sind gegenüber dem Lauf vor der Ergänzung bitgenau unverändert.

| Retriever | k | precision_at_k |
|---|---|---|
| BM25 | 1 | 0.837568 |
| BM25 | 5 | 0.189655 |
| BM25 | 10 | 0.096461 |

## doc_id-Typprüfung

Rohtypen in JSONL: docs ['str'], qrels ['str']; Ranking (Ausgabe) ['str']. qrels-doc_ids ohne Treffer in docs: 0. Der Code castet zusätzlich alle IDs mit str(); der BM25-Tie-Break `(-score, doc_id)` sortiert damit lexikografisch als String (z. B. "c10" vor "c9").

## Pfad-Limitation

Die Notebooks 03/04 enthalten in gespeicherten Zell-Ausgaben absolute Pfade der lokalen Maschine des Autors (`<PROJEKT>/...`, 4 Fundstellen per Suche in `code/*.ipynb`) sowie Python 3.9.6. Diese Pfade sind nicht auf andere Rechner übertragbar. Die Reproduktion (`repro_germanquad.py`) nutzt nur relative Pfade unter `code/output/` und lief auf Python 3.11 (Linux); die Metriken stimmen trotzdem exakt (BM25) bzw. innerhalb Toleranz (E5) überein.
