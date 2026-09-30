# Analyse GermanQuAD-Pilot: BM25 vs. multilingual-e5-large (dls-4ko.16)

**Explorativer Pilot; Auswerteprotokoll noch nicht vom Menschen freigegeben (Bead dls-4ko.25).**

Hinweis: Die Mehrfachvergleichsfamilie (Holm über Success@1/5/10 + MRR@10) und die Cluster-Inferenz (Cluster = relevantes Dokument) sind Vorschläge, keine festgelegte Analyse.

n = 2204 Queries, 474 Cluster; Seed 42; Permutation B=100000; Bootstrap B=10000; Differenz = E5 minus BM25.

## Tests

| Metrik | BM25 | E5 | Diff | b (nur BM25) | c (nur E5) | p query (optimistisch, ignoriert Cluster) | p_Holm query | p cluster-sign-flip | p_Holm cluster | 95%-CI query | 95%-CI cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Success@1 | 0.8376 | 0.8984 | +0.0608 | 150 | 284 | 1.24e-10 | 4.95e-10 | <= 1/(B+1) = 1e-05 | 4e-05 | [+0.0426, +0.0794] | [+0.0384, +0.0835] |
| Success@5 | 0.9483 | 0.9791 | +0.0309 | 38 | 106 | 1.27e-08 | 3.82e-08 | <= 1/(B+1) = 1e-05 | 4e-05 | [+0.0200, +0.0413] | [+0.0189, +0.0433] |
| Success@10 | 0.9646 | 0.9868 | +0.0222 | 28 | 77 | 1.85e-06 | 3.71e-06 | 3e-05 | 4e-05 | [+0.0132, +0.0313] | [+0.0118, +0.0333] |
| MRR@10 | 0.8844 | 0.9333 | +0.0488 | None | None | <= 1/(B+1) = 1e-05 | <= 1/(B+1) = 1e-05 | <= 1/(B+1) = 1e-05 | 4e-05 | [+0.0358, +0.0619] | [+0.0328, +0.0662] |

Query-Ebene (McNemar exakt; MRR@10 Sign-Flip pro Query) ist optimistisch, da sie die Cluster (Queries mit gleichem Gold-Dokument) ignoriert. Cluster-Sign-Flip: ein zufälliges Vorzeichen je Cluster auf der je Cluster aggregierten Differenz, zweiseitig, p=(Treffer+1)/(B+1), B=100000, Seed 42; Holm separat über die 4 Cluster-p-Werte. Monte-Carlo-p-Werte an der Untergrenze sind als Schranke "<= 1/(B+1)" (= 1e-05) ausgewiesen; McNemar-p sind exakt. Holm-Lesart: Die vier Cluster-p-Werte liegen bei oder knapp über 1/(B+1); Holm multipliziert den kleinsten mit 4 (Familiengröße), daher 4/(B+1) = 4e-05 als Untergrenze für alle vier p_Holm cluster.

## Clustergrößenverteilung (Queries je Gold-Dokument)

Anzahl Cluster: 474; min 1, Median 4, max 25.

| Clustergröße | Anzahl Cluster |
|---|---|
| 1 | 56 |
| 2 | 81 |
| 3-5 | 201 |
| 6-10 | 104 |
| 11-20 | 30 |
| >=21 | 2 |

Tests: Success@k exakter McNemar (zweiseitig); MRR@10 gepaarter Sign-Flip-Randomisierungstest (zweiseitig).

## Konsistenzprüfung gegen REPRO.md (Werte aus per_query_germanquad.csv)

| Retriever | k | Metrik | REPRO | aus CSV | abs. Diff |
|---|---|---|---|---|---|
| bm25 | 1 | success_at_k | 0.837568 | 0.837568 | 0.0e+00 |
| bm25 | 1 | mrr_at_k | 0.837568 | 0.837568 | 0.0e+00 |
| bm25 | 5 | success_at_k | 0.948276 | 0.948276 | 0.0e+00 |
| bm25 | 5 | mrr_at_k | 0.882116 | 0.882116 | 2.2e-16 |
| bm25 | 10 | success_at_k | 0.964610 | 0.964610 | 0.0e+00 |
| bm25 | 10 | mrr_at_k | 0.884404 | 0.884404 | 2.2e-16 |
| e5 | 1 | success_at_k | 0.898367 | 0.898367 | 0.0e+00 |
| e5 | 1 | mrr_at_k | 0.898367 | 0.898367 | 0.0e+00 |
| e5 | 5 | success_at_k | 0.979129 | 0.979129 | 0.0e+00 |
| e5 | 5 | mrr_at_k | 0.932214 | 0.932214 | 0.0e+00 |
| e5 | 10 | success_at_k | 0.986842 | 0.986842 | 0.0e+00 |
| e5 | 10 | mrr_at_k | 0.933254 | 0.933254 | 1.1e-16 |

Ergebnis: alle Werte exakt (Diff < 1e-12).

Grafik: fig_success_at_k_germanquad.png/.svg (Punktdiagramm, y-Achse ab 0, 95%-Cluster-Bootstrap-CI, als explorativ gekennzeichnet).

Limitationen: Ein Datensatz, ein Lauf; Cluster = erstes relevantes Dokument; E5 auf CPU (numerische Reihenfolge bei Gleichstand nicht garantiert).
