# Signifikanz

Differenz = zweiter minus erster Retriever. Test: gepaarter Vorzeichenwechsel auf Cluster-Ebene, zweiseitig, B = 100000, Seed 42; CI: 95-%-Cluster-Bootstrap (B = 10000); exakt bei n≠0 ≤ 16; Monte-Carlo-p an der Untergrenze als Schranke (≤). Holm in der konfirmatorischen Familie mit Familiengröße 6 (geplant; aktuell 6 Tests vorhanden). Datensatz GermanQuAD, n = 2204 Queries, 474 Cluster.

## Konfirmatorisch: Embedder vs. BM25, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | MRR@10 | +0,049 | [+0,032; +0,066] | ≤ 1,0e-05 | ≤ 6,0e-05 | 230 |  |
| bge-m3 − BM25 | MRR@10 | +0,061 | [+0,046; +0,077] | ≤ 1,0e-05 | ≤ 6,0e-05 | 224 |  |
| gte-multilingual-base − BM25 | MRR@10 | +0,002 | [-0,018; +0,022] | 0,8837 | 0,8837 | 249 |  |
| jina-embeddings-v3 − BM25 | MRR@10 | +0,058 | [+0,043; +0,075] | ≤ 1,0e-05 | ≤ 6,0e-05 | 234 |  |
| text-embedding-3-small − BM25 | MRR@10 | +0,032 | [+0,014; +0,051] | 0,0003 | 0,0006 | 255 |  |
| text-embedding-3-large − BM25 | MRR@10 | +0,060 | [+0,043; +0,077] | ≤ 1,0e-05 | ≤ 6,0e-05 | 231 |  |

## Explorativ (Holm nur innerhalb der jeweiligen Familie, keine konfirmatorische Aussage)

### embedder pairs, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| bge-m3 − multilingual-e5-large | MRR@10 | +0,012 | [+0,004; +0,021] | 0,0048 | 0,0287 | 126 |  |
| gte-multilingual-base − multilingual-e5-large | MRR@10 | -0,047 | [-0,060; -0,035] | ≤ 1,0e-05 | ≤ 1,5e-04 | 164 |  |
| jina-embeddings-v3 − multilingual-e5-large | MRR@10 | +0,010 | [+0,000; +0,019] | 0,0532 | 0,2662 | 133 |  |
| text-embedding-3-small − multilingual-e5-large | MRR@10 | -0,016 | [-0,028; -0,005] | 0,0029 | 0,0205 | 142 |  |
| text-embedding-3-large − multilingual-e5-large | MRR@10 | +0,011 | [+0,000; +0,022] | 0,0573 | 0,2662 | 133 |  |
| gte-multilingual-base − bge-m3 | MRR@10 | -0,060 | [-0,074; -0,046] | ≤ 1,0e-05 | ≤ 1,5e-04 | 158 |  |
| jina-embeddings-v3 − bge-m3 | MRR@10 | -0,003 | [-0,010; +0,004] | 0,4553 | 1,0000 | 120 |  |
| text-embedding-3-small − bge-m3 | MRR@10 | -0,029 | [-0,040; -0,018] | ≤ 1,0e-05 | ≤ 1,5e-04 | 153 |  |
| text-embedding-3-large − bge-m3 | MRR@10 | -0,001 | [-0,011; +0,009] | 0,8058 | 1,0000 | 125 |  |
| jina-embeddings-v3 − gte-multilingual-base | MRR@10 | +0,057 | [+0,043; +0,073] | ≤ 1,0e-05 | ≤ 1,5e-04 | 160 |  |
| text-embedding-3-small − gte-multilingual-base | MRR@10 | +0,031 | [+0,016; +0,047] | 3,0e-05 | 0,0002 | 162 |  |
| text-embedding-3-large − gte-multilingual-base | MRR@10 | +0,058 | [+0,042; +0,076] | ≤ 1,0e-05 | ≤ 1,5e-04 | 164 |  |
| text-embedding-3-small − jina-embeddings-v3 | MRR@10 | -0,026 | [-0,036; -0,016] | ≤ 1,0e-05 | ≤ 1,5e-04 | 147 |  |
| text-embedding-3-large − jina-embeddings-v3 | MRR@10 | +0,001 | [-0,009; +0,012] | 0,7865 | 1,0000 | 133 |  |
| text-embedding-3-large − text-embedding-3-small | MRR@10 | +0,027 | [+0,019; +0,037] | ≤ 1,0e-05 | ≤ 1,5e-04 | 139 |  |

### embedder vs BM25, Success@1

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | Success@1 | +0,061 | [+0,038; +0,084] | ≤ 1,0e-05 | ≤ 6,0e-05 | 211 |  |
| bge-m3 − BM25 | Success@1 | +0,079 | [+0,060; +0,100] | ≤ 1,0e-05 | ≤ 6,0e-05 | 198 |  |
| gte-multilingual-base − BM25 | Success@1 | -0,004 | [-0,030; +0,023] | 0,8162 | 0,8162 | 217 |  |
| jina-embeddings-v3 − BM25 | Success@1 | +0,070 | [+0,048; +0,093] | ≤ 1,0e-05 | ≤ 6,0e-05 | 215 |  |
| text-embedding-3-small − BM25 | Success@1 | +0,035 | [+0,010; +0,062] | 0,0071 | 0,0143 | 222 |  |
| text-embedding-3-large − BM25 | Success@1 | +0,071 | [+0,048; +0,095] | ≤ 1,0e-05 | ≤ 6,0e-05 | 210 |  |

McNemar (exakt, optimistisch, ignoriert Cluster; deskriptiv), b = nur erster, c = nur zweiter Retriever: multilingual-e5-large: b=150, c=284, p=1,2e-10; bge-m3: b=107, c=282, p=2,7e-19; gte-multilingual-base: b=246, c=238, p=0,7504; jina-embeddings-v3: b=132, c=286, p=3,7e-14; text-embedding-3-small: b=198, c=276, p=0,0004; text-embedding-3-large: b=136, c=292, p=3,5e-14

### embedder vs BM25, Success@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | Success@10 | +0,022 | [+0,012; +0,033] | 6,0e-05 | 0,0002 | 72 |  |
| bge-m3 − BM25 | Success@10 | +0,024 | [+0,014; +0,035] | ≤ 1,0e-05 | ≤ 6,0e-05 | 71 |  |
| gte-multilingual-base − BM25 | Success@10 | +0,003 | [-0,012; +0,017] | 0,7572 | 0,7572 | 86 |  |
| jina-embeddings-v3 − BM25 | Success@10 | +0,027 | [+0,017; +0,037] | ≤ 1,0e-05 | ≤ 6,0e-05 | 69 |  |
| text-embedding-3-small − BM25 | Success@10 | +0,018 | [+0,007; +0,029] | 0,0015 | 0,0031 | 81 |  |
| text-embedding-3-large − BM25 | Success@10 | +0,029 | [+0,019; +0,039] | ≤ 1,0e-05 | ≤ 6,0e-05 | 66 |  |

McNemar (exakt, optimistisch, ignoriert Cluster; deskriptiv), b = nur erster, c = nur zweiter Retriever: multilingual-e5-large: b=28, c=77, p=1,9e-06; bge-m3: b=23, c=76, p=8,5e-08; gte-multilingual-base: b=67, c=73, p=0,6728; jina-embeddings-v3: b=15, c=75, p=9,2e-11; text-embedding-3-small: b=35, c=74, p=0,0002; text-embedding-3-large: b=14, c=78, p=6,2e-12

### embedder vs BM25, Success@5

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | Success@5 | +0,031 | [+0,019; +0,044] | ≤ 1,0e-05 | ≤ 6,0e-05 | 96 |  |
| bge-m3 − BM25 | Success@5 | +0,032 | [+0,021; +0,044] | ≤ 1,0e-05 | ≤ 6,0e-05 | 99 |  |
| gte-multilingual-base − BM25 | Success@5 | +0,004 | [-0,013; +0,021] | 0,7175 | 0,7175 | 112 |  |
| jina-embeddings-v3 − BM25 | Success@5 | +0,038 | [+0,027; +0,050] | ≤ 1,0e-05 | ≤ 6,0e-05 | 95 |  |
| text-embedding-3-small − BM25 | Success@5 | +0,026 | [+0,013; +0,039] | 4,0e-05 | 8,0e-05 | 104 |  |
| text-embedding-3-large − BM25 | Success@5 | +0,041 | [+0,030; +0,054] | ≤ 1,0e-05 | ≤ 6,0e-05 | 88 |  |

McNemar (exakt, optimistisch, ignoriert Cluster; deskriptiv), b = nur erster, c = nur zweiter Retriever: multilingual-e5-large: b=38, c=106, p=1,3e-08; bge-m3: b=36, c=107, p=2,3e-09; gte-multilingual-base: b=90, c=98, p=0,6098; jina-embeddings-v3: b=24, c=108, p=6,4e-14; text-embedding-3-small: b=50, c=107, p=6,3e-06; text-embedding-3-large: b=20, c=111, p=1,8e-16

Quelle: output/harness/comparison_germanquad.json; Fehlende Läufe: keine
