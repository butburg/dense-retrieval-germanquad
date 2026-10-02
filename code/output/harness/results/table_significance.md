# Signifikanz

Alle Tests auf der Primärmetrik MRR@10; Differenz = zweiter minus erster Retriever. Test: gepaarter Vorzeichenwechsel auf Cluster-Ebene, zweiseitig, B = 100000, Seed 42; CI: 95-%-Cluster-Bootstrap (B = 10000); exakt bei n≠0 ≤ 16; Monte-Carlo-p an der Untergrenze als Schranke (≤). Holm je Familie mit der tatsächlichen Testzahl als Familiengröße. Datensatz GermanQuAD, n = 2204 Queries, 474 Cluster. Sekundärmetriken stehen deskriptiv in table_main.md.

## Kernübersicht: MRR@10 je Modell mit p Holm gegen die Baselines (F2 und F3, je 9 Tests)

| Modell | MRR@10 | p Holm gegen BM25 | p Holm gegen BM25-de |
|---|---|---|---|
| bge-m3 | 0,946 | ≤ 9,0e-05 | 0,0022 |
| text-embedding-3-large | 0,944 | ≤ 9,0e-05 | 0,0093 |
| snowflake-arctic-embed-l-v2.0 | 0,944 | ≤ 9,0e-05 | 0,0038 |
| jina-embeddings-v3 | 0,943 | ≤ 9,0e-05 | 0,0093 |
| multilingual-e5-large | 0,933 | ≤ 9,0e-05 | 0,4346 |
| Qwen3-Embedding-0.6B | 0,924 | ≤ 9,0e-05 | 1,0000 |
| jina-embeddings-v2-base-de | 0,920 | 9,0e-05 | 1,0000 |
| text-embedding-3-small | 0,917 | 0,0006 | 1,0000 |
| gte-multilingual-base | 0,886 | 0,8837 | 0,0024 |

BM25-de gegen BM25 (F4, ohne Korrektur): Differenz +0,037, p ≤ 1,0e-05

## F1: Modellpaare (36 Tests, Holm über 36)

| Vergleich | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|
| bge-m3 − multilingual-e5-large | +0,012 | [+0,004; +0,021] | 0,0048 | 0,0717 | 126 |  |
| gte-multilingual-base − multilingual-e5-large | -0,047 | [-0,060; -0,035] | ≤ 1,0e-05 | ≤ 3,6e-04 | 164 |  |
| jina-embeddings-v3 − multilingual-e5-large | +0,010 | [+0,000; +0,019] | 0,0532 | 0,5857 | 133 |  |
| text-embedding-3-small − multilingual-e5-large | -0,016 | [-0,028; -0,005] | 0,0029 | 0,0469 | 142 |  |
| text-embedding-3-large − multilingual-e5-large | +0,010 | [-0,000; +0,022] | 0,0720 | 0,7197 | 133 |  |
| Qwen3-Embedding-0.6B − multilingual-e5-large | -0,009 | [-0,017; -0,002] | 0,0180 | 0,2343 | 146 |  |
| snowflake-arctic-embed-l-v2.0 − multilingual-e5-large | +0,010 | [+0,001; +0,020] | 0,0321 | 0,3847 | 129 |  |
| jina-embeddings-v2-base-de − multilingual-e5-large | -0,014 | [-0,024; -0,002] | 0,0144 | 0,2010 | 152 |  |
| gte-multilingual-base − bge-m3 | -0,060 | [-0,074; -0,046] | ≤ 1,0e-05 | ≤ 3,6e-04 | 158 |  |
| jina-embeddings-v3 − bge-m3 | -0,003 | [-0,010; +0,004] | 0,4553 | 1,0000 | 120 |  |
| text-embedding-3-small − bge-m3 | -0,029 | [-0,040; -0,018] | ≤ 1,0e-05 | ≤ 3,6e-04 | 153 |  |
| text-embedding-3-large − bge-m3 | -0,002 | [-0,012; +0,008] | 0,7169 | 1,0000 | 127 |  |
| Qwen3-Embedding-0.6B − bge-m3 | -0,021 | [-0,030; -0,014] | ≤ 1,0e-05 | ≤ 3,6e-04 | 139 |  |
| snowflake-arctic-embed-l-v2.0 − bge-m3 | -0,002 | [-0,009; +0,006] | 0,6260 | 1,0000 | 120 |  |
| jina-embeddings-v2-base-de − bge-m3 | -0,026 | [-0,035; -0,017] | ≤ 1,0e-05 | ≤ 3,6e-04 | 153 |  |
| jina-embeddings-v3 − gte-multilingual-base | +0,057 | [+0,043; +0,073] | ≤ 1,0e-05 | ≤ 3,6e-04 | 160 |  |
| text-embedding-3-small − gte-multilingual-base | +0,031 | [+0,016; +0,047] | 3,0e-05 | 0,0006 | 162 |  |
| text-embedding-3-large − gte-multilingual-base | +0,058 | [+0,042; +0,075] | ≤ 1,0e-05 | ≤ 3,6e-04 | 164 |  |
| Qwen3-Embedding-0.6B − gte-multilingual-base | +0,038 | [+0,026; +0,051] | ≤ 1,0e-05 | ≤ 3,6e-04 | 162 |  |
| snowflake-arctic-embed-l-v2.0 − gte-multilingual-base | +0,058 | [+0,044; +0,073] | ≤ 1,0e-05 | ≤ 3,6e-04 | 163 |  |
| jina-embeddings-v2-base-de − gte-multilingual-base | +0,034 | [+0,019; +0,050] | 2,0e-05 | 0,0004 | 176 |  |
| text-embedding-3-small − jina-embeddings-v3 | -0,026 | [-0,036; -0,016] | ≤ 1,0e-05 | ≤ 3,6e-04 | 147 |  |
| text-embedding-3-large − jina-embeddings-v3 | +0,001 | [-0,010; +0,011] | 0,8725 | 1,0000 | 133 |  |
| Qwen3-Embedding-0.6B − jina-embeddings-v3 | -0,019 | [-0,028; -0,010] | 4,0e-05 | 0,0007 | 148 |  |
| snowflake-arctic-embed-l-v2.0 − jina-embeddings-v3 | +0,001 | [-0,007; +0,008] | 0,8275 | 1,0000 | 113 |  |
| jina-embeddings-v2-base-de − jina-embeddings-v3 | -0,023 | [-0,032; -0,015] | ≤ 1,0e-05 | ≤ 3,6e-04 | 145 |  |
| text-embedding-3-large − text-embedding-3-small | +0,027 | [+0,018; +0,036] | ≤ 1,0e-05 | ≤ 3,6e-04 | 140 |  |
| Qwen3-Embedding-0.6B − text-embedding-3-small | +0,007 | [-0,002; +0,017] | 0,1268 | 1,0000 | 147 |  |
| snowflake-arctic-embed-l-v2.0 − text-embedding-3-small | +0,027 | [+0,016; +0,038] | ≤ 1,0e-05 | ≤ 3,6e-04 | 153 |  |
| jina-embeddings-v2-base-de − text-embedding-3-small | +0,003 | [-0,006; +0,012] | 0,5257 | 1,0000 | 164 |  |
| Qwen3-Embedding-0.6B − text-embedding-3-large | -0,020 | [-0,030; -0,010] | 3,0e-05 | 0,0006 | 145 |  |
| snowflake-arctic-embed-l-v2.0 − text-embedding-3-large | -0,000 | [-0,011; +0,011] | 0,9964 | 1,0000 | 138 |  |
| jina-embeddings-v2-base-de − text-embedding-3-large | -0,024 | [-0,034; -0,014] | ≤ 1,0e-05 | ≤ 3,6e-04 | 159 |  |
| snowflake-arctic-embed-l-v2.0 − Qwen3-Embedding-0.6B | +0,020 | [+0,011; +0,029] | 4,0e-05 | 0,0007 | 152 |  |
| jina-embeddings-v2-base-de − Qwen3-Embedding-0.6B | -0,004 | [-0,014; +0,005] | 0,3562 | 1,0000 | 162 |  |
| jina-embeddings-v2-base-de − snowflake-arctic-embed-l-v2.0 | -0,024 | [-0,033; -0,015] | ≤ 1,0e-05 | ≤ 3,6e-04 | 142 |  |

## F2: Modell gegen BM25 (9 Tests, Holm über 9)

| Vergleich | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | +0,049 | [+0,032; +0,066] | ≤ 1,0e-05 | ≤ 9,0e-05 | 230 |  |
| bge-m3 − BM25 | +0,061 | [+0,046; +0,077] | ≤ 1,0e-05 | ≤ 9,0e-05 | 224 |  |
| gte-multilingual-base − BM25 | +0,002 | [-0,018; +0,022] | 0,8837 | 0,8837 | 249 |  |
| jina-embeddings-v3 − BM25 | +0,058 | [+0,043; +0,075] | ≤ 1,0e-05 | ≤ 9,0e-05 | 234 |  |
| text-embedding-3-small − BM25 | +0,032 | [+0,014; +0,051] | 0,0003 | 0,0006 | 255 |  |
| text-embedding-3-large − BM25 | +0,059 | [+0,043; +0,076] | ≤ 1,0e-05 | ≤ 9,0e-05 | 232 |  |
| Qwen3-Embedding-0.6B − BM25 | +0,040 | [+0,024; +0,056] | ≤ 1,0e-05 | ≤ 9,0e-05 | 241 |  |
| snowflake-arctic-embed-l-v2.0 − BM25 | +0,059 | [+0,044; +0,075] | ≤ 1,0e-05 | ≤ 9,0e-05 | 233 |  |
| jina-embeddings-v2-base-de − BM25 | +0,035 | [+0,019; +0,053] | 2,0e-05 | 9,0e-05 | 259 |  |

## F3: Modell gegen BM25-de (9 Tests, Holm über 9)

| Vergleich | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25-de | +0,012 | [-0,003; +0,027] | 0,1087 | 0,4346 | 199 |  |
| bge-m3 − BM25-de | +0,024 | [+0,012; +0,038] | 0,0002 | 0,0022 | 187 |  |
| gte-multilingual-base − BM25-de | -0,035 | [-0,054; -0,017] | 0,0003 | 0,0024 | 219 |  |
| jina-embeddings-v3 − BM25-de | +0,022 | [+0,008; +0,036] | 0,0016 | 0,0093 | 202 |  |
| text-embedding-3-small − BM25-de | -0,004 | [-0,020; +0,011] | 0,5722 | 1,0000 | 209 |  |
| text-embedding-3-large − BM25-de | +0,022 | [+0,008; +0,037] | 0,0015 | 0,0093 | 196 |  |
| Qwen3-Embedding-0.6B − BM25-de | +0,003 | [-0,011; +0,017] | 0,6866 | 1,0000 | 206 |  |
| snowflake-arctic-embed-l-v2.0 − BM25-de | +0,022 | [+0,010; +0,036] | 0,0005 | 0,0038 | 189 |  |
| jina-embeddings-v2-base-de − BM25-de | -0,002 | [-0,016; +0,014] | 0,8366 | 1,0000 | 223 |  |

## F4: BM25-de gegen BM25 (1 Test, ohne Korrektur)

| Vergleich | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|
| BM25-de − BM25 | +0,037 | [+0,027; +0,048] | ≤ 1,0e-05 | ≤ 1,0e-05 | 196 |  |

Quelle: code/output/harness/comparison_germanquad.json; Fehlende Läufe: keine
