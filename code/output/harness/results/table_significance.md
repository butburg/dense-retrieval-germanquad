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
| text-embedding-3-large − BM25 | MRR@10 | +0,059 | [+0,043; +0,076] | ≤ 1,0e-05 | ≤ 6,0e-05 | 232 |  |

## Explorativ (Holm nur innerhalb der jeweiligen Familie, keine konfirmatorische Aussage)

### explorative (a): new model vs BM25 standard, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25 | MRR@10 | +0,040 | [+0,024; +0,056] | ≤ 1,0e-05 | ≤ 3,0e-05 | 241 |  |
| snowflake-arctic-embed-l-v2.0 − BM25 | MRR@10 | +0,059 | [+0,044; +0,075] | ≤ 1,0e-05 | ≤ 3,0e-05 | 233 |  |
| jina-embeddings-v2-base-de − BM25 | MRR@10 | +0,035 | [+0,019; +0,053] | 2,0e-05 | 3,0e-05 | 259 |  |

### explorative (a): new model vs BM25 standard, Success@1

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25 | Success@1 | +0,046 | [+0,024; +0,068] | 3,0e-05 | 6,0e-05 | 204 |  |
| snowflake-arctic-embed-l-v2.0 − BM25 | Success@1 | +0,074 | [+0,053; +0,096] | ≤ 1,0e-05 | ≤ 3,0e-05 | 206 |  |
| jina-embeddings-v2-base-de − BM25 | Success@1 | +0,042 | [+0,020; +0,066] | 0,0002 | 0,0002 | 225 |  |

### explorative (a): new model vs BM25 standard, Success@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25 | Success@10 | +0,020 | [+0,010; +0,031] | 0,0002 | 0,0005 | 79 |  |
| snowflake-arctic-embed-l-v2.0 − BM25 | Success@10 | +0,026 | [+0,017; +0,036] | ≤ 1,0e-05 | ≤ 3,0e-05 | 63 |  |
| jina-embeddings-v2-base-de − BM25 | Success@10 | +0,018 | [+0,008; +0,029] | 0,0007 | 0,0007 | 80 |  |

### explorative (a): new model vs BM25 standard, Success@5

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25 | Success@5 | +0,025 | [+0,013; +0,039] | 9,0e-05 | 0,0002 | 96 |  |
| snowflake-arctic-embed-l-v2.0 − BM25 | Success@5 | +0,033 | [+0,022; +0,045] | ≤ 1,0e-05 | ≤ 3,0e-05 | 94 |  |
| jina-embeddings-v2-base-de − BM25 | Success@5 | +0,020 | [+0,007; +0,033] | 0,0030 | 0,0030 | 113 |  |

### explorative (b): new model vs BM25-de, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25-de | MRR@10 | +0,003 | [-0,011; +0,017] | 0,6866 | 1,0000 | 206 |  |
| snowflake-arctic-embed-l-v2.0 − BM25-de | MRR@10 | +0,022 | [+0,010; +0,036] | 0,0005 | 0,0016 | 189 |  |
| jina-embeddings-v2-base-de − BM25-de | MRR@10 | -0,002 | [-0,016; +0,014] | 0,8366 | 1,0000 | 223 |  |

### explorative (b): new model vs BM25-de, Success@1

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25-de | Success@1 | +0,000 | [-0,018; +0,020] | 1,0000 | 1,0000 | 187 |  |
| snowflake-arctic-embed-l-v2.0 − BM25-de | Success@1 | +0,029 | [+0,011; +0,047] | 0,0019 | 0,0058 | 166 |  |
| jina-embeddings-v2-base-de − BM25-de | Success@1 | -0,003 | [-0,024; +0,018] | 0,7959 | 1,0000 | 197 |  |

### explorative (b): new model vs BM25-de, Success@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25-de | Success@10 | +0,000 | [-0,008; +0,008] | 1,0000 | 1,0000 | 47 |  |
| snowflake-arctic-embed-l-v2.0 − BM25-de | Success@10 | +0,006 | [-0,000; +0,013] | 0,0925 | 0,2774 | 34 |  |
| jina-embeddings-v2-base-de − BM25-de | Success@10 | -0,002 | [-0,009; +0,006] | 0,7243 | 1,0000 | 51 |  |

### explorative (b): new model vs BM25-de, Success@5

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − BM25-de | Success@5 | +0,002 | [-0,010; +0,013] | 0,8194 | 1,0000 | 69 |  |
| snowflake-arctic-embed-l-v2.0 − BM25-de | Success@5 | +0,010 | [+0,000; +0,019] | 0,0627 | 0,1882 | 60 |  |
| jina-embeddings-v2-base-de − BM25-de | Success@5 | -0,004 | [-0,014; +0,007] | 0,5603 | 1,0000 | 83 |  |

### explorative (c): new model vs bge-m3, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − bge-m3 | MRR@10 | -0,021 | [-0,030; -0,014] | ≤ 1,0e-05 | ≤ 3,0e-05 | 139 |  |
| snowflake-arctic-embed-l-v2.0 − bge-m3 | MRR@10 | -0,002 | [-0,009; +0,006] | 0,6260 | 0,6260 | 120 |  |
| jina-embeddings-v2-base-de − bge-m3 | MRR@10 | -0,026 | [-0,035; -0,017] | ≤ 1,0e-05 | ≤ 3,0e-05 | 153 |  |

### explorative (c): new model vs bge-m3, Success@1

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − bge-m3 | Success@1 | -0,034 | [-0,046; -0,021] | ≤ 1,0e-05 | ≤ 3,0e-05 | 112 |  |
| snowflake-arctic-embed-l-v2.0 − bge-m3 | Success@1 | -0,005 | [-0,017; +0,006] | 0,3967 | 0,3967 | 91 |  |
| jina-embeddings-v2-base-de − bge-m3 | Success@1 | -0,037 | [-0,051; -0,024] | ≤ 1,0e-05 | ≤ 3,0e-05 | 125 |  |

### explorative (c): new model vs bge-m3, Success@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − bge-m3 | Success@10 | -0,004 | [-0,009; +0,000] | 0,1374 | 0,2747 | 23 |  |
| snowflake-arctic-embed-l-v2.0 − bge-m3 | Success@10 | +0,002 | [-0,002; +0,007] | 0,4073 | 0,4073 | 20 |  |
| jina-embeddings-v2-base-de − bge-m3 | Success@10 | -0,006 | [-0,011; -0,000] | 0,0452 | 0,1357 | 23 |  |

### explorative (c): new model vs bge-m3, Success@5

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B − bge-m3 | Success@5 | -0,007 | [-0,014; -0,000] | 0,0571 | 0,1141 | 37 |  |
| snowflake-arctic-embed-l-v2.0 − bge-m3 | Success@5 | +0,001 | [-0,004; +0,006] | 0,8608 | 0,8608 | 32 |  |
| jina-embeddings-v2-base-de − bge-m3 | Success@5 | -0,012 | [-0,019; -0,006] | 0,0005 | 0,0016 | 44 |  |

### BM25-de vs BM25 standard (4 metrics)

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| BM25-de − BM25 | MRR@10 | +0,037 | [+0,027; +0,048] | ≤ 1,0e-05 | ≤ 4,0e-05 | 196 |  |
| BM25-de − BM25 | Success@1 | +0,045 | [+0,031; +0,060] | ≤ 1,0e-05 | ≤ 4,0e-05 | 145 |  |
| BM25-de − BM25 | Success@5 | +0,024 | [+0,014; +0,034] | ≤ 1,0e-05 | ≤ 4,0e-05 | 70 |  |
| BM25-de − BM25 | Success@10 | +0,020 | [+0,012; +0,028] | ≤ 1,0e-05 | ≤ 4,0e-05 | 39 |  |

### embedder pairs, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| bge-m3 − multilingual-e5-large | MRR@10 | +0,012 | [+0,004; +0,021] | 0,0048 | 0,0287 | 126 |  |
| gte-multilingual-base − multilingual-e5-large | MRR@10 | -0,047 | [-0,060; -0,035] | ≤ 1,0e-05 | ≤ 1,5e-04 | 164 |  |
| jina-embeddings-v3 − multilingual-e5-large | MRR@10 | +0,010 | [+0,000; +0,019] | 0,0532 | 0,2662 | 133 |  |
| text-embedding-3-small − multilingual-e5-large | MRR@10 | -0,016 | [-0,028; -0,005] | 0,0029 | 0,0205 | 142 |  |
| text-embedding-3-large − multilingual-e5-large | MRR@10 | +0,010 | [-0,000; +0,022] | 0,0720 | 0,2879 | 133 |  |
| gte-multilingual-base − bge-m3 | MRR@10 | -0,060 | [-0,074; -0,046] | ≤ 1,0e-05 | ≤ 1,5e-04 | 158 |  |
| jina-embeddings-v3 − bge-m3 | MRR@10 | -0,003 | [-0,010; +0,004] | 0,4553 | 1,0000 | 120 |  |
| text-embedding-3-small − bge-m3 | MRR@10 | -0,029 | [-0,040; -0,018] | ≤ 1,0e-05 | ≤ 1,5e-04 | 153 |  |
| text-embedding-3-large − bge-m3 | MRR@10 | -0,002 | [-0,012; +0,008] | 0,7169 | 1,0000 | 127 |  |
| jina-embeddings-v3 − gte-multilingual-base | MRR@10 | +0,057 | [+0,043; +0,073] | ≤ 1,0e-05 | ≤ 1,5e-04 | 160 |  |
| text-embedding-3-small − gte-multilingual-base | MRR@10 | +0,031 | [+0,016; +0,047] | 3,0e-05 | 0,0002 | 162 |  |
| text-embedding-3-large − gte-multilingual-base | MRR@10 | +0,058 | [+0,042; +0,075] | ≤ 1,0e-05 | ≤ 1,5e-04 | 164 |  |
| text-embedding-3-small − jina-embeddings-v3 | MRR@10 | -0,026 | [-0,036; -0,016] | ≤ 1,0e-05 | ≤ 1,5e-04 | 147 |  |
| text-embedding-3-large − jina-embeddings-v3 | MRR@10 | +0,001 | [-0,010; +0,011] | 0,8725 | 1,0000 | 133 |  |
| text-embedding-3-large − text-embedding-3-small | MRR@10 | +0,027 | [+0,018; +0,036] | ≤ 1,0e-05 | ≤ 1,5e-04 | 140 |  |

### embedder vs BM25, Success@1

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | Success@1 | +0,061 | [+0,038; +0,084] | ≤ 1,0e-05 | ≤ 6,0e-05 | 211 |  |
| bge-m3 − BM25 | Success@1 | +0,079 | [+0,060; +0,100] | ≤ 1,0e-05 | ≤ 6,0e-05 | 198 |  |
| gte-multilingual-base − BM25 | Success@1 | -0,004 | [-0,030; +0,023] | 0,8162 | 0,8162 | 217 |  |
| jina-embeddings-v3 − BM25 | Success@1 | +0,070 | [+0,048; +0,093] | ≤ 1,0e-05 | ≤ 6,0e-05 | 215 |  |
| text-embedding-3-small − BM25 | Success@1 | +0,035 | [+0,010; +0,062] | 0,0071 | 0,0143 | 222 |  |
| text-embedding-3-large − BM25 | Success@1 | +0,070 | [+0,047; +0,094] | ≤ 1,0e-05 | ≤ 6,0e-05 | 211 |  |

### embedder vs BM25, Success@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | Success@10 | +0,022 | [+0,012; +0,033] | 6,0e-05 | 0,0002 | 72 |  |
| bge-m3 − BM25 | Success@10 | +0,024 | [+0,014; +0,035] | ≤ 1,0e-05 | ≤ 6,0e-05 | 71 |  |
| gte-multilingual-base − BM25 | Success@10 | +0,003 | [-0,012; +0,017] | 0,7572 | 0,7572 | 86 |  |
| jina-embeddings-v3 − BM25 | Success@10 | +0,027 | [+0,017; +0,037] | ≤ 1,0e-05 | ≤ 6,0e-05 | 69 |  |
| text-embedding-3-small − BM25 | Success@10 | +0,018 | [+0,007; +0,029] | 0,0015 | 0,0031 | 81 |  |
| text-embedding-3-large − BM25 | Success@10 | +0,029 | [+0,019; +0,039] | ≤ 1,0e-05 | ≤ 6,0e-05 | 66 |  |

### embedder vs BM25, Success@5

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25 | Success@5 | +0,031 | [+0,019; +0,044] | ≤ 1,0e-05 | ≤ 6,0e-05 | 96 |  |
| bge-m3 − BM25 | Success@5 | +0,032 | [+0,021; +0,044] | ≤ 1,0e-05 | ≤ 6,0e-05 | 99 |  |
| gte-multilingual-base − BM25 | Success@5 | +0,004 | [-0,013; +0,021] | 0,7175 | 0,7175 | 112 |  |
| jina-embeddings-v3 − BM25 | Success@5 | +0,038 | [+0,027; +0,050] | ≤ 1,0e-05 | ≤ 6,0e-05 | 95 |  |
| text-embedding-3-small − BM25 | Success@5 | +0,026 | [+0,013; +0,039] | 4,0e-05 | 8,0e-05 | 104 |  |
| text-embedding-3-large − BM25 | Success@5 | +0,041 | [+0,030; +0,054] | ≤ 1,0e-05 | ≤ 6,0e-05 | 88 |  |

### embedder vs BM25-de, MRR@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25-de | MRR@10 | +0,012 | [-0,003; +0,027] | 0,1087 | 0,2173 | 199 |  |
| bge-m3 − BM25-de | MRR@10 | +0,024 | [+0,012; +0,038] | 0,0002 | 0,0014 | 187 |  |
| gte-multilingual-base − BM25-de | MRR@10 | -0,035 | [-0,054; -0,017] | 0,0003 | 0,0015 | 219 |  |
| jina-embeddings-v3 − BM25-de | MRR@10 | +0,022 | [+0,008; +0,036] | 0,0016 | 0,0062 | 202 |  |
| text-embedding-3-small − BM25-de | MRR@10 | -0,004 | [-0,020; +0,011] | 0,5722 | 0,5722 | 209 |  |
| text-embedding-3-large − BM25-de | MRR@10 | +0,022 | [+0,008; +0,037] | 0,0015 | 0,0062 | 196 |  |

### embedder vs BM25-de, Success@1

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25-de | Success@1 | +0,015 | [-0,005; +0,036] | 0,1489 | 0,2979 | 184 |  |
| bge-m3 − BM25-de | Success@1 | +0,034 | [+0,017; +0,052] | 0,0001 | 0,0009 | 167 |  |
| gte-multilingual-base − BM25-de | Success@1 | -0,049 | [-0,073; -0,025] | 0,0002 | 0,0009 | 196 |  |
| jina-embeddings-v3 − BM25-de | Success@1 | +0,025 | [+0,005; +0,045] | 0,0159 | 0,0636 | 180 |  |
| text-embedding-3-small − BM25-de | Success@1 | -0,010 | [-0,033; +0,013] | 0,4024 | 0,4024 | 182 |  |
| text-embedding-3-large − BM25-de | Success@1 | +0,025 | [+0,004; +0,046] | 0,0206 | 0,0636 | 180 |  |

### embedder vs BM25-de, Success@10

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25-de | Success@10 | +0,002 | [-0,006; +0,011] | 0,6672 | 1,0000 | 45 |  |
| bge-m3 − BM25-de | Success@10 | +0,004 | [-0,003; +0,012] | 0,3370 | 1,0000 | 40 |  |
| gte-multilingual-base − BM25-de | Success@10 | -0,017 | [-0,030; -0,005] | 0,0086 | 0,0516 | 58 |  |
| jina-embeddings-v3 − BM25-de | Success@10 | +0,007 | [+0,001; +0,014] | 0,0399 | 0,1595 | 39 |  |
| text-embedding-3-small − BM25-de | Success@10 | -0,002 | [-0,010; +0,006] | 0,6611 | 1,0000 | 49 |  |
| text-embedding-3-large − BM25-de | Success@10 | +0,009 | [+0,002; +0,016] | 0,0118 | 0,0592 | 34 |  |

### embedder vs BM25-de, Success@5

| Vergleich | Metrik | Differenz | 95%-CI | p roh | p Holm | n≠0 | Hinweis |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large − BM25-de | Success@5 | +0,007 | [-0,004; +0,018] | 0,2140 | 0,4281 | 61 |  |
| bge-m3 − BM25-de | Success@5 | +0,009 | [-0,001; +0,019] | 0,1030 | 0,3089 | 64 |  |
| gte-multilingual-base − BM25-de | Success@5 | -0,020 | [-0,038; -0,003] | 0,0253 | 0,1013 | 74 |  |
| jina-embeddings-v3 − BM25-de | Success@5 | +0,015 | [+0,006; +0,024] | 0,0018 | 0,0091 | 59 |  |
| text-embedding-3-small − BM25-de | Success@5 | +0,002 | [-0,008; +0,013] | 0,7413 | 0,7413 | 73 |  |
| text-embedding-3-large − BM25-de | Success@5 | +0,018 | [+0,008; +0,028] | 0,0002 | 0,0014 | 53 |  |

Quelle: code/output/harness/comparison_germanquad.json; Fehlende Läufe: keine
