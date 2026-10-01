# Robustheit: Holm-Korrektur über alle neun Modelle gegen BM25-de

Differenz = Embedding-Modell minus BM25-de; p roh und p Holm (alt) unverändert aus den Ergebnisdateien. A: Holm je Metrik über neun Vergleiche; B: Holm über alle 36 Vergleiche. Urteil: Holm-p < 0,05. Alle Werte auf vier Nachkommastellen gerundet.

| Modell | Metrik | Tabelle | Differenz | p roh | p Holm alt | p Holm A | p Holm B | Urteil alt | Urteil A | Urteil B |
|---|---|---|---|---|---|---|---|---|---|---|
| multilingual-e5-large | MRR@10 | Tabelle 4 | +0,0120 | 0,1087 | 0,2173 | 0,4346 | 1,0000 | n. s. | n. s. | n. s. |
| bge-m3 | MRR@10 | Tabelle 4 | +0,0243 | 0,0002 | 0,0014 | 0,0022 | 0,0082 | signifikant | signifikant | signifikant |
| gte-multilingual-base | MRR@10 | Tabelle 4 | -0,0354 | 0,0003 | 0,0015 | 0,0024 | 0,0096 | signifikant | signifikant | signifikant |
| jina-embeddings-v3 | MRR@10 | Tabelle 4 | +0,0216 | 0,0016 | 0,0062 | 0,0093 | 0,0465 | signifikant | signifikant | signifikant |
| text-embedding-3-small | MRR@10 | Tabelle 4 | -0,0045 | 0,5722 | 0,5722 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| text-embedding-3-large | MRR@10 | Tabelle 4 | +0,0224 | 0,0015 | 0,0062 | 0,0093 | 0,0465 | signifikant | signifikant | signifikant |
| multilingual-e5-large | Success@1 | D6/D7 | +0,0154 | 0,1489 | 0,2979 | 0,5958 | 1,0000 | n. s. | n. s. | n. s. |
| bge-m3 | Success@1 | D6/D7 | +0,0340 | 0,0001 | 0,0009 | 0,0013 | 0,0054 | signifikant | signifikant | signifikant |
| gte-multilingual-base | Success@1 | D6/D7 | -0,0490 | 0,0002 | 0,0009 | 0,0015 | 0,0066 | signifikant | signifikant | signifikant |
| jina-embeddings-v3 | Success@1 | D6/D7 | +0,0245 | 0,0159 | 0,0636 | 0,0953 | 0,3814 | n. s. | n. s. | n. s. |
| text-embedding-3-small | Success@1 | D6/D7 | -0,0100 | 0,4024 | 0,4024 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| text-embedding-3-large | Success@1 | D6/D7 | +0,0245 | 0,0206 | 0,0636 | 0,1028 | 0,4731 | n. s. | n. s. | n. s. |
| multilingual-e5-large | Success@5 | D6/D7 | +0,0073 | 0,2140 | 0,4281 | 0,8561 | 1,0000 | n. s. | n. s. | n. s. |
| bge-m3 | Success@5 | D6/D7 | +0,0086 | 0,1030 | 0,3089 | 0,5149 | 1,0000 | n. s. | n. s. | n. s. |
| gte-multilingual-base | Success@5 | D6/D7 | -0,0200 | 0,0253 | 0,1013 | 0,1773 | 0,5573 | n. s. | n. s. | n. s. |
| jina-embeddings-v3 | Success@5 | D6/D7 | +0,0145 | 0,0018 | 0,0091 | 0,0146 | 0,0512 | signifikant | signifikant | n. s. |
| text-embedding-3-small | Success@5 | D6/D7 | +0,0023 | 0,7413 | 0,7413 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| text-embedding-3-large | Success@5 | D6/D7 | +0,0177 | 0,0002 | 0,0014 | 0,0022 | 0,0082 | signifikant | signifikant | signifikant |
| multilingual-e5-large | Success@10 | D6/D7 | +0,0023 | 0,6672 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| bge-m3 | Success@10 | D6/D7 | +0,0041 | 0,3370 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| gte-multilingual-base | Success@10 | D6/D7 | -0,0172 | 0,0086 | 0,0516 | 0,0774 | 0,2236 | n. s. | n. s. | n. s. |
| jina-embeddings-v3 | Success@10 | D6/D7 | +0,0073 | 0,0399 | 0,1595 | 0,2792 | 0,8375 | n. s. | n. s. | n. s. |
| text-embedding-3-small | Success@10 | D6/D7 | -0,0023 | 0,6611 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| text-embedding-3-large | Success@10 | D6/D7 | +0,0091 | 0,0118 | 0,0592 | 0,0947 | 0,2960 | n. s. | n. s. | n. s. |
| Qwen3-Embedding-0.6B | MRR@10 | Tabelle 4 | +0,0028 | 0,6866 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| snowflake-arctic-embed-l-v2.0 | MRR@10 | Tabelle 4 | +0,0224 | 0,0005 | 0,0016 | 0,0038 | 0,0167 | signifikant | signifikant | signifikant |
| jina-embeddings-v2-base-de | MRR@10 | Tabelle 4 | -0,0016 | 0,8366 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| Qwen3-Embedding-0.6B | Success@1 | D6/D7 | +0,0005 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| snowflake-arctic-embed-l-v2.0 | Success@1 | D6/D7 | +0,0286 | 0,0019 | 0,0058 | 0,0135 | 0,0521 | signifikant | signifikant | n. s. |
| jina-embeddings-v2-base-de | Success@1 | D6/D7 | -0,0032 | 0,7959 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| Qwen3-Embedding-0.6B | Success@5 | D6/D7 | +0,0018 | 0,8194 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| snowflake-arctic-embed-l-v2.0 | Success@5 | D6/D7 | +0,0095 | 0,0627 | 0,1882 | 0,3765 | 1,0000 | n. s. | n. s. | n. s. |
| jina-embeddings-v2-base-de | Success@5 | D6/D7 | -0,0036 | 0,5603 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| Qwen3-Embedding-0.6B | Success@10 | D6/D7 | +0,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |
| snowflake-arctic-embed-l-v2.0 | Success@10 | D6/D7 | +0,0064 | 0,0925 | 0,2774 | 0,5548 | 1,0000 | n. s. | n. s. | n. s. |
| jina-embeddings-v2-base-de | Success@10 | D6/D7 | -0,0018 | 0,7243 | 1,0000 | 1,0000 | 1,0000 | n. s. | n. s. | n. s. |

Urteilswechsel bei 0,05: Variante A 0 von 36, Variante B 2 von 36.
