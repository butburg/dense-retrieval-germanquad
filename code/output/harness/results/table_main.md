# Hauptergebnisse

| Retriever | MRR@10 | Success@1 | Success@5 | Success@10 | MRR@5 | Modellrevision / API-Modell |
|---|---|---|---|---|---|---|
| BM25 | 0,884 | 0,838 | 0,948 | 0,965 | 0,882 | rank_bm25 (k1=1.5, b=0.75), output/bm25_v2/per_query_ranks.csv |
| multilingual-e5-large | 0,933 | 0,898 | 0,979 | 0,987 | 0,932 | intfloat/multilingual-e5-large@3d7cfbdacd47 |
| bge-m3 | 0,946 | 0,917 | 0,980 | 0,989 | 0,944 | BAAI/bge-m3@5617a9f61b02 |
| gte-multilingual-base | 0,886 | 0,834 | 0,952 | 0,967 | 0,884 | Alibaba-NLP/gte-multilingual-base@9bbca17d9273 |
| jina-embeddings-v3 | 0,943 | 0,907 | 0,986 | 0,992 | 0,942 | jinaai/jina-embeddings-v3@ab036b023d30 |
| text-embedding-3-small | 0,917 | 0,873 | 0,974 | 0,982 | 0,916 | text-embedding-3-small (API, abgerufen 2026-09-30) |
| text-embedding-3-large | 0,944 | 0,908 | 0,990 | 0,994 | 0,944 | text-embedding-3-large (API, abgerufen 2026-09-30) |

GermanQuAD (Test), n = 2204 Queries, 474 Cluster (Gold-Passagen), Corpus 474 Passagen; Primärmetrik MRR@10, alle Werte auf dem vollen Ranking, Cosine-Similarity (BM25: BM25Okapi). Modelle mit nativer Eingabelänge. Fehlende Läufe: keine
