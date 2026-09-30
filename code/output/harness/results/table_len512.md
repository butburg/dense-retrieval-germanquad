# Sensitivität Eingabelänge (explorativ, ohne Holm)

nativ vs. max_seq_length = 512 (nur Passagen); CI: 95-%-Cluster-Bootstrap der Differenz (B = 10000, Seed 42); keine p-Werte.

| Modell | Metrik | nativ | 512 | Differenz (512 − nativ) | 95%-CI |
|---|---|---|---|---|---|
| bge-m3 | MRR@10 | 0,946 | 0,929 | -0,016 | [-0,025; -0,009] |
| bge-m3 | Success@1 | 0,917 | 0,895 | -0,022 | [-0,033; -0,012] |
| bge-m3 | Success@5 | 0,980 | 0,972 | -0,009 | [-0,015; -0,002] |
| bge-m3 | Success@10 | 0,989 | 0,982 | -0,007 | [-0,012; -0,002] |
| gte-multilingual-base | MRR@10 | 0,886 | 0,896 | +0,010 | [+0,004; +0,017] |
| gte-multilingual-base | Success@1 | 0,834 | 0,845 | +0,011 | [+0,004; +0,019] |
| gte-multilingual-base | Success@5 | 0,952 | 0,964 | +0,012 | [+0,004; +0,022] |
| gte-multilingual-base | Success@10 | 0,967 | 0,975 | +0,008 | [+0,001; +0,016] |
| jina-embeddings-v3 | MRR@10 | 0,943 | 0,930 | -0,013 | [-0,019; -0,006] |
| jina-embeddings-v3 | Success@1 | 0,907 | 0,892 | -0,016 | [-0,026; -0,006] |
| jina-embeddings-v3 | Success@5 | 0,986 | 0,979 | -0,008 | [-0,013; -0,003] |
| jina-embeddings-v3 | Success@10 | 0,992 | 0,987 | -0,005 | [-0,010; -0,000] |

Fehlende Läufe: keine
