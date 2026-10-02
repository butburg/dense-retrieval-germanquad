# Evaluationsprotokoll

Dieses Dokument beschreibt Datensatz, Retriever, Metriken und statistische Tests des Hauptexperiments. Die Umsetzung liegt in `code/embed_eval.py`, `code/compare_models.py` und `code/pylib/`.

## Datensatz
- GermanQuAD-Retrieval, Testsplit: 2.204 Fragen, 474 Passagen, je Frage eine relevante Passage; normalisiert in `code/output/germanquad/`.
- Quellen: `mteb/germanquad-retrieval` (Revision 9af36714ebd9bba5235c4b9862779adaddbfae52; corpus, queries) und `mteb/germanquad-retrieval-qrels` (Revision ed232776ce5d736028c1da18e81ab7048969f369; Split test), geladen über `code/01_dataset_download.ipynb`.

## Retriever
- Baselines: BM25 (`code/scripts/run_bm25.py`, Ausgabe `code/output/bm25_v2/`) und BM25-de (`code/bm25_german.py`; Stoppwörter aus stopwordsiso, Snowball-Stemmer).
- Embedding-Modelle (Dense Retrieval, Dokumentebene, Kosinus-Ähnlichkeit): intfloat/multilingual-e5-large, BAAI/bge-m3, Alibaba-NLP/gte-multilingual-base, jinaai/jina-embeddings-v3, OpenAI text-embedding-3-small, OpenAI text-embedding-3-large, Qwen/Qwen3-Embedding-0.6B, Snowflake/snowflake-arctic-embed-l-v2.0, jinaai/jina-embeddings-v2-base-de. Alle neun Modelle werden gleich behandelt. Modellspezifische Präfixe und Adapter folgen der Modellkarte und stehen in `code/pylib/embedders.py`.

## Metriken
- Recall@k, Success@k, MRR@k und Precision@k für k = 1, 5, 10.
- Primärmetrik: MRR@10. Getestet wird nur MRR@10; die übrigen Metriken und k werden deskriptiv berichtet.

## Signifikanztests
- Gepaarter Vorzeichenwechsel-Randomisierungstest auf Cluster-Ebene (Cluster = Gold-Passage der Frage), zweiseitig, B = 100.000, p = (Treffer + 1)/(B + 1). Bei n≠0 ≤ 16 Clustern mit Differenz ≠ 0 wird der p-Wert exakt über alle 2^n Vorzeichenkombinationen berechnet. Monte-Carlo-p-Werte an der Auflösungsgrenze werden als „≤ 1/(B + 1)“ berichtet.
- Effektgröße: Differenz der Mittelwerte mit 95-%-Perzentil-Intervall aus einem Cluster-Bootstrap (B = 10.000).
- Seeds: `numpy.random.default_rng(42)`, für jeden Vergleich und jedes Verfahren neu initialisiert, damit das Ergebnis nicht von der Reihenfolge der Vergleiche abhängt.
- Testfamilien mit Holm-Korrektur je Familie (Familiengröße = Testzahl, alpha = 0,05): F1 alle 36 Paare der neun Modelle; F2 neun Modelle gegen BM25; F3 neun Modelle gegen BM25-de; F4 BM25-de gegen BM25 (ein Test, unkorrigiert). Zusammen 55 Tests, keine Korrektur über Familien hinweg.
- Berichtet werden p-Werte roh und Holm-korrigiert, Dataset, k, Seed und Modellversion.
- Deckeneffekt: Beträgt n≠0 weniger als 20 (bei Success@k analog b + c < 20), gilt ein Vergleich als „nicht aussagekräftig“. Der Test bleibt in seiner Familie, n≠0 wird für jeden Vergleich ausgegeben.
- Ergebnisdateien: `code/output/harness/comparison_germanquad.json`, `code/output/harness/results/table_significance.md`.

## Modellversionierung
- Hugging-Face-Modelle: Gewichte über den Commit-Hash gepinnt, Revision nachgeladenen Remote-Codes gepinnt; beides steht in `metrics_germanquad.json` des jeweiligen Durchlaufs.
- OpenAI-Modelle: angefragter und zurückgegebener Modellname, Version des `openai`-Pakets, UTC-Zeitstempel der ersten und letzten Anfrage, volle Standarddimension (1536 bzw. 3072). Jeder Text wird einmal eingebettet und gecacht; ein erneuter Abruf ist ein neuer Lauf. API-Vektoren sind nicht bitgenau reproduzierbar.
- Je Lauf protokolliert: Python-, sentence-transformers-, transformers-, torch-, numpy- und rank_bm25-Version, Hardware und Laufzeit.

## Eingabelänge
- 143 von 474 Passagen haben mehr als 512 Tokens (bge-m3-Tokenizer; Median 349, Maximum 2.652; `code/output/harness/token_lengths.json`, erzeugt von `code/token_lengths.py`).
- Hauptanalyse: native maximale Eingabelänge je Modell (Modellstandard, `max_seq_length` nicht überschrieben, je Modell protokolliert). multilingual-e5-large kürzt auf 512 Tokens (Modellgrenze).
- Sensitivitätsanalyse (deskriptiv, ohne Test): bge-m3, gte-multilingual-base und jina-embeddings-v3 zusätzlich mit `max_seq_length = 512` (`--max-seq-length 512`, Durchläufe `<Modell>__len512`, Tabelle `code/output/harness/results/table_len512.md`). Alle 55 Tests nutzen nur die Läufe mit nativer Länge.
