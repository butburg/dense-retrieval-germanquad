# Code-Bereich: Datensatz-PoC und Hauptexperiment

Der Code-Bereich enthält zwei Teile: den abgeschlossenen Datensatz-PoC (Notebooks 01 bis 04), der Rohdaten in ein einheitliches Retrieval-Eingabeformat überführt und die Auswertung an einem ersten BM25- und Dense-Lauf erprobt, und das Hauptexperiment (Notebook 05 plus Harness-Skripte), das neun Embedding-Modelle, BM25 und BM25-de auf GermanQuAD nach `docs/evaluation_protocol.md` vergleicht.

## Teil 1: Datensatz-PoC (01 bis 04, abgeschlossen)

Zweck: Die Quelldaten werden lokal reproduzierbar eingelesen, geparst und in Rollen (`queries`, `docs`, `qrels`) normalisiert; Retrieval-Metriken laufen unter identischen Cutoffs `k = [1, 5, 10]`.

`01_dataset_download.ipynb` materialisiert externe Quellen nach `code/input/` oder verwendet vorhandene lokale Rohdateien wieder und validiert deren Verfügbarkeit.

`02_dataset_parsing_inspection.ipynb` parst die Dateiformate aus `code/input/`, normalisiert beide Datensätze auf eine gemeinsame Struktur, führt Basisprüfungen aus und schreibt Ergebnisse unter `code/output/`.

`03_bm25_retrieval_poc.ipynb` bewertet die normalisierten Datensätze mit einer deterministischen BM25-Baseline (Ausgabe: `code/output/bm25/`).

`04_dense_retrieval_poc.ipynb` führt denselben Retrieval-only-Ablauf für `intfloat/multilingual-e5-large` auf `GermanQuAD` aus (Ausgabe: `code/output/dense_e5/`).

Pilotwerte (BM25 und e5-large, Pilotanalyse in `code/output/analysis/`): Sie dienten der Prüfung der Pipeline; die maßgeblichen Zahlen liefert das Hauptexperiment. Reproduktion der Pilotwerte: `repro_germanquad.py` und `repro_compare.py`, Dokumentation in `code/output/repro/REPRO.md`.

## Teil 2: Hauptexperiment (05 und Harness-Skripte)

`05_hauptexperiment_germanquad.ipynb` ist in sieben Schritte gegliedert (0 Überblick und Glossar, 1 Daten, 2 Encoding, 3 Ranking und Kennzahlen, 4 Signifikanztests, 5 Sensitivität 512 Tokens, 6 BM25 Standard und BM25-de, 7 Zusammenfassung und Konsistenzprüfung); jeder Schritt beginnt mit „Was? Warum? Wie?“. Die Modelle sind vortrainiert: Das Encoding (474 Passagen und 2.204 Fragen zu Vektoren, Stunden auf CPU) ist der langsame Teil, das Ranking per Kosinus aus dem Cache dauert Sekunden. Zwei Schalter steuern die Neuberechnung: `ENCODE` (Vektoren aller neun Modelle berechnen, Cache) und `SCORE` (Ranking, BM25, BM25-de, Kennzahlen und alle Tests aus dem Cache); beide stehen standardmäßig auf `False`, dann lädt das Notebook nur gespeicherte Ergebnisse (wenige Sekunden, ohne Netz). `ONLY = [...]` beschränkt die Läufe auf einzelne Modellschlüssel; die Dauer jedes gestarteten Schritts steht in `output/harness/run_log.jsonl`, die Encodierzeiten in `encode_germanquad.json` bzw. `metrics_germanquad.json`. Das Notebook liest die normalisierten Dateien in `code/output/germanquad/`; die Notebooks 01 und 02 sind nur nötig, um sie aus den Rohdaten neu zu erzeugen. Es zeigt Tabelle 3 (Qualität), Tabelle 4 (Tests gegen BM25), Paarvergleiche, Tabelle 5 (BM25-de), Tabelle D4 (512-Sensitivität) und Abbildung 1 und prüft Kennzahlen und Holm-Korrektur gegen `comparison_germanquad.json`.

Retriever-Registry (`pylib/embedders.py`): `e5-large` (intfloat/multilingual-e5-large), `bge-m3` (BAAI/bge-m3), `gte-multilingual-base` (Alibaba-NLP/gte-multilingual-base), `jina-v3` (jinaai/jina-embeddings-v3), `openai-3-small` und `openai-3-large` (text-embedding-3-*, Key aus `EMBED_OPENAI_API_KEY`); dazu BM25 (`scripts/run_bm25.py`, nur GermanQuAD, Ausgabe `output/bm25_v2/`) und BM25-de (`bm25_german.py`, Stoppwörter und Snowball-Stemming, Ausgabe `output/bm25_de/`). Ebenfalls in der Registry stehen `qwen3-embedding-0.6b` (Qwen/Qwen3-Embedding-0.6B), `arctic-embed-l-v2` (Snowflake/snowflake-arctic-embed-l-v2.0) und `jina-v2-base-de` (jinaai/jina-embeddings-v2-base-de); alle neun Modelle laufen gleich (Ausgabe `output/harness/<model>/`).

Testfamilien: `compare_models.py` rechnet in einem Lauf 55 Tests auf MRR@10 mit demselben Cluster-Randomisierungstest (Holm je Familie, Familiengröße = Testzahl): F1 alle 36 Paare der neun Modelle (`MODELS`), F2 je Modell gegen BM25 (9), F3 je Modell gegen BM25-de (9), F4 BM25-de gegen BM25 (1, ohne Korrektur). Success@k und MRR@k erscheinen nur deskriptiv in `table_main.md`. `table_main.md` (Tabelle 3) und Abbildung 1 zeigen alle elf Retriever; `tests/test_comparison_design.py` prüft Testzahl, Familiengrößen und Holm-Werte der JSON.

Harness-Skripte:
- `embed_eval.py`: ein Embedder je Aufruf; `--max-seq-length 512` für die Sensitivität (Ausgabe `<model>__len512`).
- `compare_models.py`: alle gepaarten Tests (Cluster-Sign-Flip, Cluster-Bootstrap, Holm je Familie) nach `comparison_germanquad.json`.
- `make_results.py`: Tabellen, Abbildung 1, Paarmatrix der MRR@10-Differenzen (`fig_pair_matrix.png`, ohne Tests) und Ablaufschema der Tests (`fig_test_flow.png`) nach `output/harness/results/`.
- `token_lengths.py`: Tokenlängen der Passagen nach `output/harness/token_lengths.json`.

Ausgaben unter `output/harness/`: `<model>/metrics_germanquad.json` und `per_query_germanquad.csv` je Modell, `comparison_germanquad.json`, `results/` (`table_main`, `table_significance`, `table_len512`, `fig_mrr10_ci`, `fig_pair_matrix`, `fig_test_flow`), `token_lengths.json`, `VALIDATION.md` (Validierung, Stack, Revisionen). Der Embedding-Cache `output/harness/cache/` ist git-ignoriert.

Reproduktion (im Ordner `code/`, Pakete aus `requirements-embed.txt`):

```
python embed_eval.py --dataset-dir output/germanquad --model <key> --stage encode   # langsam: Vektoren in den Cache
python embed_eval.py --dataset-dir output/germanquad --model <key> --stage score    # schnell: Ranking und Kennzahlen aus dem Cache (ohne --stage: beides)
python embed_eval.py --dataset-dir output/germanquad --model <bge-m3|gte-multilingual-base|jina-v3> --max-seq-length 512
python scripts/run_bm25.py && python bm25_german.py    # BM25 und BM25-de
python compare_models.py                                # alle 55 Tests
python make_results.py
```

Details zur Validierung: `output/harness/VALIDATION.md`.

## Ergebnisdateien je Tabelle

Die Tabellen und Abbildung 1 der schriftlichen Ausarbeitung stammen aus folgenden Dateien (Pfade relativ zum Repository):

| Tabelle / Abbildung | Inhalt | Ergebnisdatei |
|---|---|---|
| Tabelle 3, Tabelle D1 | Metriken aller Retriever | `code/output/harness/<Durchlauf>/metrics_germanquad.json` (neun Durchläufe, Schlüssel aus `pylib/embedders.py`), BM25 `code/output/bm25_v2/bm25_results.json`, BM25-de `code/output/bm25_de/bm25_results.json`; aufbereitet in `results/table_main.md` |
| Tabellen 4 und 5, D2, D3, D5, D6 | alle Tests (55 in 4 Familien) | `code/output/harness/comparison_germanquad.json`, aufbereitet in `code/output/harness/results/table_significance.md` |
| Tabelle D4 | Sensitivität 512 Tokens | Durchläufe `bge-m3__len512`, `gte-multilingual-base__len512`, `jina-v3__len512`; aufbereitet in `code/output/harness/results/table_len512.md` |
| Abbildung 1 | MRR@10 mit 95-%-Konfidenzintervall | `code/output/harness/results/fig_mrr10_ci.png` |
| Abbildung 2 | Ablauf der gepaarten Tests | `code/output/harness/results/fig_test_flow.png` |
| Abbildung D1 | MRR@10-Differenzen aller 55 Paare | `code/output/harness/results/fig_pair_matrix.png` |

Per-Query-Ränge liegen als `per_query_germanquad.csv` in den Verzeichnissen der Durchläufe bzw. als `per_query_ranks.csv` unter `code/output/bm25_v2/` und `code/output/bm25_de/`. Tokenlängen der Passagen stehen in `code/output/harness/token_lengths.json`, Validierung und Reproduktion in `code/output/harness/VALIDATION.md` und `code/output/repro/REPRO.md`.

MTEB-Abgleich: Das Referenzergebnis `results/intfloat__multilingual-e5-large/ab10c1a7…/GermanQuAD-Retrieval.json` aus dem Repository embeddings-benchmark/results wird mit `code/output/harness/e5-large/metrics_germanquad.json` verglichen, entsprechend `results/Snowflake__snowflake-arctic-embed-l-v2.0/edc2df7b…/GermanQuAD-Retrieval.json` mit `code/output/harness/arctic-embed-l-v2/metrics_germanquad.json`. Die Datensatzrevision der MTEB-Aufgabe ist `f5c87ae5`, die eigene `9af36714`.

Encoding-Zeiten der Durchläufe mit nativer Eingabelänge (Dokumente und Queries ohne Cache-Treffer, Feld `encode_seconds` in `metrics_germanquad.json`): multilingual-e5-large 464,6 s, bge-m3 766,2 s, gte-multilingual-base 317,0 s, jina-embeddings-v3 1.751,0 s, text-embedding-3-small 21,3 s, text-embedding-3-large 25,1 s (OpenAI-API), snowflake-arctic-embed-l-v2.0 613,0 s, Qwen3-Embedding-0.6B 1.240,5 s, jina-embeddings-v2-base-de 405,4 s.

## GermanQuAD: Eingangsdateien und Felder

GermanQuAD wird lokal über drei JSONL-Dateien verarbeitet: `queries.jsonl`, `corpus.jsonl` und `qrels.jsonl` unter `code/input/germanquad/`. Die Query-Datei liefert eine Anfrage-ID und den zugehörigen Query-Text, die Corpus-Datei liefert eine Dokument-ID und den Dokumenttext, und die Qrels-Datei liefert Zuordnungen zwischen Query-ID und Dokument-ID samt Relevanzwert.

## GerLeRB: Eingangsdateien und Metadaten

GerLeRB erwartet `topics.txt`, `qrels.txt` und `corpus.trec.gz` unter `code/input/gerlerb/`. `topics.txt` und `qrels.txt` werden als TSV gelesen, während `corpus.trec.gz` als gzip-komprimiertes TREC-XML-ähnliches Zeilenformat geparst wird. Die Datei `zenodo_record_15745124.json` enthält technische Zenodo-Metadaten zur Quelle und dokumentiert die Nachvollziehbarkeit von Datensatzherkunft und Dateistruktur.

## Parsing der Formate

JSONL wird zeilenweise als einzelne JSON-Objekte gelesen und in Python-Records überführt. TSV wird über Tab-Spaltennamen in strukturierte Dict-Zeilen geparst. TREC in GZIP wird als Textstrom geöffnet und über `<DOC>`, `<DOCNO>`, `<TEXT...>` und `</DOC>` in Dokumentobjekte überführt, wobei Parser-Differenzen separat protokolliert werden.

## Gemeinsame Zielstruktur

Die Normalisierung erzeugt drei Rollen mit stabilen Schlüsseln: `queries`, `docs` und `qrels`. `queries` enthält `query_id` und `query_text`, `docs` enthält `doc_id` und `text`, und `qrels` enthält `query_id`, `doc_id` und `relevance`. Diese Struktur ist bewusst datensatzübergreifend gleich, damit spätere Retriever auf identischen Schnittstellen arbeiten.

## Warum der erste Dense-Pilot so aufgebaut ist

Der erste Dense-Pilot beantwortet noch nicht die gesamte Forschungsfrage nach dem besten deutschsprachigen Embedding-Modell. Er prüft zuerst, ob ein deutsch-taugliches Dense-Modell unter denselben lokalen Bedingungen wie die BM25-Baseline reproduzierbar evaluiert werden kann.

`GermanQuAD` ist dafür der erste Datensatz, weil er klein, technisch kontrollierbar und bereits im BM25-PoC sauber durchlaufen ist. `GerLeRB` bleibt methodisch wichtig, wird aber für Dense erst nach dem ersten stabilen Pilotlauf nachgezogen, damit Parser- und Goldstandardfragen die erste Modellintegration nicht mit einer zweiten Unsicherheit vermischen.

Das Notebook arbeitet auf Dokument-Ebene statt auf Chunk-Ebene, weil die aktuellen qrels auf `doc_id` verweisen. Ein frühes Chunking würde neue Retrieval-Einheiten schaffen und damit sofort eine zusätzliche Goldstandard- und Aggregationsentscheidung erzwingen.

Die Dense-Auswertung verwendet dieselben `k`-Werte und dieselbe Evaluationslogik wie der BM25-PoC. Dadurch bleibt sichtbar, ob Unterschiede später vom Retriever kommen oder nur aus veränderten Cutoffs, Metriken oder Datenpfaden entstehen. Erst wenn dieser erste Dense-Lauf stabil und nachvollziehbar funktioniert, lohnt sich der kontrollierte Austausch weiterer Embedding-Modelle.

## Embedding-Harness (`embed_eval.py`)

Der Harness bewertet Dense-Retriever auf jedem normalisierten Datensatz (Retrieval only, Cosine, k=1/5/10).

- Eingabe: `--dataset-dir` mit `{docs,queries,qrels}.normalized.jsonl`, `--model <key>` aus der Registry `pylib/embedders.py` (`e5-large`, `bge-m3`, `gte-multilingual-base`, `jina-v3`, `openai-3-small`, `openai-3-large`, `qwen3-embedding-0.6b`, `arctic-embed-l-v2`, `jina-v2-base-de`; Backend sentence-transformers oder OpenAI mit Key aus `EMBED_OPENAI_API_KEY`), optional `--max-seq-length`, `--out-dir` (Default `output/harness`).
- Verarbeitung: Präfixe, Tasks und Normalisierung je Modellkonfiguration, Embeddings als `.npy` gecacht, Ranking und Metriken aus `pylib/retrieval_metrics.py`.
- Ausgabe: `output/harness/<run_name>/metrics_<dataset>.json` (Metriken, Konfiguration, Versionen), `per_query_<dataset>.csv` (Rang des ersten relevanten Docs), Cache in `output/harness/cache/` (git-ignoriert). Validierung: `output/harness/VALIDATION.md`.

## Demo-Skript (`demo_query.py`)

Zeigt für eine freie Frage (`--query`) oder eine Frage aus dem Datensatz (`--query-id`, mit `[GOLD]`-Markierung und Gold-Rang) die Top-k von BM25 und einem offenen Embedding-Modell nebeneinander. BM25 kommt aus `scripts/run_bm25.py`, Embedding und Cache aus `pylib/embedders.py` und `embed_eval.py`; nur CPU, kein API-Key.

- Einrichtung: venv mit `requirements-embed.txt` (torch CPU), Modellgewichte per Netz laden (gepinnte Revision).
- Cache füllen (einmalig, 11 Min. auf 4 CPU-Kernen): `python demo_query.py --warm-cache` (Cache unter `output/harness/cache/`, git-ignoriert).
- Aufruf: `python demo_query.py --model e5-large --query-id q40369 --k 5`. Das Laden des Modells und die Query-Kodierung dauern beim ersten Aufruf rund 12 s, danach bei `--interactive` kürzer.
- Prüfung: `tests/test_demo_query.py` vergleicht für 25 Query-IDs Gold-Rang und BM25-Top-10 mit den gespeicherten per-query-CSVs.
- Hinweis: `HF_HUB_OFFLINE=1` scheitert mit sentence-transformers 5.7 beim Laden des Modells (Fehler „Unrecognized processing class“); die Demo braucht daher Netzzugang zum Hugging-Face-Hub.

## Outputs unter code/output

Die normalisierten Ergebnisse werden je Datensatz nach `code/output/germanquad/` und `code/output/gerlerb/` geschrieben (die GerLeRB-Ausgaben sind wegen ihres Umfangs von rund 130 MB nicht Teil des Abgabe-Repositorys und lassen sich mit Notebook 02 neu erzeugen). Pro Datensatz entstehen `queries.normalized.jsonl`, `docs.normalized.jsonl`, `qrels.normalized.jsonl` sowie eine `summary.json` mit Counts und Konsistenzhinweisen. Sobald eine JSONL-Ausgabe `57.000` Zeilen erreicht, schreibt der Helper automatisch in fortlaufende Dateien wie `docs.normalized.part2.jsonl` weiter. Zusätzlich fasst `code/output/dataset_inspection_summary.json` zentrale Rohbefunde und Parser-Differenzen beider Datensätze zusammen.

Die Retrieval-Notebooks schreiben ihre Laufartefakte getrennt nach Retrievertyp. Der BM25-PoC exportiert nach `code/output/bm25/`, während der erste Dense-Pilot seine Kennzahlen und Beispielrankings nach `code/output/dense_e5/` schreibt. Diese Trennung hält Konfiguration, Modellname und Ergebnisdateien pro Retriever nachvollziehbar auseinander.

## Ignore-Regeln und Versionierung

Die `.gitignore`-Regeln fokussieren Commits auf reproduzierbare Verarbeitungsergebnisse: `code/input/.gitkeep` hält die Verzeichnisstruktur stabil, während lokale Rohdaten unter `code/input/**`, Download-Caches unter `code/input/hf_cache/` und `zenodo_record_15745124.json` lokal verwaltet werden. Für `code/output/` gibt es derzeit eine offene Versionierungsmöglichkeit für normalisierte Ergebnisdateien.

## Ausführungsreihenfolge und Verhalten bei fehlenden Dateien

Die vorgesehene Reihenfolge ist erst `01_dataset_download.ipynb`, danach `02_dataset_parsing_inspection.ipynb` und anschließend ein Retriever-Notebook wie `03_bm25_retrieval_poc.ipynb` oder `04_dense_retrieval_poc.ipynb`; das Hauptexperiment `05_hauptexperiment_germanquad.ipynb` liest die persistierten Harness-Ergebnisse. Falls Eingabedateien fehlen, meldet das Download-Notebook den manuellen Bedarf für lokale Rohdateien, während die nachgelagerten Notebooks mit einer klaren `FileNotFoundError`-Meldung stoppen und den Ergebnisstand pro Datensatz konsistent halten.
