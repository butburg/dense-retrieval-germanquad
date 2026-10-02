# Retrieval-Qualität von neun Embedding-Modellen auf dem deutschsprachigen Datensatz GermanQuAD mit BM25-Referenzen

Dieses Repository enthält Code, Notebooks, Tests und Ergebnisdateien eines Retrieval-Experiments auf dem deutschsprachigen Datensatz GermanQuAD. Verglichen werden neun Embedding-Modelle (multilingual-e5-large, bge-m3, gte-multilingual-base, jina-embeddings-v3, text-embedding-3-small, text-embedding-3-large, Qwen3-Embedding-0.6B, snowflake-arctic-embed-l-v2.0 und jina-embeddings-v2-base-de) mit einer BM25-Baseline und BM25-de (BM25 mit deutscher Vorverarbeitung) im Retrieval-only-Setting auf dem GermanQuAD-Testsplit (Cosine-Ähnlichkeit, k = 1, 5, 10, Primärmetrik MRR@10, Seed 42). `compare_models.py` rechnet in einem Lauf 55 gepaarte Tests auf MRR@10 in vier Familien (Holm je Familie, Familiengröße = Testzahl): alle 36 Modellpaare, je Modell gegen BM25 (9), je Modell gegen BM25-de (9) und BM25-de gegen BM25 (1). Success@k und MRR@k werden nur deskriptiv berichtet.

## Inhalt

| Pfad | Inhalt |
|---|---|
| `code/` | Notebooks 01 bis 05, Harness-Skripte (`embed_eval.py`, `compare_models.py`, `make_results.py`, `token_lengths.py`), BM25 (`scripts/run_bm25.py`, `bm25_german.py`), Demo (`demo_query.py`), wiederverwendbare Logik in `code/pylib/`; Details in `code/README.md` |
| `code/output/harness/` | Ergebnisdateien der neun Modelle: Metriken und Per-Query-Ränge je Modell, alle Tests in `comparison_germanquad.json`, Tabellen und Abbildungen (`results/`) |
| `code/output/bm25_de/` | BM25-de (Stoppwörter, Snowball-Stemming): Metriken und Per-Query-Ränge |
| `code/output/germanquad/`, `code/output/bm25_v2/` | Normalisierte GermanQuAD-Daten (Queries, Passagen, Qrels) und BM25-Ergebnisse |
| `code/output/bm25/`, `code/output/dense_e5/` | Pilotwerte (BM25, multilingual-e5-large) |
| `tests/` | pytest-Tests für Metriken, Signifikanztests, Embedder-Registry, Cache, Datensatz-I/O, Ergebnistabellen und Demo |
| `docs/` | Sphinx-API-Dokumentation von `code/pylib/` und das Evaluationsprotokoll (`docs/evaluation_protocol.md`) |

## Setup

Vorausgesetzt wird Python 3.11 (getestet mit 3.11.x).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-Wheel
pip install -r code/requirements-embed.txt                                  # Stack des Hauptexperiments
pip install -r requirements.txt                                             # Notebook- und Sphinx-Umgebung
```

Die Modellgewichte lädt sentence-transformers beim ersten Lauf vom Hugging-Face-Hub (gepinnte Revisionen, gespeichert in `code/output/harness/<Modell>/metrics_germanquad.json`). Die OpenAI-Modelle benötigen den Schlüssel in der Umgebungsvariable `EMBED_OPENAI_API_KEY`.

## Reproduktion über Notebook 05

`code/05_hauptexperiment_germanquad.ipynb` beschreibt den Versuchsaufbau, lädt die persistierten Ergebnisse aus `code/output/harness/`, zeigt die Retrieval-Metriken aller Retriever, die 55 Tests (Modellpaare, Vergleiche gegen BM25 und gegen BM25-de), die Sensitivität bei 512 Token und Abbildung 1 und prüft Kennzahlen und Holm-Korrektur gegen `comparison_germanquad.json`. Mit `ENCODE = SCORE = False` (Standard) läuft es in wenigen Sekunden ohne Netz, GPU und API. `ENCODE = True` berechnet die Vektoren neu (CPU, mehrere Stunden; OpenAI-Läufe benötigen den API-Schlüssel), `SCORE = True` rechnet Ranking und Tests aus dem Cache.

Die Läufe lassen sich auch einzeln über die Skripte starten (im Ordner `code/`):

```bash
python embed_eval.py --dataset-dir output/germanquad --model <e5-large|bge-m3|gte-multilingual-base|jina-v3|openai-3-small|openai-3-large|qwen3-embedding-0.6b|arctic-embed-l-v2|jina-v2-base-de>
python embed_eval.py --dataset-dir output/germanquad --model bge-m3 --max-seq-length 512   # Sensitivität
python scripts/run_bm25.py && python bm25_german.py   # BM25 und BM25-de
python compare_models.py                 # alle 55 Tests in 4 Familien
python make_results.py                   # Tabellen und Abbildungen
```

Die Notebooks 01 bis 04 dokumentieren den Datensatz-PoC (Download, Parsing, BM25- und Dense-Pilot); sie erwarten Rohdaten unter `code/input/`, die per Notebook 01 aus dem Hugging-Face-Hub geladen werden.

## Demo

`code/demo_query.py` zeigt für eine freie Frage (`--query`) oder eine Frage aus dem Datensatz (`--query-id`) die Top-k-Passagen von BM25 und multilingual-e5-large nebeneinander (CPU, kein API-Schlüssel):

```bash
cd code
python demo_query.py --warm-cache                              # einmalig: Passage-Embeddings berechnen
python demo_query.py --model e5-large --query-id q40369 --k 5
```

## Tests

```bash
python -m pytest tests
```

Tests, die Modellgewichte oder den Embedding-Cache benötigen, werden ohne diese Voraussetzungen übersprungen.

## Ergebnisse

Die maßgeblichen Zahlen stehen in `code/output/harness/results/table_main.md` (alle Retriever), `table_significance.md` (alle Tests) und `table_len512.md` (Sensitivität); die Grundlage bilden `code/output/harness/<Modell>/metrics_germanquad.json` und `comparison_germanquad.json`; die Zuordnung der Tabellen zu den Dateien steht in `code/README.md`.

## Sphinx-Dokumentation

```bash
sphinx-build -b html docs docs/_build/html
```

Der Build liest die Docstrings aus `code/pylib/` und schreibt die API-Dokumentation nach `docs/_build/html/`. Die fertig gerenderte Fassung liegt unter `docs/html/index.html` und lässt sich ohne eigenen Build öffnen; zum Neubauen dient der Befehl oben (Voraussetzung: `pip install sphinx myst-parser`).

## Repository aktualisieren

Änderungen an Code oder Ergebnisdateien werden im Clone committet und auf den aktuellen Branch gepusht:

```bash
git status
git add -A
git commit -m "Aktualisierung $(date +%F)"
git push origin "$(git branch --show-current)"
```

## Lizenz

Code und Ergebnisdateien stehen unter der MIT-Lizenz (Copyright 2026 butburg, siehe `LICENSE`). Die unter `code/output/germanquad/` enthaltenen normalisierten GermanQuAD-Daten stammen aus dem Datensatz von deepset; laut den Dataset Cards von `deepset/germanquad` und `mteb/germanquad-retrieval` auf dem Hugging-Face-Hub steht dieser unter der Lizenz CC BY 4.0 (Stand der Abfrage: 30. September 2026). Für diese Daten gilt die dort genannte Datenlizenz.
