# Dense Retrieval mit sechs Embedding-Modellen im Vergleich zu BM25 auf GermanQuAD

Dieses Repository enthält Code, Notebooks, Tests und Ergebnisdateien eines Retrieval-Experiments auf dem deutschsprachigen Datensatz GermanQuAD. Verglichen werden sechs Embedding-Modelle (multilingual-e5-large, bge-m3, gte-multilingual-base, jina-embeddings-v3, text-embedding-3-small und text-embedding-3-large) mit einer BM25-Baseline im Retrieval-only-Setting auf dem GermanQuAD-Testsplit (Cosine-Ähnlichkeit, k = 1, 5, 10, Primärmetrik MRR@10, Seed 42).

## Inhalt

| Pfad | Inhalt |
|---|---|
| `code/` | Notebooks 01 bis 05, Harness-Skripte (`embed_eval.py`, `compare_models.py`, `make_results.py`, `token_lengths.py`), BM25 (`scripts/run_bm25.py`), Demo (`demo_query.py`), wiederverwendbare Logik in `code/pylib/`; Details in `code/README.md` |
| `code/output/harness/` | Ergebnisdateien des Hauptexperiments: Metriken und Per-Query-Ränge je Modell, `comparison_germanquad.json`, Tabellen und Abbildung 1 (`results/`), `VALIDATION.md` |
| `code/output/germanquad/`, `code/output/bm25_v2/` | Normalisierte GermanQuAD-Daten (Queries, Passagen, Qrels) und BM25-Ergebnisse |
| `code/output/analysis/`, `code/output/repro/`, `code/output/bm25/`, `code/output/dense_e5/` | Pilotanalyse und Reproduktionsläufe (BM25, multilingual-e5-large) |
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

Die Modellgewichte lädt sentence-transformers beim ersten Lauf vom Hugging-Face-Hub (gepinnte Revisionen, siehe `code/output/harness/VALIDATION.md`). Die OpenAI-Modelle benötigen den Schlüssel in der Umgebungsvariable `EMBED_OPENAI_API_KEY`.

## Reproduktion über Notebook 05

`code/05_hauptexperiment_germanquad.ipynb` beschreibt den Versuchsaufbau, lädt die persistierten Ergebnisse aus `code/output/harness/`, zeigt Tabelle 2 (Retrieval-Metriken), Tabelle 3 (konfirmatorische Tests gegen BM25), Paarvergleiche, Tabelle 4 (Sensitivität bei 512 Token) und Abbildung 1 und prüft die Kennzahlen gegen `comparison_germanquad.json`. Mit `RUN_EXPERIMENTS = False` (Standard) läuft es in wenigen Sekunden ohne Netz, GPU und API. Mit `RUN_EXPERIMENTS = True` führt es die Embedding-Läufe neu aus (CPU, mehrere Stunden; OpenAI-Läufe benötigen den API-Schlüssel).

Die Läufe lassen sich auch einzeln über die Skripte starten (im Ordner `code/`):

```bash
python embed_eval.py --dataset-dir output/germanquad --model <e5-large|bge-m3|gte-multilingual-base|jina-v3|openai-3-small|openai-3-large>
python embed_eval.py --dataset-dir output/germanquad --model bge-m3 --max-seq-length 512   # Sensitivität
python scripts/run_bm25.py
python compare_models.py
python make_results.py
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

Die maßgeblichen Zahlen stehen in `code/output/harness/results/table_main.md` (Tabelle 2), `table_significance.md` (Tabelle 3) und `table_len512.md` (Tabelle 4); die Grundlage bilden `code/output/harness/<Modell>/metrics_germanquad.json` und `comparison_germanquad.json`. Die Validierung der Läufe (Stack, Revisionen, Konsistenzprüfungen) steht in `code/output/harness/VALIDATION.md`.

## Sphinx-Dokumentation

```bash
sphinx-build -b html docs docs/_build/html
```

Der Build liest die Docstrings aus `code/pylib/` und schreibt die API-Dokumentation nach `docs/_build/html/`.

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
