# Verbindliches Evaluationsprotokoll (freigegeben 2026-09-30)

Stand: 30. September 2026, 02:25 UTC

Dieses Dokument hält die vom Menschen freigegebenen Festlegungen für den Hauptversuch fest.

## Datensatz
- Ein Datensatz: **GermanQuAD-Retrieval** (Testsplit: 2.204 Fragen, 474 Passagen, je Frage eine relevante Passage), normalisiert in `code/output/germanquad/`.
- GerLeRB wird nicht ausgewertet. Parser-Audit und BM25-Lauf (`code/output/bm25_v2/`) bleiben als Vorarbeit im Repo, fließen aber nicht in die Arbeit ein.

## Retriever
- Baseline: BM25 gemäß dem BM25-Protokoll (Konfiguration in `code/output/bm25_v2/bm25_results.json`).
- Embedder (Dense Retrieval, Dokumentebene, Cosine): intfloat/multilingual-e5-large, BAAI/bge-m3, Alibaba-NLP/gte-multilingual-base, jinaai/jina-embeddings-v3, OpenAI text-embedding-3-small, OpenAI text-embedding-3-large. Modellspezifische Präfixe/Adapter gemäß Modellkarte, dokumentiert in `code/pylib/embedders.py`.

## Metriken
- Recall@k, Success@k, MRR@k, Precision@k für k = 1, 5, 10.
- **Primärmetrik: MRR@10.** Übrige Metriken und k werden berichtet, gelten aber als sekundär/explorativ.

## Signifikanz
- Primärer p-Wert: gepaarter Vorzeichenwechsel-Randomisierungstest auf **Cluster-Ebene** (Cluster = Gold-Passage der Frage), zweiseitig, B = 100.000, fester Seed, p = (Treffer + 1)/(B + 1).
- Success@k: exakter McNemar-Test (b, c) nur ergänzend/deskriptiv, ausdrücklich als „optimistisch, ignoriert Cluster“ gekennzeichnet.
- Effektgröße: Differenz der Mittelwerte mit 95-%-Cluster-Bootstrap-CI (B = 10.000, fester Seed).
- Mehrfachvergleiche: Holm, alpha = 0,05. Konfirmatorische Familie: jeder Embedder gegen BM25 auf der Primärmetrik (6 Tests). Vergleiche der Embedder untereinander und sekundäre Metriken: explorativ, eigene Kennzeichnung.
- Deckeneffekt: Die Schwelle für „nicht aussagekräftig“ (Anzahl diskordanter Paare bzw. Cluster mit Differenz ≠ 0) legt der Student **vor** den Hauptläufen fest, begründet sie in Kapitel 3 und trägt den Wert hier ein; Examiner prüft. Methodische Detailfestlegungen dieser Art prüft und begründet der Student in der Arbeit selbst, sie brauchen keine gesonderte Freigabe des Menschen. Festgelegter Wert: siehe Abschnitt „Detailfestlegungen des Students“.

## Berichtsregeln
- p-Werte roh und Holm-korrigiert, nie nur „signifikant“; Datensatz, k, Seed und Modellversion (Hugging-Face-Revision bzw. API-Modellname und Abrufdatum) nennen.

## Detailfestlegungen des Students (festgelegt vor den Hauptläufen am 2026-09-30)

Die freigegebenen Festlegungen oben bleiben unverändert; dieser Abschnitt konkretisiert sie. Herleitungen sind eigene Rechnungen, keine Quellenaussagen. Die Pilotwerte BM25 vs. E5 (`code/output/analysis/SUMMARY.md`) waren bei der Festlegung bekannt; die Schwelle wurde allein mit Blick auf die Auflösung des Tests gewählt (Selbstauskunft, extern nicht prüfbar; so auch in Kap. 3.4 offengelegt). Überarbeitet am 2026-09-30; Schwellenwert und Folgeregel unverändert.

### Deckeneffekt-Schwelle
- Zählgröße je Vergleich und Metrik: n≠0 = Anzahl der Cluster (Gold-Passagen, max. 474), deren über die Queries summierte Differenz ≠ 0 ist. Ergänzend für Success@k: Zahl diskordanter Query-Paare b + c.
- **Schwelle: n≠0 < 20 → Vergleich gilt als „nicht aussagekräftig“.** Für den deskriptiven McNemar-Wert gilt analog b + c < 20.
- Folge: Der Test bleibt in seiner Holm-Familie, p-Werte (roh, Holm) und Bootstrap-CI werden berichtet, das Ergebnis wird gekennzeichnet und weder als Beleg für einen Unterschied noch als Beleg für Gleichwertigkeit gedeutet. n≠0 wird für jeden Vergleich berichtet.
- Formale Untergrenze (eigene Herleitung): Bei n≠0 Clustern ist der kleinste erreichbare zweiseitige p-Wert des Vorzeichenwechsel-Tests 2/2^n≠0. Die strengste Holm-Stufe der konfirmatorischen Familie ist 0,05/6 ≈ 0,0083; sie ist erst ab n≠0 ≥ 8 überhaupt erreichbar (2/2^7 ≈ 0,0156; 2/2^8 ≈ 0,0078).
- Schwelle 20 als Konvention: Der Wert 20 ist eine vorab gesetzte, konservative Konvention oberhalb der formalen Untergrenze, keine Ableitung. Eine reine Vorzeichenbetrachtung (Binomialtest, Größen ignoriert) verlangt auf der strengsten Holm-Stufe bei n≠0 = 10 alle 10, bei 12 mindestens 11, bei 19 mindestens 16 (p ≈ 0,0044) und bei 20 mindestens 17 Cluster in derselben Richtung (p ≈ 0,0026; 16 von 20: p ≈ 0,0118). Die Anforderung ändert sich stetig mit n; 20 ist kein durch die Rechnung ausgezeichneter Punkt. Die Vorzeichenbetrachtung ist eine Näherung, der verwendete Test nutzt zusätzlich die Größe der Differenzen. Weil n≠0 (und b + c) für jeden Vergleich berichtet wird, können Lesende eine andere Schwelle anlegen.
- Hinweis zur Metrik: Für MRR@10 zählt schon eine Verschiebung um einen Rang als Differenz ≠ 0; n≠0 fällt dort groß aus, die Schwelle bindet nur bei nahezu identischen Rankings. Für Success@k ist b + c das schärfere Maß.

### p-Wert bei kleinem n≠0
- Ist n≠0 ≤ 16 (2^n ≤ 65.536), wird der p-Wert des Vorzeichenwechsel-Tests exakt über alle 2^n Vorzeichenkombinationen berechnet (Anteil der Kombinationen mit |Statistik| ≥ beobachtet), nicht per Monte-Carlo. Grund: Nahe der Holm-Stufe 0,0083 (n = 8: exakt 0,0078) würde der Monte-Carlo-Schätzer streuen. Für n≠0 > 16 gilt Monte-Carlo mit B = 100.000. Die Umsetzung im Analyse-Code liegt beim Code-Agent.

### Seeds
- Randomisierungstest: `numpy.random.default_rng(42)`, B = 100.000. Cluster-Bootstrap: `numpy.random.default_rng(42)`, B = 10.000, 95-%-Perzentil-Intervall. Der Generator wird für jeden Vergleich, jede Metrik und jedes Verfahren neu mit Seed 42 initialisiert, damit Ergebnisse nicht von der Reihenfolge der Vergleiche abhängen (Seed 42 wie im Pilot, `code/analysis_germanquad.py`).
- Monte-Carlo-p-Werte an der Auflösungsgrenze werden als „≤ 1/(B + 1)“ berichtet. Encoding und BM25 haben keine Zufallskomponente; numpy-Version wird protokolliert.

### Modellversionierung
- Hugging-Face-Modelle: Modellgewichte werden über den Hugging-Face-Commit-Hash gepinnt; die Revision von nachgeladenem Remote-Code wird gepinnt bzw. protokolliert (siehe `code/output/harness/VALIDATION.md`). Der Hash wird in der Ergebnis-JSON gespeichert. Der bisher protokollierte Hub-HEAD (`model_revision_hub_head`) genügt nicht, weil er nicht zwingend der geladenen Revision entspricht.
- Datensatz: `mteb/germanquad-retrieval` (Revision 9af36714ebd9bba5235c4b9862779adaddbfae52; corpus, queries) und `mteb/germanquad-retrieval-qrels` (Revision ed232776ce5d736028c1da18e81ab7048969f369; Split test), geladen über `code/01_dataset_download.ipynb`.
- OpenAI-Modelle: angefragter Modellname, von der API zurückgegebener Modellname, Version des `openai`-Pakets, UTC-Zeitstempel der ersten und letzten Anfrage (= Abrufdatum), volle Standarddimension (1536 bzw. 3072, kein `dimensions`-Parameter). Jeder Text wird einmal eingebettet und gecacht; alle Auswertungen nutzen diesen einen Abruf. Ein erneuter Abruf gilt als neuer Lauf mit eigenem Abrufdatum.
- Je Lauf zusätzlich: Python-, sentence-transformers-, transformers-, torch-, numpy- und rank_bm25-Version, Hardware (CPU, Kerne, RAM), Laufzeitstempel.
- Maximale Sequenzlänge: Modellstandard, nicht überschrieben; der Wert wird je Modell protokolliert.

### Eingabelänge (festgelegt 2026-09-30, vor den Hauptläufen; Bead dls-4ko.48)
- Befund: 143 von 474 Passagen haben mehr als 512 Tokens (gezählt mit dem bge-m3-Tokenizer; Median 349, Maximum 2.652; `code/output/harness/VALIDATION.md`, Abschnitt 4). multilingual-e5-large kürzt auf 512 Tokens (Modellgrenze), bge-m3, gte-multilingual-base und jina-embeddings-v3 verarbeiten bis 8.192 Tokens, die OpenAI-Modelle laut Anbieterdokumentation bis 8.192 Tokens (OpenAI-Dokumentation, abgerufen 2026-09-29; Tabelle 1 in Kap. 3; Maximum der Passagen 2.652 Tokens nach bge-m3-Tokenizer, OpenAI-Tokenzahlen nicht gemessen). Längen unter anderen Tokenizern wurden nicht gemessen.
- **Hauptanalyse (konfirmatorisch und explorativ wie oben): native maximale Eingabelänge je Modell** (Modellstandard, siehe Modellversionierung). Begründung: entspricht Modellkarten und praktischer Nutzung; eine künstliche Kürzung würde die Modelle mit längerem Kontext unter ihrem dokumentierten Einsatz evaluieren. Die e5-Grenze ist Eigenschaft des Modells und wird als solche berichtet.
- **Sensitivitätsanalyse (ausschließlich explorativ):** bge-m3, gte-multilingual-base und jina-embeddings-v3 werden zusätzlich mit einheitlich `max_seq_length = 512` (nur Passagen betroffen) gerechnet; multilingual-e5-large ist bereits auf 512 begrenzt und wird nicht erneut gerechnet. OpenAI-Modelle entfallen (kein lokal steuerbarer Tokenizer-Parameter, zusätzliche API-Kosten). Berichtet werden MRR@10 und Success@k je Modell unter beiden Längen sowie deskriptiv die Differenz; optional Cluster-Bootstrap-CI der Differenz nativ vs. 512, gekennzeichnet als explorativ, ohne Holm-Korrektur.
- Die konfirmatorische Familie (6 Tests, MRR@10, Holm) bleibt unverändert und nutzt ausschließlich die Läufe mit nativer Länge. Ergebnisse der Sensitivitätsanalyse ändern keine konfirmatorische Aussage; sie dienen nur der Einordnung, ob Unterschiede zu e5-large mit der Kürzung zusammenhängen können (Diskussion, Abschnitt 5.3).
