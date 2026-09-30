# Sphinx-Dokumentation

Dieses Verzeichnis enthält die Sphinx-Quellen und den Build-Ablauf für die HTML-Dokumentation.

## Voraussetzungen

Installiere Sphinx in der aktiven Python-Umgebung:

```bash
python -m pip install sphinx
```

Optional können zusätzliche Sphinx-Erweiterungen über `docs/conf.py` aktiviert werden.

## Dokumentation erstellen

Führe den Build aus dem Repository-Hauptverzeichnis aus:

```bash
sphinx-build -b html docs docs/_build/html
```

Der Befehl liest die Konfiguration aus `docs/conf.py` und legt die HTML-Dateien unter `docs/_build/html/` ab.

## Dokumentation aufrufen

Öffne die Startseite nach dem Build im Browser:

```bash
open docs/_build/html/index.html
```

Alternativ kann ein lokaler HTTP-Server verwendet werden:

```bash
python -m http.server --directory docs/_build/html 8000
```

Rufe anschließend `http://localhost:8000` im Browser auf.

## Struktur

`docs/index.rst` ist die Einstiegseite und bindet Dokumentationsseiten über die `toctree` ein.

Weitere `.rst`-Dateien definieren Inhalt, Gliederung und Verweise der Dokumentation.

`docs/conf.py` definiert Projektname, Python-Pfad, aktivierte Sphinx-Erweiterungen, Type-Hint-Darstellung und HTML-Theme.

Falls autodoc verwendet wird, müssen benötigte Importpfade und Erweiterungen dort gepflegt werden.

Die Erweiterung `sphinx.ext.autodoc` liest Module und Docstrings automatisch aus dem Python-Code.

Die Erweiterung `sphinx.ext.napoleon` übersetzt Google-Style-Docstrings in Sphinx-Strukturen.

## Inhalte erweitern

Ergänze oder passe `.rst`-Dateien und die Sphinx-Konfiguration an, wenn neue Seiten, Erweiterungen oder Autodoc-Einbindungen hinzukommen.

Führe danach den Sphinx-Build erneut aus, damit die HTML-Seiten aktualisiert werden.
……
`docs/_build/` enthält generierte Dateien und bleibt durch `docs/.gitignore` aus der Versionsverwaltung ausgeschlossen.
