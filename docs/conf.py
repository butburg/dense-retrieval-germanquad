from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../code"))

project = "deeplearningSearch"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
try:  # Markdown-Seiten (Evaluationsprotokoll) nur, wenn myst-parser installiert ist
    import myst_parser  # noqa: F401
    extensions.append("myst_parser")
except ImportError:
    pass

autodoc_mock_imports = ["torch", "sentence_transformers", "openai", "transformers"]
html_show_sourcelink = False
html_copy_source = False

autodoc_typehints = "signature"

html_theme = "alabaster"

# Interne Markdown-Dateien gehören nicht in die HTML-Doku
exclude_patterns = ["_build", "html", "README.md", "AGENT_ARCHITECTURE.md"]
