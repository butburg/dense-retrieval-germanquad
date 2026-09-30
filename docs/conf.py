from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../code"))

project = "deeplearningSearch"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_typehints = "signature"

html_theme = "alabaster"
