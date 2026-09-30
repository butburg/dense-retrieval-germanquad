"""German text preprocessing for lexical retrieval (lowercase, tokenization, stopwords, Snowball stemming).

Components: regex tokenizer identical to the standard BM25 run (``scripts/run_bm25.py``), the German stopword
list of ``stopwordsiso`` (pip, version pinned via ``importlib.metadata``, 620 entries), and the Snowball German
stemmer of ``snowballstemmer`` (the reference implementation used by nltk's ``SnowballStemmer('german')``).
Compound splitting is intentionally omitted: no reviewed decompounding library is available offline/via pip
in this environment (``german-compound-splitter`` not installable; CharSplit is an unreviewed statistical splitter).
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import version

import snowballstemmer
import stopwordsiso

TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÄÖÜäöüß]+")


@lru_cache(maxsize=1)
def german_stopwords() -> frozenset[str]:
    """Return the lowercase German stopword set of stopwordsiso (fixed by the installed package version)."""
    return frozenset(w.lower() for w in stopwordsiso.stopwords("de"))


@lru_cache(maxsize=1)
def _stemmer():
    return snowballstemmer.stemmer("german")


def preprocess_de(text: str | None, stopwords: bool = True, stem: bool = True) -> list[str]:
    """Tokenize German text into index terms.

    Steps in order: lowercase, regex tokenization (letters incl. umlauts/ß and digits), removal of stopwords
    (matched on the unstemmed lowercase token), Snowball German stemming.

    Args:
        text: Raw text; ``None`` is treated as empty.
        stopwords: Remove German stopwords if True.
        stem: Apply the Snowball stemmer if True.

    Returns:
        List of terms in text order (duplicates kept for term-frequency weighting).
    """
    toks = TOKEN_PATTERN.findall((text or "").lower())
    if stopwords:
        sw = german_stopwords()
        toks = [t for t in toks if t not in sw]
    return _stemmer().stemWords(toks) if stem else toks


def preprocessing_info() -> dict:
    """Describe the preprocessing configuration and package versions for result files."""
    return {"steps": ["lowercase", "regex-tokenization", "stopword removal (stopwordsiso, de)", "snowball stemming (german)"],
            "tokenizer_regex": TOKEN_PATTERN.pattern, "n_stopwords": len(german_stopwords()),
            "stopwordsiso_version": version("stopwordsiso"), "snowballstemmer_version": version("snowballstemmer"),
            "compound_splitting": "omitted (no reviewed library available)"}
