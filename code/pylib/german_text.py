"""German text preprocessing for lexical retrieval (lowercase, tokenization, stopwords, Snowball stemming).

Components: regex tokenizer ``tokenize_de`` (shared with the standard BM25 run ``scripts/run_bm25.py``), the German stopword
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


def tokenize_de(text: str | None) -> list[str]:
    """Lowercase a text and split it into letter/digit tokens.

    Args:
        text: Raw text, e.g. ``"Käse, ÄPFEL & 42!"``. ``None`` counts as empty.

    Returns:
        Lowercase tokens in text order (umlauts and ß count as letters), e.g.
        ``["käse", "äpfel", "42"]``. Duplicates are kept.
    """
    return TOKEN_PATTERN.findall((text or "").lower())


@lru_cache(maxsize=1)
def german_stopwords() -> frozenset[str]:
    """Return the lowercase German stopword set of stopwordsiso.

    The content is fixed by the installed package version (see :func:`preprocessing_info`).
    The set is built once and cached.
    """
    return frozenset(word.lower() for word in stopwordsiso.stopwords("de"))


@lru_cache(maxsize=1)
def _stemmer():
    """Return the cached Snowball German stemmer."""
    return snowballstemmer.stemmer("german")


def preprocess_de(text: str | None, remove_stopwords: bool = True, apply_stemming: bool = True) -> list[str]:
    """Turn German text into index terms for lexical retrieval.

    Steps in order: lowercase, regex tokenization (letters incl. umlauts/ß and digits), removal of
    stopwords (matched on the unstemmed lowercase token), Snowball German stemming.

    Args:
        text: Raw text; ``None`` is treated as empty.
        remove_stopwords: Drop tokens that are German stopwords if True.
        apply_stemming: Apply the Snowball stemmer to the remaining tokens if True.

    Returns:
        List of terms in text order (duplicates kept for term-frequency weighting).
    """
    tokens = tokenize_de(text)
    if remove_stopwords:
        stopwords = german_stopwords()
        # For every token: keep it only if it is not a stopword (stopwords carry no topical signal for BM25).
        tokens = [token for token in tokens if token not in stopwords]
    return _stemmer().stemWords(tokens) if apply_stemming else tokens


def preprocessing_info() -> dict:
    """Describe the preprocessing configuration and package versions for result files.

    Returns:
        JSON-serializable dict with the processing steps, the tokenizer regex, the number of
        stopwords, the installed ``stopwordsiso`` and ``snowballstemmer`` versions and a note
        on the omitted compound splitting.
    """
    return {"steps": ["lowercase", "regex-tokenization", "stopword removal (stopwordsiso, de)",
                      "snowball stemming (german)"],
            "tokenizer_regex": TOKEN_PATTERN.pattern,
            "n_stopwords": len(german_stopwords()),
            "stopwordsiso_version": version("stopwordsiso"),
            "snowballstemmer_version": version("snowballstemmer"),
            "compound_splitting": "omitted (no reviewed library available)"}
