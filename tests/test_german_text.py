"""German preprocessing and BM25-de comparison helpers."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "pylib"))
from german_text import german_stopwords, preprocess_de, preprocessing_info  # noqa: E402


def test_lowercase_tokenize_and_umlauts():
    assert preprocess_de("Käse, ÄPFEL & 42!", stopwords=False, stem=False) == ["käse", "äpfel", "42"]


def test_stopwords_removed_before_stemming():
    assert "die" in german_stopwords()
    assert preprocess_de("Die Häuser der Stadt", stem=False) == ["häuser", "stadt"]


def test_snowball_stemming_conflates_inflections():
    assert preprocess_de("Häuser", stopwords=False) == preprocess_de("Haus", stopwords=False) == ["haus"]
    assert preprocess_de("Katzen", stopwords=False) == preprocess_de("Katze", stopwords=False)
    assert preprocess_de("Kindern", stopwords=False) == preprocess_de("Kinder", stopwords=False)


def test_none_and_empty():
    assert preprocess_de(None) == [] and preprocess_de("") == [] and preprocess_de("der die das") == []


def test_info_documents_config():
    i = preprocessing_info()
    assert i["n_stopwords"] > 100 and "omitted" in i["compound_splitting"]

