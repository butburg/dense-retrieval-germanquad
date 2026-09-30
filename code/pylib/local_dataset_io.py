"""Local dataset I/O helpers for JSONL, TSV and TREC-in-GZIP parsing.

This module provides thin file readers and writers used by the dataset parsing
notebook. Parsers are intentionally lightweight and optimized for predictable,
line-oriented input files in this repository.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

JSONL_SPLIT_THRESHOLD = 57_000


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a UTF-8 JSONL file into a list of dictionaries.

    Empty lines are ignored. Each non-empty line must contain exactly one valid
    JSON object. If split files like ``.part2`` or ``.part3`` exist for the
    same base name, they are loaded in numeric order after the base file.

    :param path: Path to the JSONL file.
    :type path: Path
    :return: Parsed records in file order.
    :rtype: List[Dict[str, Any]]
    :raises FileNotFoundError: If the base file does not exist.
    :raises json.JSONDecodeError: If a line is not valid JSON.
    """
    rows: List[Dict[str, Any]] = []
    part_pattern = re.compile(rf"^{re.escape(path.stem)}\.part(?P<part>\d+){re.escape(path.suffix)}$")
    part_files: List[Tuple[int, Path]] = []
    for candidate in path.parent.glob(f"{path.stem}.part*{path.suffix}"):
        match = part_pattern.match(candidate.name)
        if match is None:
            continue
        part_files.append((int(match.group("part")), candidate))

    source_files = [path] + [part_path for _, part_path in sorted(part_files, key=lambda item: item[0])]
    for source_path in source_files:
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records as ASCII-safe JSONL and return the number of written rows.

    The parent directory is created automatically when missing. Non-ASCII
    characters are escaped because ``ensure_ascii=True`` is used. Once a file
    reaches ``JSONL_SPLIT_THRESHOLD`` rows, additional rows continue in
    ``.part2``, ``.part3`` and so on.

    :param path: Target JSONL file path.
    :type path: Path
    :param records: Iterable of JSON-serializable dictionary records.
    :type records: Iterable[Dict[str, Any]]
    :return: Number of serialized records.
    :rtype: int
    :raises TypeError: If a record is not JSON serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_parts = sorted(path.parent.glob(f"{path.stem}.part*{path.suffix}"))
    for existing_part in existing_parts:
        existing_part.unlink()

    handle = path.open("w", encoding="utf-8")
    written = 0
    try:
        for row in records:
            if written > 0 and written % JSONL_SPLIT_THRESHOLD == 0:
                handle.close()
                part_number = (written // JSONL_SPLIT_THRESHOLD) + 1
                part_path = path.with_name(f"{path.stem}.part{part_number}{path.suffix}")
                handle = part_path.open("w", encoding="utf-8")
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1
    finally:
        handle.close()
    return written


def count_text_lines(path: Path) -> int:
    """Count text lines in a UTF-8 file.

    :param path: Path to a text file.
    :type path: Path
    :return: Number of lines in the file.
    :rtype: int
    :raises FileNotFoundError: If the file does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_tsv(path: Path) -> List[Dict[str, Any]]:
    """Parse a UTF-8 TSV file with a header row.

    The parser uses ``csv.DictReader`` and returns one dictionary per data line.
    Header names are taken from the first row and values remain untyped strings.

    :param path: Path to a tab-separated values file.
    :type path: Path
    :return: Parsed rows as dictionaries keyed by TSV header names.
    :rtype: List[Dict[str, Any]]
    :raises FileNotFoundError: If the file does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _logical_lines(lines: Iterable[str], stats: Dict[str, Any]) -> Iterable[str]:
    """Yield lines, counting physical lines and splitting ``</DOC><DOC>`` in two."""
    for raw_line in lines:
        stats["line_count"] += 1
        if raw_line.strip() == "</DOC><DOC>":
            yield "</DOC>"
            yield "<DOC>"
        else:
            yield raw_line


def parse_trec_corpus(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a gzip-compressed, line-oriented TREC-like corpus file.

    Expected structure is tag-delimited documents with ``<DOC>``, ``<DOCNO>``,
    ``<TEXT...>`` and closing tags, usually one tag per line; a combined ``</DOC><DOC>`` line is treated as
    two tags. The parser is
    intentionally permissive and tracks structural differences in the returned
    stats dictionary.

    Limitations: this is not a full XML parser; nested tags and inline
    multi-tag constructs outside the expected line-oriented pattern may be
    ignored or flattened.

    :param path: Path to a ``.trec.gz`` corpus file.
    :type path: Path
    :return: Tuple of parsed documents and parser statistics.
    :rtype: Tuple[List[Dict[str, Any]], Dict[str, Any]]
    :raises FileNotFoundError: If the file does not exist.
    :raises OSError: If the gzip stream cannot be opened or decoded.
    """
    docs: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "line_count": 0,
        "doc_open_tags": 0,
        "doc_close_tags": 0,
        "docno_tags": 0,
        "text_open_tags": 0,
        "text_close_tags": 0,
        "parsed_docs": 0,
        "docs_missing_docno": 0,
        "docs_missing_text": 0,
    }

    current: Dict[str, Any] | None = None
    text_lines: List[str] = []
    in_text = False

    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in _logical_lines(handle, stats):
            line = raw_line.strip()

            if line == "<DOC>":
                stats["doc_open_tags"] += 1
                current = {}
                text_lines = []
                in_text = False
                continue

            if line == "</DOC>":
                stats["doc_close_tags"] += 1
                if current is None:
                    continue
                if "docno" not in current:
                    stats["docs_missing_docno"] += 1
                if "text" not in current:
                    current["text"] = " ".join(text_lines).strip()
                if not current.get("text", ""):
                    stats["docs_missing_text"] += 1
                if "docno" in current:
                    docs.append({"docno": str(current["docno"]), "text": str(current.get("text", ""))})
                    stats["parsed_docs"] += 1
                current = None
                text_lines = []
                in_text = False
                continue

            if current is None:
                continue

            if line.startswith("<DOCNO>") and line.endswith("</DOCNO>"):
                stats["docno_tags"] += 1
                value = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
                current["docno"] = value
                continue

            if line.startswith("<TEXT"):
                stats["text_open_tags"] += 1
                if ">" in line:
                    after_open = line.split(">", 1)[1]
                else:
                    after_open = line[len("<TEXT") :]

                if "</TEXT>" in after_open:
                    inline_text = after_open.split("</TEXT>", 1)[0].strip()
                    if inline_text:
                        text_lines.append(inline_text)
                    stats["text_close_tags"] += 1
                    in_text = False
                    current["text"] = " ".join(text_lines).strip()
                else:
                    inline_text = after_open.strip()
                    if inline_text:
                        text_lines.append(inline_text)
                    in_text = True
                continue

            if line == "</TEXT>":
                stats["text_close_tags"] += 1
                in_text = False
                current["text"] = " ".join(text_lines).strip()
                continue

            if in_text:
                if "</TEXT>" in line:
                    before_close = line.split("</TEXT>", 1)[0].strip()
                    if before_close:
                        text_lines.append(before_close)
                    stats["text_close_tags"] += 1
                    in_text = False
                    current["text"] = " ".join(text_lines).strip()
                else:
                    text_lines.append(line)

    stats["parser_diff_doc_open_minus_parsed"] = stats["doc_open_tags"] - stats["parsed_docs"]
    stats["parser_diff_docno_minus_parsed"] = stats["docno_tags"] - stats["parsed_docs"]
    stats["parser_diff_text_open_minus_text_close"] = stats["text_open_tags"] - stats["text_close_tags"]

    return docs, stats
