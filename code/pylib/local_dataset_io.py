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


def doc_text(doc: Dict[str, Any]) -> str:
    """Compose the indexed text of a normalized document.

    Args:
        doc: Normalized document with optional ``title`` and ``text`` fields.

    Returns:
        ``title``, a newline and ``text`` if both are non-empty after stripping,
        otherwise whichever of the two exists (empty string if neither).
    """
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a UTF-8 JSONL file into a list of dictionaries.

    Empty lines are ignored. Each non-empty line must contain exactly one valid
    JSON object. If split files like ``.part2`` or ``.part3`` exist for the
    same base name, they are loaded in numeric order after the base file.

    Args:
        path: Path of the base JSONL file, e.g. ``Path("docs.normalized.jsonl")``.

    Returns:
        Parsed records in file order: base file first, then the part files by part number.

    Raises:
        FileNotFoundError: If the base file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    rows: List[Dict[str, Any]] = []
    part_pattern = re.compile(rf"^{re.escape(path.stem)}\.part(?P<part>\d+){re.escape(path.suffix)}$")
    part_files: List[Tuple[int, Path]] = []
    # For every file next to the base file that matches "<stem>.part*<suffix>": keep it if the name
    # ends in a numeric part number and remember (number, path); large corpora are split into such parts.
    for candidate in path.parent.glob(f"{path.stem}.part*{path.suffix}"):
        match = part_pattern.match(candidate.name)
        if match is None:
            continue
        part_files.append((int(match.group("part")), candidate))

    # Sort by the numeric part number so that part10 follows part9.
    part_files_in_order = sorted(part_files, key=lambda part_file: part_file[0])
    # List comprehension: base file first, then the part files, ordered by part number.
    source_files = [path] + [part_path for _, part_path in part_files_in_order]
    # For every file (base file, then parts): read it line by line so the parts form one continuous record list.
    for source_path in source_files:
        with source_path.open("r", encoding="utf-8") as handle:
            # For every line of the file: skip blank lines, parse each remaining line as one JSON record.
            for line in handle:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                rows.append(json.loads(stripped_line))
    return rows


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records as ASCII-safe JSONL and return the number of written rows.

    The parent directory is created automatically when missing. Non-ASCII
    characters are escaped because ``ensure_ascii=True`` is used. Once a file
    reaches ``JSONL_SPLIT_THRESHOLD`` rows, additional rows continue in
    ``.part2``, ``.part3`` and so on.

    Existing ``.part*`` files of the same base name are deleted first so that no stale
    parts remain.

    Args:
        path: Target JSONL file path, e.g. ``Path("output/docs.normalized.jsonl")``.
        records: JSON-serializable dictionaries, written one per line in iteration order.

    Returns:
        Number of written records.

    Raises:
        TypeError: If a record is not JSON serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_part_files = sorted(path.parent.glob(f"{path.stem}.part*{path.suffix}"))
    # For every old part file of this base name: delete it so rows of an earlier, longer export cannot remain.
    for stale_part_file in stale_part_files:
        stale_part_file.unlink()

    handle = path.open("w", encoding="utf-8")
    written = 0
    try:
        # For every record in iteration order: write it as one JSON line; start a new part file whenever
        # JSONL_SPLIT_THRESHOLD rows have been written (splits large outputs into several part files).
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

    Args:
        path: Path of the text file.

    Returns:
        Number of lines in the file (empty lines included).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        # Generator yielding 1 per line; the sum is the line count without holding the file in memory.
        return sum(1 for _ in handle)


def parse_tsv(path: Path) -> List[Dict[str, Any]]:
    """Parse a UTF-8 TSV file with a header row.

    The parser uses ``csv.DictReader`` and returns one dictionary per data line.
    Header names are taken from the first row and values remain untyped strings.

    Args:
        path: Path of the tab-separated values file.

    Returns:
        One dictionary per data line, keyed by the header names, e.g.
        ``[{"query-id": "1", "corpus-id": "d1", "score": "1"}]``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        # For every CSV row: copy it into a plain dict (header name -> string value).
        return [dict(row) for row in reader]


def _logical_lines(physical_lines: Iterable[str], stats: Dict[str, Any]) -> Iterable[str]:
    """Yield lines, counting physical lines in ``stats["line_count"]`` and splitting ``</DOC><DOC>`` in two."""
    # For every raw line of the file: count it and pass it on; a combined "</DOC><DOC>" line is split into
    # two logical lines so the state machine below sees both tags.
    for physical_line in physical_lines:
        stats["line_count"] += 1
        if physical_line.strip() == "</DOC><DOC>":
            yield "</DOC>"
            yield "<DOC>"
        else:
            yield physical_line


def parse_trec_corpus(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a gzip-compressed, line-oriented TREC-like corpus file.

    Expected structure is tag-delimited documents with ``<DOC>``, ``<DOCNO>``,
    ``<TEXT...>`` and closing tags, usually one tag per line; a combined
    ``</DOC><DOC>`` line is treated as two tags. The parser is intentionally
    permissive and tracks structural differences in the returned stats dictionary.

    Limitations: this is not a full XML parser; nested tags and inline
    multi-tag constructs outside the expected line-oriented pattern may be
    ignored or flattened. Undecodable bytes are dropped.

    Args:
        path: Path of the ``.trec.gz`` corpus file.

    Returns:
        Tuple ``(docs, stats)``. ``docs`` holds one ``{"docno": str, "text": str}`` per
        document that has a ``<DOCNO>``. ``stats`` counts physical lines, tags, parsed
        documents, documents without DOCNO or text, and the differences between tag
        counts (``parser_diff_*``).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the gzip stream cannot be opened or decoded.
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

    current_doc: Dict[str, Any] | None = None
    text_lines: List[str] = []
    in_text = False  # inside a multi-line <TEXT> block

    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        # State machine over all logical lines: <DOC> opens a document, <DOCNO> sets its ID, <TEXT>...</TEXT>
        # collects its text lines, </DOC> closes it and stores (docno, text); stats count every tag for later sanity
        # checks.
        for logical_line in _logical_lines(handle, stats):
            stripped_line = logical_line.strip()

            if stripped_line == "<DOC>":
                stats["doc_open_tags"] += 1
                current_doc = {}
                text_lines = []
                in_text = False
                continue

            if stripped_line == "</DOC>":
                stats["doc_close_tags"] += 1
                if current_doc is None:
                    continue
                if "docno" not in current_doc:
                    stats["docs_missing_docno"] += 1
                if "text" not in current_doc:
                    current_doc["text"] = " ".join(text_lines).strip()
                if not current_doc.get("text", ""):
                    stats["docs_missing_text"] += 1
                if "docno" in current_doc:
                    docs.append({"docno": str(current_doc["docno"]), "text": str(current_doc.get("text", ""))})
                    stats["parsed_docs"] += 1
                current_doc = None
                text_lines = []
                in_text = False
                continue

            if current_doc is None:
                continue

            if stripped_line.startswith("<DOCNO>") and stripped_line.endswith("</DOCNO>"):
                stats["docno_tags"] += 1
                value = stripped_line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
                current_doc["docno"] = value
                continue

            if stripped_line.startswith("<TEXT"):
                stats["text_open_tags"] += 1
                if ">" in stripped_line:
                    after_open = stripped_line.split(">", 1)[1]
                else:
                    after_open = stripped_line[len("<TEXT") :]

                if "</TEXT>" in after_open:
                    inline_text = after_open.split("</TEXT>", 1)[0].strip()
                    if inline_text:
                        text_lines.append(inline_text)
                    stats["text_close_tags"] += 1
                    in_text = False
                    current_doc["text"] = " ".join(text_lines).strip()
                else:
                    inline_text = after_open.strip()
                    if inline_text:
                        text_lines.append(inline_text)
                    in_text = True
                continue

            if stripped_line == "</TEXT>":
                stats["text_close_tags"] += 1
                in_text = False
                current_doc["text"] = " ".join(text_lines).strip()
                continue

            if in_text:
                if "</TEXT>" in stripped_line:
                    before_close = stripped_line.split("</TEXT>", 1)[0].strip()
                    if before_close:
                        text_lines.append(before_close)
                    stats["text_close_tags"] += 1
                    in_text = False
                    current_doc["text"] = " ".join(text_lines).strip()
                else:
                    text_lines.append(stripped_line)

    stats["parser_diff_doc_open_minus_parsed"] = stats["doc_open_tags"] - stats["parsed_docs"]
    stats["parser_diff_docno_minus_parsed"] = stats["docno_tags"] - stats["parsed_docs"]
    stats["parser_diff_text_open_minus_text_close"] = stats["text_open_tags"] - stats["text_close_tags"]

    return docs, stats
