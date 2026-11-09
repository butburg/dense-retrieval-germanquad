"""Normalization helpers for retrieval dataset roles.

The functions in this module map heterogeneous raw records into a shared
structure with ``queries``, ``docs`` and ``qrels`` records and provide simple
consistency checks used in notebook-based preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _pick_first(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first non-null value found for the candidate keys.

    Args:
        record: Source dictionary to inspect, e.g. ``{"_id": "q1", "text": "Frage"}``.
        keys: Candidate keys in priority order, e.g. ``["query_id", "_id"]``.

    Returns:
        The value of the first key that is present and not ``None``, otherwise ``None``.
    """
    # Try candidate keys in priority order; the first one with a non-null value wins
    # (handles datasets that name the same field differently).
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _to_text(value: Any) -> str:
    """Convert a value to text; ``None`` becomes an empty string."""
    if value is None:
        return ""
    return str(value)


def normalize_queries(
    records: Iterable[Dict[str, Any]],
    *,
    source: str,
    split: str,
    id_keys: Sequence[str],
    text_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Normalize raw query records into the shared query schema.

    A row is skipped when either query identifier or query text cannot be
    resolved from the configured key lists.

    Args:
        records: Raw query records from source files.
        source: Dataset source label stored in every normalized record, e.g. ``"germanquad"``.
        split: Split label stored in every normalized record, e.g. ``"test"``.
        id_keys: Candidate keys for the query identifier in priority order.
        text_keys: Candidate keys for the query text in priority order.

    Returns:
        Normalized records in input order with the keys ``source``, ``split``, ``query_id``,
        ``query_text`` (both converted to ``str``) and ``meta`` (sorted ``raw_keys`` of the
        source row).
    """
    normalized: List[Dict[str, Any]] = []
    # For every raw query row: resolve ID and text via the candidate keys, skip incomplete rows, emit one
    # schema-conform record.
    for row in records:
        query_id = _pick_first(row, id_keys)
        query_text = _pick_first(row, text_keys)
        if query_id is None or query_text is None:
            continue
        normalized.append(
            {
                "source": source,
                "split": split,
                "query_id": _to_text(query_id),
                "query_text": _to_text(query_text),
                "meta": {"raw_keys": sorted(list(row.keys()))},
            }
        )
    return normalized


def normalize_docs(
    records: Iterable[Dict[str, Any]],
    *,
    source: str,
    split: str,
    id_keys: Sequence[str],
    text_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Normalize raw document records into the shared document schema.

    A row is skipped when document identifier or document text is missing after
    key resolution.

    Args:
        records: Raw document records from source files.
        source: Dataset source label stored in every normalized record, e.g. ``"germanquad"``.
        split: Split label stored in every normalized record, e.g. ``"test"``.
        id_keys: Candidate keys for the document identifier in priority order.
        text_keys: Candidate keys for the document text in priority order.

    Returns:
        Normalized records in input order with the keys ``source``, ``split``, ``doc_id``,
        ``text`` (both converted to ``str``) and ``meta`` (sorted ``raw_keys`` of the source row).
    """
    normalized: List[Dict[str, Any]] = []
    # For every raw document row: resolve ID and text via the candidate keys, skip incomplete rows, emit one
    # schema-conform record.
    for row in records:
        doc_id = _pick_first(row, id_keys)
        text = _pick_first(row, text_keys)
        if doc_id is None or text is None:
            continue
        normalized.append(
            {
                "source": source,
                "split": split,
                "doc_id": _to_text(doc_id),
                "text": _to_text(text),
                "meta": {"raw_keys": sorted(list(row.keys()))},
            }
        )
    return normalized


def normalize_qrels(
    records: Iterable[Dict[str, Any]],
    *,
    source: str,
    split: str,
    query_id_keys: Sequence[str],
    doc_id_keys: Sequence[str],
    score_keys: Sequence[str],
    default_score: int = 1,
) -> List[Dict[str, Any]]:
    """Normalize raw qrels records into the shared relevance schema.

    A row is skipped when query or document identifiers are missing after key
    resolution. Relevance values are coerced to ``int`` and fall back to
    ``default_score`` when absent or non-numeric.

    Args:
        records: Raw qrels records from source files.
        source: Dataset source label stored in every normalized record, e.g. ``"germanquad"``.
        split: Split label stored in every normalized record, e.g. ``"test"``.
        query_id_keys: Candidate keys for the query identifier in priority order.
        doc_id_keys: Candidate keys for the document identifier in priority order.
        score_keys: Candidate keys for the relevance score in priority order.
        default_score: Relevance used when the score is missing or not convertible to ``int``.

    Returns:
        Normalized records in input order with the keys ``source``, ``split``, ``query_id``,
        ``doc_id`` (both ``str``), ``relevance`` (``int``) and ``meta`` (sorted ``raw_keys``
        of the source row).
    """
    normalized: List[Dict[str, Any]] = []
    # For every raw qrels row: resolve query ID, doc ID and relevance score (default if absent), skip rows without
    # IDs.
    for row in records:
        query_id = _pick_first(row, query_id_keys)
        doc_id = _pick_first(row, doc_id_keys)
        if query_id is None or doc_id is None:
            continue
        score = _pick_first(row, score_keys)
        if score is None:
            score = default_score
        try:
            relevance = int(score)
        except (TypeError, ValueError):
            relevance = default_score
        normalized.append(
            {
                "source": source,
                "split": split,
                "query_id": _to_text(query_id),
                "doc_id": _to_text(doc_id),
                "relevance": relevance,
                "meta": {"raw_keys": sorted(list(row.keys()))},
            }
        )
    return normalized


def check_id_consistency(
    queries: Sequence[Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
    qrels: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute basic ID consistency diagnostics across normalized roles.

    The check compares query and document IDs referenced by qrels against the
    available normalized query and document sets.

    Args:
        queries: Normalized query records with ``query_id``.
        docs: Normalized document records with ``doc_id``.
        qrels: Normalized qrels records with ``query_id`` and ``doc_id``.

    Returns:
        Dict with the counts ``query_count``, ``doc_count``, ``qrel_count``,
        ``qrels_query_id_unique``, ``qrels_doc_id_unique``,
        ``missing_query_ids_in_queries_count`` and ``missing_doc_ids_in_docs_count``
        (qrels IDs without a matching query or document) plus the first ten sorted
        missing IDs in ``missing_query_ids_in_queries_sample`` and
        ``missing_doc_ids_in_docs_sample``.
    """
    # Set comprehension over all query records: collects every distinct query ID for cheap membership checks.
    query_ids = {row["query_id"] for row in queries}
    # Same for documents: the set of all available doc IDs.
    doc_ids = {row["doc_id"] for row in docs}

    # Distinct query IDs that qrels refer to (a qrel pointing to an unknown query is a dangling judgment).
    qrel_query_ids = {row["query_id"] for row in qrels}
    # Distinct doc IDs that qrels refer to; compared with doc_ids below.
    qrel_doc_ids = {row["doc_id"] for row in qrels}

    missing_query_ids = sorted(list(qrel_query_ids - query_ids))
    missing_doc_ids = sorted(list(qrel_doc_ids - doc_ids))

    return {
        "query_count": len(query_ids),
        "doc_count": len(doc_ids),
        "qrel_count": len(qrels),
        "qrels_query_id_unique": len(qrel_query_ids),
        "qrels_doc_id_unique": len(qrel_doc_ids),
        "missing_query_ids_in_queries_count": len(missing_query_ids),
        "missing_doc_ids_in_docs_count": len(missing_doc_ids),
        "missing_query_ids_in_queries_sample": missing_query_ids[:10],
        "missing_doc_ids_in_docs_sample": missing_doc_ids[:10],
    }


def dedupe_normalized(
    normalized: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Remove duplicate normalized records by role-specific identity keys.

    Deduplication keeps the first occurrence order per role. Identity keys are
    ``query_id`` for queries, ``doc_id`` for docs and
    ``(query_id, doc_id, relevance)`` for qrels. Queries and docs with an empty or
    missing ID are dropped.

    Args:
        normalized: Mapping with the optional keys ``queries``, ``docs`` and ``qrels``,
            each a list of normalized records.

    Returns:
        New mapping with the keys ``queries``, ``docs`` and ``qrels`` (missing roles
        become empty lists), duplicates removed.
    """
    seen_query_ids = set()
    seen_doc_ids = set()
    seen_qrel_keys = set()

    queries: List[Dict[str, Any]] = []
    # For every query record in input order: keep it only the first time its ID appears (drops duplicates and empty
    # IDs).
    for row in normalized.get("queries", []):
        query_id = row.get("query_id", "")
        if query_id and query_id not in seen_query_ids:
            seen_query_ids.add(query_id)
            queries.append(row)

    docs: List[Dict[str, Any]] = []
    # For every document record in input order: keep only the first record per doc ID (drops duplicates and empty
    # IDs).
    for row in normalized.get("docs", []):
        doc_id = row.get("doc_id", "")
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            docs.append(row)

    qrels: List[Dict[str, Any]] = []
    # For every qrels record: keep only the first occurrence of each (query, doc, relevance) triple.
    for row in normalized.get("qrels", []):
        qrel_key = (row.get("query_id", ""), row.get("doc_id", ""), row.get("relevance", 0))
        if qrel_key not in seen_qrel_keys:
            seen_qrel_keys.add(qrel_key)
            qrels.append(row)

    return {"queries": queries, "docs": docs, "qrels": qrels}
