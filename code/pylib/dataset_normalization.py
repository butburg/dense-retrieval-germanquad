"""Normalization helpers for retrieval dataset roles.

The functions in this module map heterogeneous raw records into a shared
structure with ``queries``, ``docs`` and ``qrels`` records and provide simple
consistency checks used in notebook-based preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _pick_first(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first non-null value found for candidate keys.

    :param record: Source dictionary to inspect.
    :type record: Dict[str, Any]
    :param keys: Candidate keys in priority order.
    :type keys: Sequence[str]
    :return: First available non-null value, else ``None``.
    :rtype: Any
    """
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _to_text(value: Any) -> str:
    """Convert a value to text and map ``None`` to an empty string.

    :param value: Arbitrary scalar input value.
    :type value: Any
    :return: String representation suitable for normalized fields.
    :rtype: str
    """
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

    :param records: Raw query records from source files.
    :type records: Iterable[Dict[str, Any]]
    :param source: Dataset source label persisted in normalized output.
    :type source: str
    :param split: Split label persisted in normalized output.
    :type split: str
    :param id_keys: Candidate keys for query identifiers in priority order.
    :type id_keys: Sequence[str]
    :param text_keys: Candidate keys for query text in priority order.
    :type text_keys: Sequence[str]
    :return: Normalized query records with ``query_id`` and ``query_text``.
    :rtype: List[Dict[str, Any]]
    """
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        qid = _pick_first(row, id_keys)
        text = _pick_first(row, text_keys)
        if qid is None or text is None:
            continue
        normalized.append(
            {
                "source": source,
                "split": split,
                "query_id": _to_text(qid),
                "query_text": _to_text(text),
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

    :param records: Raw document records from source files.
    :type records: Iterable[Dict[str, Any]]
    :param source: Dataset source label persisted in normalized output.
    :type source: str
    :param split: Split label persisted in normalized output.
    :type split: str
    :param id_keys: Candidate keys for document identifiers.
    :type id_keys: Sequence[str]
    :param text_keys: Candidate keys for document text.
    :type text_keys: Sequence[str]
    :return: Normalized document records with ``doc_id`` and ``text``.
    :rtype: List[Dict[str, Any]]
    """
    normalized: List[Dict[str, Any]] = []
    for row in records:
        did = _pick_first(row, id_keys)
        text = _pick_first(row, text_keys)
        if did is None or text is None:
            continue
        normalized.append(
            {
                "source": source,
                "split": split,
                "doc_id": _to_text(did),
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

    :param records: Raw qrels records from source files.
    :type records: Iterable[Dict[str, Any]]
    :param source: Dataset source label persisted in normalized output.
    :type source: str
    :param split: Split label persisted in normalized output.
    :type split: str
    :param query_id_keys: Candidate keys for query identifiers.
    :type query_id_keys: Sequence[str]
    :param doc_id_keys: Candidate keys for document identifiers.
    :type doc_id_keys: Sequence[str]
    :param score_keys: Candidate keys for relevance scores.
    :type score_keys: Sequence[str]
    :param default_score: Fallback relevance when score is missing or invalid.
    :type default_score: int
    :return: Normalized qrels records with integer ``relevance``.
    :rtype: List[Dict[str, Any]]
    """
    normalized: List[Dict[str, Any]] = []
    for row in records:
        qid = _pick_first(row, query_id_keys)
        did = _pick_first(row, doc_id_keys)
        if qid is None or did is None:
            continue
        score = _pick_first(row, score_keys)
        if score is None:
            score = default_score
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = default_score
        normalized.append(
            {
                "source": source,
                "split": split,
                "query_id": _to_text(qid),
                "doc_id": _to_text(did),
                "relevance": score,
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

    :param queries: Normalized query records.
    :type queries: Sequence[Dict[str, Any]]
    :param docs: Normalized document records.
    :type docs: Sequence[Dict[str, Any]]
    :param qrels: Normalized qrels records.
    :type qrels: Sequence[Dict[str, Any]]
    :return: Summary metrics and small samples of missing IDs.
    :rtype: Dict[str, Any]
    """
    qids = {row["query_id"] for row in queries}
    dids = {row["doc_id"] for row in docs}

    qrel_qids = {row["query_id"] for row in qrels}
    qrel_dids = {row["doc_id"] for row in qrels}

    missing_queries = sorted(list(qrel_qids - qids))
    missing_docs = sorted(list(qrel_dids - dids))

    return {
        "query_count": len(qids),
        "doc_count": len(dids),
        "qrel_count": len(qrels),
        "qrels_query_id_unique": len(qrel_qids),
        "qrels_doc_id_unique": len(qrel_dids),
        "missing_query_ids_in_queries_count": len(missing_queries),
        "missing_doc_ids_in_docs_count": len(missing_docs),
        "missing_query_ids_in_queries_sample": missing_queries[:10],
        "missing_doc_ids_in_docs_sample": missing_docs[:10],
    }


def dedupe_normalized(
    normalized: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Remove duplicate normalized records by role-specific identity keys.

    Deduplication keeps the first occurrence order per role. Identity keys are
    ``query_id`` for queries, ``doc_id`` for docs and
    ``(query_id, doc_id, relevance)`` for qrels.

    :param normalized: Mapping with optional ``queries``, ``docs`` and ``qrels``.
    :type normalized: Dict[str, List[Dict[str, Any]]]
    :return: New mapping with duplicate entries removed.
    :rtype: Dict[str, List[Dict[str, Any]]]
    """
    q_seen = set()
    d_seen = set()
    r_seen = set()

    queries: List[Dict[str, Any]] = []
    for row in normalized.get("queries", []):
        key = row.get("query_id", "")
        if key and key not in q_seen:
            q_seen.add(key)
            queries.append(row)

    docs: List[Dict[str, Any]] = []
    for row in normalized.get("docs", []):
        key = row.get("doc_id", "")
        if key and key not in d_seen:
            d_seen.add(key)
            docs.append(row)

    qrels: List[Dict[str, Any]] = []
    for row in normalized.get("qrels", []):
        key = (row.get("query_id", ""), row.get("doc_id", ""), row.get("relevance", 0))
        if key not in r_seen:
            r_seen.add(key)
            qrels.append(row)

    return {"queries": queries, "docs": docs, "qrels": qrels}
