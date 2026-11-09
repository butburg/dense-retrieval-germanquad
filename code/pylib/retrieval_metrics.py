"""Retrieval-only ranking and metric logic (Recall, Success, MRR and Precision at k)."""
from __future__ import annotations

import numpy as np

K_VALUES = (1, 5, 10)  # cutoffs for Recall/Success/MRR@k
RELEVANCE_THRESHOLD = 1  # minimum qrels relevance grade that counts as relevant


def build_gold(qrels: list[dict], corpus_document_ids: set[str],
               relevance_threshold: int = RELEVANCE_THRESHOLD) -> dict[str, set[str]]:
    """Collect the gold-relevant document IDs per query from qrels records.

    Args:
        qrels: Normalized qrels rows with the keys ``query_id``, ``doc_id`` and ``relevance``,
            e.g. ``[{"query_id": "q1", "doc_id": "a", "relevance": 1}]``.
        corpus_document_ids: IDs of the documents in the evaluated corpus, e.g. ``{"a", "b"}``.
            Qrels rows pointing to other documents are ignored.
        relevance_threshold: Minimum relevance grade for a document to count as relevant.

    Returns:
        Mapping from query ID to its set of relevant document IDs, e.g. ``{"q1": {"a"}}``.
        IDs are converted to ``str``; queries without any relevant document are absent.
    """
    gold: dict[str, set[str]] = {}
    # For every qrels row: if it is a relevant judgment on a document of the evaluated corpus, add the doc ID
    # to that query's gold set (builds the ground truth for Recall/Precision/MRR).
    for qrel in qrels:
        is_relevant = int(qrel["relevance"]) >= relevance_threshold
        is_in_corpus = str(qrel["doc_id"]) in corpus_document_ids
        if is_relevant and is_in_corpus:
            gold.setdefault(str(qrel["query_id"]), set()).add(str(qrel["doc_id"]))
    return gold


def rank_cosine(query_embeddings: np.ndarray, document_embeddings: np.ndarray,
                max_k: int = 10) -> np.ndarray:
    """Rank documents per query by dot product, which equals cosine similarity for unit vectors.

    Args:
        query_embeddings: Array of shape ``(n_queries, dim)``. Rows must be L2-normalised.
        document_embeddings: Array of shape ``(n_documents, dim)``. Rows must be L2-normalised.
        max_k: Number of top documents kept per query. Must not exceed ``n_documents``.

    Returns:
        Integer array of shape ``(n_queries, max_k)`` with document indices (rows of
        ``document_embeddings``), best first. Ties keep the order given by ``argpartition``
        followed by a stable sort.
    """
    # Matrix product (n_queries, dim) @ (dim, n_documents) gives all query-document dot products at once;
    # entry [i, j] is the cosine similarity of query i and document j (unit vectors), e.g. shape (3, 5) for 3 queries,
    # 5 docs.
    similarities = query_embeddings.astype(np.float32) @ document_embeddings.astype(np.float32).T
    # argpartition finds the top max_k unordered; only those are sorted afterwards.
    # Negated scores turn "largest" into "smallest": argpartition moves the max_k best document indices of each row
    # to the front without fully sorting, e.g. scores [0.1, 0.9, 0.5] with max_k=2 -> indices {1, 2} in any order.
    unordered_top_indices = np.argpartition(-similarities, max_k - 1, axis=1)[:, :max_k]
    # take_along_axis picks, per row, the scores at those indices, e.g. row [0.1, 0.9, 0.5] with indices [2, 1] ->
    # [0.5, 0.9].
    top_similarities = np.take_along_axis(similarities, unordered_top_indices, 1)
    # Sort only these max_k scores per row (descending via the minus sign; stable keeps ties in the same order):
    # the result holds positions within the top-k list, e.g. [0.5, 0.9] -> [1, 0].
    sorted_positions = np.argsort(-top_similarities, axis=1, kind="stable")
    # Apply the positions to the indices to get document indices best first, e.g. [2, 1] with positions [1, 0] -> [1,
    # 2].
    return np.take_along_axis(unordered_top_indices, sorted_positions, 1)


def _check_alignment(query_ids: list[str], retrieved_document_ids_by_query: list[list[str]]) -> None:
    """Check that every query has exactly one ranked list and query IDs are unique.

    Args:
        query_ids: Query IDs that must be unique.
        retrieved_document_ids_by_query: One ranked document list per query, same order as ``query_ids``.

    Raises:
        ValueError: If the lengths differ or ``query_ids`` contain duplicates (explicit check
            instead of ``assert`` so it also holds under ``python -O``).
    """
    if len(query_ids) != len(retrieved_document_ids_by_query):
        raise ValueError(f"query_ids and retrieved_document_ids_by_query differ in length "
                         f"({len(query_ids)} vs. {len(retrieved_document_ids_by_query)}); "
                         "each query needs exactly one ranked list.")
    if len(set(query_ids)) != len(query_ids):
        # Set comprehension: query IDs that occur more than once (only used to show examples in the error message).
        duplicate_ids = sorted({q for q in query_ids if query_ids.count(q) > 1})[:5]
        raise ValueError(f"query_ids contain duplicates (examples: {duplicate_ids}); "
                         "query IDs must be unique.")


def first_relevant_ranks(query_ids: list[str],
                         retrieved_document_ids_by_query: list[list[str]],
                         gold_relevant_document_ids_by_query: dict[str, set[str]]) -> list[int | None]:
    """Find the rank of the first gold-relevant document in each ranked list.

    Args:
        query_ids: IDs of the evaluated queries, e.g. ``["q1", "q2"]``. Must be unique.
        retrieved_document_ids_by_query: Ranked document IDs per query, best first, same
            order and length as ``query_ids``, e.g. ``[["a", "c"], ["c", "b"]]``.
        gold_relevant_document_ids_by_query: Relevant document IDs per query, e.g.
            ``{"q1": {"a"}, "q2": {"b", "c"}}``. Missing queries have no relevant documents.

    Returns:
        One entry per query: the 1-based rank of the first relevant document within the
        given list, or ``None`` if the list contains none.

    Raises:
        ValueError: If ``query_ids`` and ``retrieved_document_ids_by_query`` differ in
            length or ``query_ids`` contain duplicates.
    """
    _check_alignment(query_ids, retrieved_document_ids_by_query)
    first_ranks = []
    # For every query with its ranked list: find the 1-based rank of the first gold-relevant document
    # (reported per query in the result files).
    for query_id, retrieved_document_ids in zip(
            query_ids, retrieved_document_ids_by_query, strict=True):
        gold_relevant_document_ids = gold_relevant_document_ids_by_query.get(query_id, set())
        first_rank = None
        # Walk down the ranked list from rank 1; stop at the first gold-relevant document.
        for rank, document_id in enumerate(retrieved_document_ids, start=1):
            if document_id in gold_relevant_document_ids:
                first_rank = rank
                break
        first_ranks.append(first_rank)
    return first_ranks


def compute_metrics(query_ids: list[str],
                    retrieved_document_ids_by_query: list[list[str]],
                    gold_relevant_document_ids_by_query: dict[str, set[str]],
                    dataset_name: str,
                    cutoff_values=K_VALUES) -> list[dict]:
    """Compute Recall, Success, MRR and Precision at k, averaged over all queries.

    Args:
        query_ids: IDs of the evaluated queries, e.g. ``["q1", "q2"]``. Must be unique.
        retrieved_document_ids_by_query: For each query, the document IDs returned by the
            retriever, sorted with the highest rank first. Same order and length as
            ``query_ids``, e.g. ``[["a", "c"], ["c", "b"]]``. These are candidates and not
            automatically relevant. Each list should contain at least
            ``max(cutoff_values)`` documents.
        gold_relevant_document_ids_by_query: Documents that the gold standard marks as
            relevant for each query, e.g. ``{"q1": {"a"}, "q2": {"b", "c"}}``.
        dataset_name: Name written into the ``dataset`` field of every result row.
        cutoff_values: Ranks up to which retrieved documents are evaluated, e.g. ``(1, 2)``
            evaluates the top 1 and the top 2 documents.

    Returns:
        One row (dict) per cutoff with the keys ``dataset``, ``k``, ``recall_at_k``,
        ``success_at_k``, ``mrr_at_k``, ``num_queries`` and ``precision_at_k``. The metric
        values are means over all queries. Queries without gold documents contribute 0 and
        stay in the denominator (denominator = all queries). An empty query list yields 0
        for all metrics.

    Raises:
        ValueError: If ``query_ids`` and ``retrieved_document_ids_by_query`` differ in
            length or ``query_ids`` contain duplicates.
    """
    _check_alignment(query_ids, retrieved_document_ids_by_query)

    metric_rows = []
    query_count = len(query_ids)
    divisor = query_count or 1  # empty query list: avoid division by zero, all metrics stay 0

    # For every cutoff k (e.g. 1, 5, 10): re-evaluate all queries using only the top k documents
    # and average the four metrics, which gives one result row per k.
    for cutoff in cutoff_values:
        recall_sum = 0.0
        success_sum = 0.0
        mrr_sum = 0.0
        precision_sum = 0.0

        # For every query: count gold hits in its top-k list and add the query's Recall, Success, Precision and
        # reciprocal-rank contribution to the running sums (divided by the number of queries after the loop).
        for query_id, retrieved_document_ids in zip(
                query_ids, retrieved_document_ids_by_query, strict=True):
            gold_relevant_document_ids = gold_relevant_document_ids_by_query.get(query_id, set())

            # Only the first `cutoff` retrieved documents count.
            retrieved_top_k_document_ids = retrieved_document_ids[:cutoff]

            # Number of top-k documents that are gold relevant.
            correct_retrieval_count = 0
            # For every retrieved top-k document: count it if it is gold relevant (basis for Recall, Success,
            # Precision).
            for document_id in retrieved_top_k_document_ids:
                if document_id in gold_relevant_document_ids:
                    correct_retrieval_count += 1

            # Recall: share of gold documents found in the top k (queries without gold add 0).
            if gold_relevant_document_ids:
                recall_sum += correct_retrieval_count / len(gold_relevant_document_ids)

            # Success: 1 if at least one gold document is in the top k.
            if correct_retrieval_count > 0:
                success_sum += 1

            # Precision: share of the top-k slots that hold a gold document.
            precision_sum += correct_retrieval_count / cutoff

            # MRR: rank (1-based) of the first gold document within the top k, if any.
            first_correct_retrieval_rank = None
            # Walk down the top-k list from rank 1 and stop at the first gold document (its rank gives the reciprocal
            # rank).
            for rank, document_id in enumerate(retrieved_top_k_document_ids, start=1):
                if document_id in gold_relevant_document_ids:
                    first_correct_retrieval_rank = rank
                    break
            if first_correct_retrieval_rank is not None:
                mrr_sum += 1 / first_correct_retrieval_rank

        # Mean over all queries (queries without gold stay in the denominator).
        metric_rows.append({"dataset": dataset_name, "k": cutoff,
                            "recall_at_k": recall_sum / divisor,
                            "success_at_k": success_sum / divisor,
                            "mrr_at_k": mrr_sum / divisor,
                            "num_queries": query_count,
                            "precision_at_k": precision_sum / divisor})
    return metric_rows


def success_at_k(first_ranks: np.ndarray, k: int) -> np.ndarray:
    """Success@k indicator per query from the rank of the first relevant passage.

    Args:
        first_ranks: Integer array, 1-based rank of the first relevant passage per query
            (0 = no relevant passage in the ranking), e.g. ``np.array([1, 7, 0])``.
        k: Cutoff, e.g. ``5``.

    Returns:
        Float array with 1.0 where ``1 <= rank <= k`` and 0.0 otherwise, e.g. ``[1., 0., 0.]``
        for the example and ``k=5``. Its mean over queries is Success@k.
    """
    # Element-wise boolean masks combined with & (and): True where the first relevant passage lies within the top k;
    # astype(float) turns True/False into 1.0/0.0, e.g. first_ranks = [1, 7, 0], k = 5 -> [1., 0., 0.].
    return ((first_ranks > 0) & (first_ranks <= k)).astype(float)


def reciprocal_rank_at_k(first_ranks: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal rank per query with cutoff ``k`` from the rank of the first relevant passage.

    Args:
        first_ranks: Integer array, 1-based rank of the first relevant passage per query
            (0 = no relevant passage in the ranking), e.g. ``np.array([1, 7, 12, 0])``.
        k: Cutoff, e.g. ``10``.

    Returns:
        Float array with ``1 / rank`` for ``1 <= rank <= k`` and 0.0 otherwise, e.g.
        ``[1., 0.143, 0., 0.]`` for the example and ``k=10``. Its mean over queries is MRR@k.
    """
    # np.where(mask, a, b) picks a where the mask is True, else b: reciprocal rank 1/rank for ranks 1..k, 0 otherwise.
    # maximum(first_ranks, 1) avoids 1/0 for rank 0, e.g. first_ranks = [1, 7, 12, 0], k = 10 -> [1., 0.143, 0., 0.].
    return np.where((first_ranks > 0) & (first_ranks <= k), 1.0 / np.maximum(first_ranks, 1), 0.0)


def best_gold_rank(ranking: list[tuple[str, float]], gold: set[str]) -> int | None:
    """1-based rank of the best-ranked gold document in a full ranking.

    Args:
        ranking: Full ranking as ``(doc_id, score)``, best first (any retriever).
        gold: Gold document IDs of the query, e.g. ``{"c40369"}``.

    Returns:
        Rank of the first gold document in ``ranking``, or ``None`` if ``gold`` is empty or not contained.
    """
    # Generator over the ranking with 1-based ranks: next() returns the rank of the first document that is gold,
    # or None if there is none.
    return next((rank for rank, (doc_id, _) in enumerate(ranking, 1) if doc_id in gold), None)
