"""Live-Demo: Top-k von BM25 und einem offenen Embedding-Modell nebeneinander (nur CPU, kein API-Key).

Nutzung (im Ordner code/):
  python demo_query.py --query-id q40369 --k 5
  python demo_query.py --query "Wer entwickelte den seillosen Aufzug?" --model e5-large
  python demo_query.py --warm-cache            # Dokumentvektoren einmalig berechnen und cachen
Eingaben: {docs,queries,qrels}.normalized.jsonl in --dataset-dir (Default output/germanquad).
Ausgaben: Terminal; geschrieben wird nur der Embedding-Cache output/harness/cache/.
BM25 und Embedding-Logik stammen aus scripts/run_bm25.py, embed_eval.py und pylib/embedders.py.

Role in the experiment: presentation tool, independent of the evaluation. It shows for one query (a dataset query by
ID, or free text) the top-k passages of BM25 (regex tokenization, parameters of scripts/run_bm25.py) next to those of
one open embedding model, marks gold passages (only for ``--query-id``) and prints the gold rank in the full ranking.
Passage vectors are read from / written to the same cache as embed_eval.py (output/harness/cache/), so a run after
``embed_eval.py --stage encode`` needs no re-encoding. Only the single query is encoded at runtime. The imports of
heavy modules are deferred into the functions so that ``--help`` and error messages work without them.
"""
from __future__ import annotations

import argparse, shutil, sys, textwrap, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE))

OPEN_MODELS = ("e5-large", "bge-m3", "gte-multilingual-base", "jina-v3")  # local models allowed in the demo (no API key)
DS_FILES = ("docs", "queries", "qrels")  # normalized files read from the dataset directory


def fail(msg: str) -> None:
    """Print a clear error message and exit with status 1."""
    print(f"Fehler: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dataset(dataset_dir: Path):
    """Load docs, queries and qrels; abort with a hint if the normalized files are missing.

    Args:
        dataset_dir: Folder with ``{docs,queries,qrels}.normalized.jsonl``, e.g. ``output/germanquad``.

    Returns:
        List ``[docs, queries, qrels]`` of record lists (order of ``DS_FILES``).
    """
    from local_dataset_io import load_jsonl
    # Dict comprehension: role name (docs/queries/qrels) -> expected file path in the dataset folder.
    paths = {n: dataset_dir / f"{n}.normalized.jsonl" for n in DS_FILES}
    # List comprehension: paths of the normalized files that do not exist (reported in the error message).
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        fail("Normalisierte Daten fehlen: " + ", ".join(missing)
             + ". Erzeugen mit den Notebooks 01 und 02 (Ordner code/).")
    # List comprehension: load the three files in the fixed order of DS_FILES (docs, queries, qrels).
    return [load_jsonl(paths[n]) for n in DS_FILES]


def build_bm25(doc_ids: list[str], docs_by_id: dict):
    """BM25 index with the parameters of scripts/run_bm25.py.

    Args:
        doc_ids: Document IDs in index order, e.g. ``["c1", "c2"]``.
        docs_by_id: Document ID -> normalized document record.

    Returns:
        Tuple ``(bm25, tokenize)``: the ``BM25Okapi`` index (rows in the order of ``doc_ids``) and the tokenizer
        that must be applied to queries.
    """
    try:
        import run_bm25 as rb
    except ImportError as ex:
        fail(f"BM25-Abhängigkeiten fehlen ({ex}). Umgebung: pip install -r code/requirements-embed.txt")
    # List comprehension over doc_ids: one token list per passage, so index row i belongs to doc_ids[i].
    bm25 = rb.BM25Okapi([rb.tokenize_de(rb.doc_text(docs_by_id[i])) for i in doc_ids],
                        k1=rb.BM25_K1, b=rb.BM25_B, epsilon=rb.BM25_EPSILON)
    return bm25, rb.tokenize_de


def bm25_ranking(bm25, tokenize, doc_ids: list[str], query: str) -> list[tuple[str, float]]:
    """Full BM25 ranking, ties broken by (-score, doc_id) as in run_bm25.py.

    Args:
        bm25: Index from :func:`build_bm25`.
        tokenize: Tokenizer from :func:`build_bm25`.
        doc_ids: Document IDs in index order.
        query: Query text.

    Returns:
        ``(doc_id, score)`` for every document, best first.
    """
    import run_bm25 as rb
    return rb.rank_corpus(bm25, tokenize(query), doc_ids)


def dense_ranking(q_vec, d_emb, doc_ids: list[str]) -> list[tuple[str, float]]:
    """Full ranking by dot product of L2-normalised vectors (stable order for ties).

    Args:
        q_vec: Query embedding, shape ``(1, dim)``.
        d_emb: Document embeddings, shape ``(n_docs, dim)``, rows in the order of ``doc_ids``.
        doc_ids: Document IDs in embedding order.

    Returns:
        ``(doc_id, cosine similarity)`` for every document, best first. Ties keep the document order
        (differs from the BM25 tie rule by doc_id, relevant only for exactly equal scores).
    """
    import numpy as np
    # Matrix product (1, dim) @ (dim, n_docs) = cosine similarity of the query to every passage (unit vectors),
    # shape (1, n_docs); ravel() flattens it to one score per passage.
    sims = (q_vec.astype(np.float32) @ d_emb.astype(np.float32).T).ravel()
    # argsort of the negated scores lists passage indices from highest to lowest similarity,
    # e.g. sims [0.2, 0.9, 0.5] -> [1, 2, 0]; kind="stable" keeps document order for equal scores.
    order = np.argsort(-sims, kind="stable")
    # List comprehension: for every passage index in ranked order, pair its doc ID with its similarity score.
    return [(doc_ids[int(i)], float(sims[i])) for i in order]


def title_and_body(doc: dict) -> tuple[str, str]:
    """Title and passage body; falls back to the first text line when the title field is empty.

    Args:
        doc: Normalized document record with optional ``title`` and ``text``.

    Returns:
        Tuple ``(title, body)``; without a title the first line of ``text`` is the title.
    """
    title, text = str(doc.get("title") or "").strip(), str(doc.get("text") or "").strip()
    if title:
        return title, text
    head, _, rest = text.partition("\n")
    return head.strip(), rest.strip() or head.strip()


def snippet(doc: dict, n: int = 150) -> str:
    """Whitespace-collapsed passage start of about ``n`` characters.

    Args:
        doc: Normalized document record.
        n: Maximum number of body characters before the text is cut and an ellipsis is appended.

    Returns:
        Single-line excerpt of the passage body ("=" characters of wiki headings are replaced by spaces).
    """
    # split() cuts at any whitespace run, join(" ") glues the words with single spaces: collapses line breaks and
    # repeated blanks.
    body = " ".join(title_and_body(doc)[1].replace("=", " ").split())
    return body if len(body) <= n else body[:n].rstrip() + " …"


def column(name: str, ranking, docs_by_id: dict, gold: set[str], k: int, width: int) -> list[str]:
    """Render one retriever column as a list of lines of exactly ``width`` characters.

    Args:
        name: Column heading (``"BM25"`` or the model key).
        ranking: Full ranking as ``(doc_id, score)``, best first; only the first ``k`` are shown.
        docs_by_id: Document ID -> record, used for title and snippet.
        gold: Gold document IDs; hits are marked ``[GOLD]``.
        k: Number of results shown.
        width: Column width in characters; longer lines are cut, shorter ones padded.

    Returns:
        Lines of the column (heading, rule, then per result: rank/score line, title, up to 3 snippet lines, blank line).
    """
    lines = [name, "-" * width]
    # For each of the top k results (r = rank, d = doc ID, s = score): add a rank/score line,
    # the title and up to three snippet lines, so the column shows the retrieved passages.
    for r, (d, s) in enumerate(ranking[:k], 1):
        mark = "  [GOLD]" if d in gold else ""
        lines.append(f"{r}. {d}  Score {s:.3f}{mark}")
        lines.append("   " + title_and_body(docs_by_id[d])[0][:width - 3])
        # textwrap.wrap breaks the snippet into lines of at most `width` characters; [:3] keeps the first three.
        lines += textwrap.wrap(snippet(docs_by_id[d]), width, initial_indent="   ", subsequent_indent="   ")[:3]
        lines.append("")
    # List comprehension: pad or cut every line to exactly `width` characters so the two columns align.
    return [l.ljust(width)[:width] for l in lines]


def show(query: str, rk_bm25, rk_dense, docs_by_id: dict, gold: set[str], k: int, model: str) -> None:
    """Print both top-k columns side by side and, if ``gold`` is given, the gold ranks.

    Args:
        query: Query text, printed as heading.
        rk_bm25: Full BM25 ranking ``(doc_id, score)``.
        rk_dense: Full dense ranking ``(doc_id, score)``.
        docs_by_id: Document ID -> record.
        gold: Gold document IDs; empty for free-text queries (no marking, no gold rank line).
        k: Number of results per column.
        model: Model key, used as heading of the dense column.
    """
    cols = shutil.get_terminal_size((160, 24)).columns
    w = max(40, (cols - 3) // 2)
    print(f"\nFrage: {query}\n")
    # Generator over the two (heading, ranking) pairs: renders the BM25 column and the dense column; unpacking gives a
    # and b.
    a, b = (column(n, r, docs_by_id, gold, k, w) for n, r in (("BM25", rk_bm25), (model, rk_dense)))
    # For every output line index up to the longer column: print the BM25 line and the dense line side by side,
    # padding the shorter column with blanks.
    for i in range(max(len(a), len(b))):
        print(f"{a[i] if i < len(a) else ' ' * w} | {b[i] if i < len(b) else ''}".rstrip())
    if gold:
        from retrieval_metrics import best_gold_rank
        print(f"Gold-Passage(n): {', '.join(sorted(gold))} | Gold-Rang BM25: {best_gold_rank(rk_bm25, gold)}"
              f" | Gold-Rang {model}: {best_gold_rank(rk_dense, gold)} (volle Rangliste)")


def main() -> None:
    """Parse arguments, load passage vectors (cache or encode), then answer queries with BM25 and the dense model.

    Exits with status 1 and a message on invalid model/``--k``, missing data, or a model that cannot be loaded.
    Writes only to the embedding cache ``output/harness/cache/``.
    """
    # --- Parse and validate arguments ---
    from embedders import REGISTRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="e5-large", help=f"offene Modelle: {', '.join(OPEN_MODELS)}")
    g = ap.add_mutually_exclusive_group()  # g: either a free query or a dataset query ID
    g.add_argument("--query", help="freie Frage")
    g.add_argument("--query-id", help="Frage-ID aus queries.normalized.jsonl (markiert Gold-Passagen)")
    ap.add_argument("--interactive", action="store_true", help="Eingabeschleife für freie Fragen")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dataset-dir", type=Path, default=CODE / "output" / "germanquad")
    ap.add_argument("--warm-cache", action="store_true", help="nur Dokumentvektoren cachen und beenden")
    a = ap.parse_args()
    # OpenAI models are in the registry but need an API key; the demo allows only the local models.
    if a.model in REGISTRY and a.model not in OPEN_MODELS:
        fail(f"Modell {a.model!r} braucht einen API-Schlüssel und ist in der Demo nicht zugelassen. "
             f"Zulässig: {', '.join(OPEN_MODELS)}.")
    if a.model not in OPEN_MODELS:
        fail(f"Unbekanntes Modell {a.model!r}. Zulässig: {', '.join(OPEN_MODELS)}.")
    if a.k < 1:
        fail("--k muss mindestens 1 sein.")
    if not (a.warm_cache or a.query or a.query_id or a.interactive):
        ap.error("--query, --query-id, --interactive oder --warm-cache angeben")

    # --- Load data ---
    docs, queries, qrels = load_dataset(a.dataset_dir)
    # Dict comprehension: doc ID -> doc record (lookup table for titles and snippets).
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    ds = a.dataset_dir.resolve().name  # ds: dataset name, part of the cache file tag (same tag as in embed_eval.py)

    # --- Passage vectors: from the shared embedding cache, otherwise encoded now (slow on first use) ---
    from embed_eval import cached_encode, doc_text
    from embedders import Embedder, get_config
    from retrieval_metrics import build_gold
    emb = Embedder(get_config(a.model))  # emb: encoder with the registry config (prefixes, revision, normalization)
    cache = CODE / "output" / "harness" / "cache"
    t0 = time.time()
    try:
        # d_emb: passage embeddings in the order of doc_ids; hit: True if read from the cache
        # List comprehension: indexed text (title + text) of every passage in doc_ids order, so embedding row i =
        # doc_ids[i].
        d_emb, hit, _ = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, cache, f"{ds}_docs")
    except Exception as ex:  # noqa: BLE001
        fail(f"Modell {emb.cfg.model_name} (Revision {emb.cfg.revision}) nicht ladbar: {type(ex).__name__}: {ex}\n"
             "Gewichte einmalig mit Netzzugang laden (danach HF_HUB_OFFLINE=1 möglich).")
    print(f"Dokumentvektoren: {'aus Cache' if hit else 'neu berechnet'} ({time.time() - t0:.1f} s), Form {d_emb.shape}")
    if a.warm_cache:
        return

    # --- BM25 index and gold sets (threshold 1, as in the experiment) ---
    bm25, tokenize = build_bm25(doc_ids, docs_by_id)
    gold_by_q = build_gold(qrels, set(doc_ids), 1)

    def run(query: str, gold: set[str]) -> None:
        """Encode one query, rank with both retrievers and print the side-by-side result."""
        t = time.time()
        try:
            q_vec = emb.encode([query.strip()], True)
        except Exception as ex:  # noqa: BLE001
            fail(f"Modell {emb.cfg.model_name} nicht ladbar: {type(ex).__name__}: {str(ex)[:200]}\n"
                 "Gewichte einmalig mit Netzzugang laden; HF_HUB_OFFLINE=1 funktioniert mit "
                 "sentence-transformers 5.7 nicht (siehe code/README.md).")
        print(f"Query-Encoding (inkl. Modell-Laden beim ersten Aufruf): {time.time() - t:.2f} s")
        show(query, bm25_ranking(bm25, tokenize, doc_ids, query), dense_ranking(q_vec, d_emb, doc_ids),
             docs_by_id, gold, a.k, a.model)

    # --- Answer queries: dataset query by ID (with gold), free text, or interactive loop (both without gold) ---
    if a.query_id:
        # Generator over all queries: next() takes the first one whose ID equals --query-id, else None.
        q = next((q for q in queries if str(q["query_id"]) == a.query_id), None)
        if q is None:
            fail(f"Query-ID {a.query_id!r} nicht in {a.dataset_dir}/queries.normalized.jsonl.")
        run(str(q["query_text"]), gold_by_q.get(a.query_id, set()))
    if a.query:
        run(a.query, set())
    if a.interactive:
        # Interactive loop: read one free-text question per iteration and answer it; empty input or end of input ends
        # the loop.
        while True:
            try:
                q = input("\nFrage (leer = Ende): ").strip()
            except EOFError:
                break
            if not q:
                break
            run(q, set())


if __name__ == "__main__":
    main()
