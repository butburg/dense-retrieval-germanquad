"""Live-Demo: Top-k von BM25, BM25-de und einem Open-Weight-Embedding-Modell nebeneinander (nur CPU, kein API-Key).

Nutzung (im Ordner code/):
  python demo_query.py --query-id q40321                       # drei Spalten BM25 | BM25-de | bge-m3, k = 3
  python demo_query.py --query "Wer entwickelte den seillosen Aufzug?" --model e5-large --k 5
  python demo_query.py --query "Wie lange hält eine Glühbirne?" --target-id c41186 --width 170
                                                               # Rang der gewählten Passage c41186 je Verfahren
  python demo_query.py --warm-cache                            # Dokumentvektoren einmalig berechnen und cachen
Eingaben: {docs,queries,qrels}.normalized.jsonl in --dataset-dir (Default output/germanquad).
Ausgaben: Terminal; geschrieben wird nur der Embedding-Cache output/harness/cache/.
BM25, BM25-de und Embedding-Logik stammen aus scripts/run_bm25.py, pylib/german_text.py, embed_eval.py und
pylib/embedders.py.

Role in the experiment: presentation tool, independent of the evaluation. It shows for one query (a dataset query by
ID, or free text) the top-k passages of three retrievers in three columns: BM25 (regex tokenization), BM25-de (same index
parameters, stopword removal and Snowball stemming of pylib/german_text.py) and one open embedding model. Gold passages
are marked (only for ``--query-id``) and the gold rank of each retriever in its full ranking is printed below the
columns. With ``--target-id <doc_id>`` (combinable with ``--query`` and ``--query-id``) a chosen passage is marked
``[ZIEL]`` when it is among the top k, and its rank in the full ranking of each retriever is printed below the columns
("Ziel-Rang", computed by the same ``best_gold_rank`` as the gold rank) together with its title. If the target is also
a gold passage of the query, a result line carries both marks (``[GOLD]  [ZIEL]``). Both BM25 indexes are built by ``run_bm25.build_index``, the single definition shared with the experiment
scripts. Passage vectors are read from / written to the same cache as embed_eval.py (output/harness/cache/), so a run
after ``embed_eval.py --stage encode`` needs no re-encoding. Only the single query is encoded at runtime. The imports of
heavy modules are deferred into the functions so that ``--help`` and error messages work without them.

Offline mode: if the pinned model revision is complete in the local Hugging Face cache (config, weights and, for
remote-code models, the code repo), ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` are set before transformers is
imported, so no hub request delays the start. Without a cached model the online mode stays and a note appears on
stderr. Variables set in the environment (also ``HF_HUB_OFFLINE=0``) take precedence. With transformers 4.57.3 the
offline mode needs a small workaround in ``Embedder`` (``embedders.allow_offline_tokenizer_loading``).
"""
from __future__ import annotations

import argparse, os, shutil, sys, textwrap, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE))

OPEN_MODELS = ("e5-large", "bge-m3", "gte-multilingual-base", "jina-v3")  # local models allowed in the demo (no API key)
DS_FILES = ("docs", "queries", "qrels")  # normalized files read from the dataset directory
LEXICAL_VARIANTS = ("BM25", "BM25-de")  # lexical columns shown before the dense column
DEFAULT_MODEL = "bge-m3"  # dense column of the screencast
DEFAULT_K = 3  # results per column; three columns of k = 3 stay readable in a 1080p terminal
FALLBACK_WIDTH = 170  # total terminal width in characters when neither the terminal nor --width provides one
SEPARATOR = " | "  # between columns
SNIPPET_LINES = 3  # snippet lines per result
# Files of a usable cached model: config.json plus at least one weight file or weight index.
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                      "pytorch_model.bin.index.json")
OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")  # both are read by huggingface_hub / transformers
TRUE_VALUES = ("1", "on", "yes", "true")  # values that huggingface_hub treats as "switched on"
# One source for the hint after a failed model load (README: "Hinweis" in the section on demo_query.py).
OFFLINE_HINT = ("Fehlt das Modell im lokalen Hugging-Face-Cache, einmal mit Netzzugang laden und "
                "HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 setzen (siehe code/README.md).")


def fail(msg: str) -> None:
    """Print a clear error message and exit with status 1."""
    print(f"Fehler: {msg}", file=sys.stderr)
    sys.exit(1)


def hub_cache_dir(environ=os.environ) -> Path:
    """Folder of the local Hugging Face hub cache, resolved like ``huggingface_hub`` does (without importing it).

    Args:
        environ: Environment mapping, e.g. ``os.environ``; ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE``, ``HF_HOME``
            and ``XDG_CACHE_HOME`` are honoured in this order.

    Returns:
        Path of the cache folder that contains the ``models--<org>--<name>`` directories.
    """
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if environ.get(name):
            return Path(environ[name])
    if environ.get("HF_HOME"):
        return Path(environ["HF_HOME"]) / "hub"
    if environ.get("XDG_CACHE_HOME"):
        return Path(environ["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_snapshot(repo_id: str, revision: str | None, hub_cache: Path) -> Path | None:
    """Snapshot folder of a model repo in the local cache, or ``None``.

    Args:
        repo_id: Hub repo ID, e.g. ``"BAAI/bge-m3"``.
        revision: Commit hash of the snapshot; ``None`` follows the cached ``refs/main``.
        hub_cache: Cache folder from :func:`hub_cache_dir`.

    Returns:
        ``<hub_cache>/models--BAAI--bge-m3/snapshots/<revision>`` if it exists, else ``None``. No network access.
    """
    repo_dir = hub_cache / ("models--" + repo_id.replace("/", "--"))
    if revision is None:
        ref_file = repo_dir / "refs" / "main"
        if not ref_file.is_file():
            return None
        revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def is_model_cached(cfg, hub_cache: Path) -> bool:
    """True if the pinned model can be loaded from the local cache without network.

    The check reads the cache folders only. A snapshot counts as complete if it holds ``config.json`` and a weight
    file (a tokenizer-only snapshot does not count). For models with remote code (``trust_remote_code``), the pinned
    revision of the code repo (``cfg.code_repo`` at ``cfg.code_revision``) must hold at least one ``.py`` file as well.

    Args:
        cfg: ``EmbedderConfig`` of the model (``model_name``, ``revision``, ``code_repo``, ``code_revision``).
        hub_cache: Cache folder from :func:`hub_cache_dir`.

    Returns:
        ``True`` if all required files are present (symlinks must resolve to existing blobs).
    """
    snapshot = cached_snapshot(cfg.model_name, cfg.revision, hub_cache)
    if snapshot is None or not (snapshot / "config.json").is_file():
        return False
    if not any((snapshot / name).is_file() for name in MODEL_WEIGHT_FILES):
        return False
    if cfg.trust_remote_code and cfg.code_repo:
        code_snapshot = cached_snapshot(cfg.code_repo, cfg.code_revision, hub_cache)
        if code_snapshot is None or not any(code_snapshot.glob("*.py")):
            return False
    return True


def is_offline(environ=os.environ) -> bool:
    """True if ``HF_HUB_OFFLINE`` or ``TRANSFORMERS_OFFLINE`` is switched on in ``environ``."""
    return any(environ.get(name, "").lower() in TRUE_VALUES for name in OFFLINE_VARIABLES)


def choose_offline_mode(cfg, environ=os.environ, hub_cache: Path | None = None) -> bool:
    """Switch Hugging Face to offline mode when the model is fully cached; call before importing transformers.

    ``huggingface_hub`` and ``transformers`` read the variables once at import, so this must run first. An explicit
    value of ``HF_HUB_OFFLINE`` or ``TRANSFORMERS_OFFLINE`` in ``environ`` (also ``0``) is kept and nothing is set.
    If the model is not cached, the online mode stays and a short note goes to stderr.

    Args:
        cfg: ``EmbedderConfig`` of the chosen model.
        environ: Mapping that receives ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` (``os.environ`` in use).
        hub_cache: Cache folder; ``None`` resolves it from ``environ``.

    Returns:
        ``True`` if the process runs offline afterwards (set here or by the caller), else ``False``.
    """
    if any(name in environ for name in OFFLINE_VARIABLES):
        return is_offline(environ)
    if is_model_cached(cfg, hub_cache or hub_cache_dir(environ)):
        for name in OFFLINE_VARIABLES:
            environ.setdefault(name, "1")
        return True
    print(f"Hinweis: {cfg.model_name} liegt nicht vollständig im lokalen Hugging-Face-Cache und wird einmalig "
          f"mit Netzzugang geladen.", file=sys.stderr)
    return False


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


def build_bm25(doc_ids: list[str], docs_by_id: dict, variant: str = "BM25"):
    """BM25 index of one lexical variant with the parameters of scripts/run_bm25.py.

    Args:
        doc_ids: Document IDs in index order, e.g. ``["c1", "c2"]``.
        docs_by_id: Document ID -> normalized document record.
        variant: ``"BM25"`` (lowercase, regex tokens) or ``"BM25-de"`` (additionally stopword removal and Snowball
            stemming, ``preprocess_de``); documents and queries use the same function.

    Returns:
        Tuple ``(bm25, tokenize)``: the ``BM25Okapi`` index (rows in the order of ``doc_ids``) and the tokenizer
        that must be applied to queries.

    Raises:
        ValueError: If ``variant`` is not in ``LEXICAL_VARIANTS``.
    """
    if variant not in LEXICAL_VARIANTS:
        raise ValueError(f"Unbekannte lexikalische Variante {variant!r}; erlaubt: {', '.join(LEXICAL_VARIANTS)}")
    try:
        import run_bm25 as rb
        from german_text import preprocess_de, tokenize_de
    except ImportError as ex:
        fail(f"BM25-Abhängigkeiten fehlen ({ex}). Umgebung: pip install -r code/requirements-embed.txt")
    tokenize = tokenize_de if variant == "BM25" else preprocess_de
    return rb.build_index(doc_ids, docs_by_id, tokenize), tokenize


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


def snippet_lines(doc: dict, width: int, max_lines: int = SNIPPET_LINES) -> list[str]:
    """Whitespace-collapsed passage start, wrapped to at most ``max_lines`` indented lines.

    Args:
        doc: Normalized document record.
        width: Maximum line length including the three-space indent.
        max_lines: Number of lines kept; a cut text ends with an ellipsis.

    Returns:
        Lines of the passage body ("=" characters of wiki headings are replaced by spaces).
    """
    # split() cuts at any whitespace run, join(" ") glues the words with single spaces: collapses line breaks and
    # repeated blanks.
    body = " ".join(title_and_body(doc)[1].replace("=", " ").split())
    return textwrap.wrap(body, width, initial_indent="   ", subsequent_indent="   ", max_lines=max_lines,
                         placeholder=" …")


def column(name: str, ranking, docs_by_id: dict, gold: set[str], k: int, width: int,
           target_id: str | None = None) -> list[str]:
    """Render one retriever column as a list of lines of exactly ``width`` characters.

    Args:
        name: Column heading (``"BM25"``, ``"BM25-de"`` or the model key).
        ranking: Full ranking as ``(doc_id, score)``, best first; only the first ``k`` are shown.
        docs_by_id: Document ID -> record, used for title and snippet.
        gold: Gold document IDs; hits are marked ``[GOLD]``. If not empty, the column ends with the gold rank in the
            full ranking.
        k: Number of results shown.
        width: Column width in characters; longer lines are cut, shorter ones padded.
        target_id: Optional passage chosen by the presenter; marked ``[ZIEL]`` among the top k. The column ends with
            its rank in the full ranking. A target that is also gold gets both marks.

    Returns:
        Lines of the column (heading, rule, then per result: rank/score line, title, up to 3 snippet lines, blank
        line; with ``gold`` or ``target_id`` finally a rule, the gold rank and the target rank).
    """
    from retrieval_metrics import best_gold_rank
    lines = [name, "-" * width]
    # For each of the top k results (rank, doc ID, score): add a rank/score line, the title and the snippet lines,
    # so the column shows the retrieved passages.
    for rank, (doc_id, score) in enumerate(ranking[:k], 1):
        mark = ("  [GOLD]" if doc_id in gold else "") + ("  [ZIEL]" if doc_id == target_id else "")
        lines.append(f"{rank}. {doc_id}  Score {score:.3f}{mark}")
        lines.append("   " + title_and_body(docs_by_id[doc_id])[0][:width - 3])
        lines += snippet_lines(docs_by_id[doc_id], width)
        lines.append("")
    if gold or target_id:
        lines.append("-" * width)
    if gold:
        lines.append(f"Gold-Rang: {best_gold_rank(ranking, gold)} (volle Rangliste)")
    if target_id:
        # The target is a one-element "gold set": same rank function, rank in the full ranking.
        lines.append(f"Ziel-Rang: {best_gold_rank(ranking, {target_id})} (volle Rangliste)")
    # List comprehension: pad or cut every line to exactly `width` characters so the columns align.
    return [line.ljust(width)[:width] for line in lines]


def show(query: str, rankings: list[tuple[str, list]], docs_by_id: dict, gold: set[str], k: int,
         total_width: int | None = None, target_id: str | None = None) -> None:
    """Print the top-k columns side by side; with ``gold`` the gold marks and gold ranks are included.

    Args:
        query: Query text, printed as heading.
        rankings: ``(column heading, full ranking)`` per retriever in display order, e.g.
            ``[("BM25", rk_bm25), ("BM25-de", rk_de), ("bge-m3", rk_dense)]``; a ranking is ``(doc_id, score)``
            pairs, best first.
        docs_by_id: Document ID -> record.
        gold: Gold document IDs; empty for free-text queries (no marking, no gold rank).
        k: Number of results per column.
        total_width: Total line width in characters; ``None`` uses the terminal width (``FALLBACK_WIDTH`` if unknown).
        target_id: Optional chosen passage (must be a key of ``docs_by_id``): marked ``[ZIEL]``, with target rank per
            column and, below the columns, a line with its ID and title.
    """
    if total_width is None:
        total_width = shutil.get_terminal_size((FALLBACK_WIDTH, 24)).columns
    n_columns = len(rankings)
    # Equal column widths; the separators between the columns take the rest of the line.
    width = max(30, (total_width - len(SEPARATOR) * (n_columns - 1)) // n_columns)
    print(f"\nFrage: {query}\n")
    # One rendered column (list of lines) per retriever.
    columns = [column(name, ranking, docs_by_id, gold, k, width, target_id) for name, ranking in rankings]
    # Print line i of all columns joined by the separator; shorter columns are padded with blanks.
    for i in range(max(len(c) for c in columns)):
        print(SEPARATOR.join(c[i] if i < len(c) else " " * width for c in columns).rstrip())
    if target_id:
        print(f"\nZiel-Passage: {target_id}  {title_and_body(docs_by_id[target_id])[0]}")
    if gold:
        print(f"\nGold-Passage(n): {', '.join(sorted(gold))}  ([GOLD] = Gold-Passage unter den Top {k})")


def main() -> None:
    """Parse arguments, load passage vectors (cache or encode), then answer queries with BM25, BM25-de and the dense model.

    Exits with status 1 and a message on invalid model/``--k``/``--width``, missing data, or a model that cannot be
    loaded.
    Writes only to the embedding cache ``output/harness/cache/``.
    """
    # --- Parse and validate arguments ---
    from embedders import REGISTRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Open-Weight-Modelle: {', '.join(OPEN_MODELS)} (Default: %(default)s)")
    g = ap.add_mutually_exclusive_group()  # g: either a free query or a dataset query ID
    g.add_argument("--query", help="freie Frage")
    g.add_argument("--query-id", help="Frage-ID aus queries.normalized.jsonl (markiert Gold-Passagen)")
    ap.add_argument("--target-id", help="doc_id einer gewählten Passage: Rang je Verfahren (Marker [ZIEL], Fußzeile "
                                        "\"Ziel-Rang\"); kombinierbar mit --query und --query-id")
    ap.add_argument("--interactive", action="store_true", help="Eingabeschleife für freie Fragen")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Treffer je Spalte (Default: %(default)s)")
    ap.add_argument("--width", type=int, help=f"Gesamtbreite der Ausgabe in Zeichen (Default: Terminalbreite, sonst {FALLBACK_WIDTH})")
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
    if a.width is not None and a.width < 100:
        fail("--width muss mindestens 100 sein (drei Spalten).")
    if not (a.warm_cache or a.query or a.query_id or a.interactive):
        ap.error("--query, --query-id, --interactive oder --warm-cache angeben")

    # --- Load data ---
    docs, queries, qrels = load_dataset(a.dataset_dir)
    # Dict comprehension: doc ID -> doc record (lookup table for titles and snippets).
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    if a.target_id and a.target_id not in docs_by_id:
        fail(f"Ziel-ID {a.target_id!r} nicht in {a.dataset_dir}/docs.normalized.jsonl.")
    ds = a.dataset_dir.resolve().name  # ds: dataset name, part of the cache file tag (same tag as in embed_eval.py)

    # --- Passage vectors: from the shared embedding cache, otherwise encoded now (slow on first use) ---
    from embed_eval import cached_encode, doc_text
    from embedders import Embedder, get_config
    from retrieval_metrics import build_gold
    cfg = get_config(a.model)
    # Offline decision before the first import of transformers / huggingface_hub (they read the variables at import).
    choose_offline_mode(cfg)
    emb = Embedder(cfg)  # emb: encoder with the registry config (prefixes, revision, normalization)
    cache = CODE / "output" / "harness" / "cache"
    t0 = time.time()
    try:
        # d_emb: passage embeddings in the order of doc_ids; hit: True if read from the cache
        # List comprehension: indexed text (title + text) of every passage in doc_ids order, so embedding row i =
        # doc_ids[i].
        d_emb, hit, _ = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, cache, f"{ds}_docs")
    except Exception as ex:  # noqa: BLE001
        fail(f"Modell {emb.cfg.model_name} (Revision {emb.cfg.revision}) nicht ladbar: {type(ex).__name__}: {ex}\n"
             + OFFLINE_HINT)
    print(f"Dokumentvektoren: {'aus Cache' if hit else 'neu berechnet'} ({time.time() - t0:.1f} s), Form {d_emb.shape}")
    if a.warm_cache:
        return

    # --- Both BM25 indexes and gold sets (threshold 1, as in the experiment) ---
    # Dict comprehension: column heading -> (index, tokenizer) for BM25 and BM25-de.
    lexical = {variant: build_bm25(doc_ids, docs_by_id, variant) for variant in LEXICAL_VARIANTS}
    gold_by_q = build_gold(qrels, set(doc_ids), 1)

    def run(query: str, gold: set[str]) -> None:
        """Encode one query, rank with all three retrievers and print the side-by-side result."""
        t = time.time()
        try:
            q_vec = emb.encode([query.strip()], True)
        except Exception as ex:  # noqa: BLE001
            fail(f"Modell {emb.cfg.model_name} nicht ladbar: {type(ex).__name__}: {str(ex)[:200]}\n" + OFFLINE_HINT)
        print(f"Query-Encoding (inkl. Modell-Laden beim ersten Aufruf): {time.time() - t:.2f} s")
        # List comprehension: one (heading, full ranking) pair per lexical variant, then the dense column.
        rankings = [(variant, bm25_ranking(bm25, tokenize, doc_ids, query))
                    for variant, (bm25, tokenize) in lexical.items()]
        rankings.append((a.model, dense_ranking(q_vec, d_emb, doc_ids)))
        show(query, rankings, docs_by_id, gold, a.k, a.width, a.target_id)

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
