"""Live-Demo: Top-k von BM25 und einem offenen Embedding-Modell nebeneinander (nur CPU, kein API-Key).

Nutzung (im Ordner code/):
  python demo_query.py --query-id q40369 --k 5
  python demo_query.py --query "Wer entwickelte den seillosen Aufzug?" --model e5-large
  python demo_query.py --warm-cache            # Dokumentvektoren einmalig berechnen und cachen
Eingaben: {docs,queries,qrels}.normalized.jsonl in --dataset-dir (Default output/germanquad).
Ausgaben: Terminal; geschrieben wird nur der Embedding-Cache output/harness/cache/.
BM25 und Embedding-Logik stammen aus scripts/run_bm25.py, embed_eval.py und pylib/embedders.py.
"""
from __future__ import annotations

import argparse, shutil, sys, textwrap, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE))

OPEN_MODELS = ("e5-large", "bge-m3", "gte-multilingual-base", "jina-v3")
DS_FILES = ("docs", "queries", "qrels")


def fail(msg: str) -> None:
    """Print a clear error message and exit with status 1."""
    print(f"Fehler: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dataset(dataset_dir: Path):
    """Load docs, queries and qrels; abort with a hint if the normalized files are missing."""
    from local_dataset_io import load_jsonl
    paths = {n: dataset_dir / f"{n}.normalized.jsonl" for n in DS_FILES}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        fail("Normalisierte Daten fehlen: " + ", ".join(missing)
             + ". Erzeugen mit den Notebooks 01 und 02 (Ordner code/).")
    return [load_jsonl(paths[n]) for n in DS_FILES]


def build_bm25(doc_ids: list[str], docs_by_id: dict):
    """BM25 index with the parameters of scripts/run_bm25.py."""
    try:
        import run_bm25 as rb
    except ImportError as ex:
        fail(f"BM25-Abhängigkeiten fehlen ({ex}). Umgebung: pip install -r code/requirements-embed.txt")
    bm25 = rb.BM25Okapi([rb.tokenize_de(rb.doc_text(docs_by_id[i])) for i in doc_ids],
                        k1=rb.BM25_K1, b=rb.BM25_B, epsilon=rb.BM25_EPSILON)
    return bm25, rb.tokenize_de


def bm25_ranking(bm25, tokenize, doc_ids: list[str], query: str) -> list[tuple[str, float]]:
    """Full BM25 ranking, ties broken by (-score, doc_id) as in run_bm25.py."""
    scores = bm25.get_scores(tokenize(query))
    return sorted(zip(doc_ids, (float(s) for s in scores)), key=lambda it: (-it[1], it[0]))


def dense_ranking(q_vec, d_emb, doc_ids: list[str]) -> list[tuple[str, float]]:
    """Full ranking by dot product of L2-normalised vectors (stable order for ties)."""
    import numpy as np
    sims = (q_vec.astype(np.float32) @ d_emb.astype(np.float32).T).ravel()
    order = np.argsort(-sims, kind="stable")
    return [(doc_ids[int(i)], float(sims[i])) for i in order]


def gold_rank(ranking: list[tuple[str, float]], gold: set[str]) -> int | None:
    """1-based rank of the best-ranked gold passage in the full ranking."""
    return next((r for r, (d, _) in enumerate(ranking, 1) if d in gold), None)


def title_and_body(doc: dict) -> tuple[str, str]:
    """Title and passage body; falls back to the first text line when the title field is empty."""
    title, text = str(doc.get("title") or "").strip(), str(doc.get("text") or "").strip()
    if title:
        return title, text
    head, _, rest = text.partition("\n")
    return head.strip(), rest.strip() or head.strip()


def snippet(doc: dict, n: int = 150) -> str:
    """Whitespace-collapsed passage start of about ``n`` characters."""
    body = " ".join(title_and_body(doc)[1].replace("=", " ").split())
    return body if len(body) <= n else body[:n].rstrip() + " …"


def column(name: str, ranking, docs_by_id: dict, gold: set[str], k: int, width: int) -> list[str]:
    """Render one retriever column as a list of lines of exactly ``width`` characters."""
    lines = [name, "-" * width]
    for r, (d, s) in enumerate(ranking[:k], 1):
        mark = "  [GOLD]" if d in gold else ""
        lines.append(f"{r}. {d}  Score {s:.3f}{mark}")
        lines.append("   " + title_and_body(docs_by_id[d])[0][:width - 3])
        lines += textwrap.wrap(snippet(docs_by_id[d]), width, initial_indent="   ", subsequent_indent="   ")[:3]
        lines.append("")
    return [l.ljust(width)[:width] for l in lines]


def show(query: str, rk_bm25, rk_dense, docs_by_id: dict, gold: set[str], k: int, model: str) -> None:
    """Print both top-k columns side by side and, if ``gold`` is given, the gold ranks."""
    cols = shutil.get_terminal_size((160, 24)).columns
    w = max(40, (cols - 3) // 2)
    print(f"\nFrage: {query}\n")
    a, b = (column(n, r, docs_by_id, gold, k, w) for n, r in (("BM25", rk_bm25), (model, rk_dense)))
    for i in range(max(len(a), len(b))):
        print(f"{a[i] if i < len(a) else ' ' * w} | {b[i] if i < len(b) else ''}".rstrip())
    if gold:
        print(f"Gold-Passage(n): {', '.join(sorted(gold))} | Gold-Rang BM25: {gold_rank(rk_bm25, gold)}"
              f" | Gold-Rang {model}: {gold_rank(rk_dense, gold)} (volle Rangliste)")


def main() -> None:
    from embedders import REGISTRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="e5-large", help=f"offene Modelle: {', '.join(OPEN_MODELS)}")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--query", help="freie Frage")
    g.add_argument("--query-id", help="Frage-ID aus queries.normalized.jsonl (markiert Gold-Passagen)")
    ap.add_argument("--interactive", action="store_true", help="Eingabeschleife für freie Fragen")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dataset-dir", type=Path, default=CODE / "output" / "germanquad")
    ap.add_argument("--warm-cache", action="store_true", help="nur Dokumentvektoren cachen und beenden")
    a = ap.parse_args()
    if a.model in REGISTRY and a.model not in OPEN_MODELS:
        fail(f"Modell {a.model!r} braucht einen API-Schlüssel und ist in der Demo nicht zugelassen. "
             f"Zulässig: {', '.join(OPEN_MODELS)}.")
    if a.model not in OPEN_MODELS:
        fail(f"Unbekanntes Modell {a.model!r}. Zulässig: {', '.join(OPEN_MODELS)}.")
    if a.k < 1:
        fail("--k muss mindestens 1 sein.")
    if not (a.warm_cache or a.query or a.query_id or a.interactive):
        ap.error("--query, --query-id, --interactive oder --warm-cache angeben")

    docs, queries, qrels = load_dataset(a.dataset_dir)
    docs_by_id = {str(d["doc_id"]): d for d in docs}
    doc_ids = list(docs_by_id)
    ds = a.dataset_dir.resolve().name

    from embed_eval import cached_encode, doc_text
    from embedders import Embedder, get_config
    from retrieval_metrics import build_gold
    emb = Embedder(get_config(a.model))
    cache = CODE / "output" / "harness" / "cache"
    t0 = time.time()
    try:
        d_emb, hit, _ = cached_encode(emb, [doc_text(docs_by_id[i]) for i in doc_ids], False, cache, f"{ds}_docs")
    except Exception as ex:  # noqa: BLE001
        fail(f"Modell {emb.cfg.model_name} (Revision {emb.cfg.revision}) nicht ladbar: {type(ex).__name__}: {ex}\n"
             "Gewichte einmalig mit Netzzugang laden (danach HF_HUB_OFFLINE=1 möglich).")
    print(f"Dokumentvektoren: {'aus Cache' if hit else 'neu berechnet'} ({time.time() - t0:.1f} s), Form {d_emb.shape}")
    if a.warm_cache:
        return

    bm25, tokenize = build_bm25(doc_ids, docs_by_id)
    gold_by_q = build_gold(qrels, set(doc_ids), 1)

    def run(query: str, gold: set[str]) -> None:
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

    if a.query_id:
        q = next((q for q in queries if str(q["query_id"]) == a.query_id), None)
        if q is None:
            fail(f"Query-ID {a.query_id!r} nicht in {a.dataset_dir}/queries.normalized.jsonl.")
        run(str(q["query_text"]), gold_by_q.get(a.query_id, set()))
    if a.query:
        run(a.query, set())
    if a.interactive:
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
