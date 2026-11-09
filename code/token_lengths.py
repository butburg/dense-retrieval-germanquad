"""Token lengths of the GermanQuAD passages per local model tokenizer (incl. special tokens and doc prefix).

Usage: python token_lengths.py   ->   output/harness/token_lengths.json
Only tokenizers are loaded (pinned revisions from the registry), no model weights.

Role in the experiment: supplementary check for the 512-token sensitivity analysis (--max-seq-length 512). For e5-large
and the three models that are also run with that setting (bge-m3, gte-multilingual-base, jina-v3) it counts how many
passages exceed 512 tokens, i.e. would be truncated at that limit. Independent of the embed_eval.py results; reads
output/germanquad/docs.normalized.jsonl and needs the Hugging Face tokenizers (network or local HF cache). The
resulting JSON is shown in notebook 05 and its values are quoted in the manuscript (sections 3.3, 4.4, appendix B).
The computation runs in main(); importing the module has no side effects (no file is written, no tokenizer is loaded),
and running the script overwrites output/harness/token_lengths.json.
"""
import json, sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
from embedders import get_config  # noqa: E402
from local_dataset_io import doc_text, load_jsonl  # noqa: E402

LIMIT = 512  # token limit whose exceedance is counted


def main() -> None:
    """Count passage token lengths for the four local models and write output/harness/token_lengths.json."""
    # --- Load passages; texts use the same title+text composition as the embedding run ---
    docs = load_jsonl(CODE / "output" / "germanquad" / "docs.normalized.jsonl")
    # List comprehension: indexed text (title + text) of each passage, same composition as in the embedding run.
    texts = [doc_text(d) for d in docs]
    # res: result record; per-model statistics are added below
    res = {"dataset": "germanquad", "n_passages": len(texts), "counted": "add_special_tokens=True, doc prefix prepended, no truncation",
           "threshold": LIMIT, "models": {}}
    from transformers import AutoTokenizer  # noqa: E402
    import transformers  # noqa: E402
    res["transformers"] = transformers.__version__
    # --- Tokenize all passages per model and summarise the length distribution ---
    # For every local model: load only its tokenizer, count the tokens of every passage and store summary statistics
    # (how many passages would be truncated at 512 tokens).
    for key in ("e5-large", "bge-m3", "gte-multilingual-base", "jina-v3"):
        c = get_config(key)  # c: model config (name, pinned revision, document prefix)
        tok = AutoTokenizer.from_pretrained(c.model_name, revision=c.revision)
        # n: token count per passage (special tokens and doc prefix included, no truncation)
        # Inner comprehension: prepend the doc prefix to every passage; the tokenizer returns one input_ids list per
        # passage;
        # the outer comprehension takes its length -> array n with one token count per passage.
        n = np.array([len(x) for x in tok([c.doc_prefix + t for t in texts], add_special_tokens=True, truncation=False)["input_ids"]])
        # (n > LIMIT) is a boolean mask over all passages, e.g. n = [300, 700, 520] -> [False, True, True]; .sum() counts
        # the Trues (here 2 passages longer than 512 tokens). median/max/min summarize the length distribution.
        res["models"][key] = {"model_name": c.model_name, "revision": c.revision, "n_over_512": int((n > LIMIT).sum()),
                              "median": float(np.median(n)), "max": int(n.max()), "min": int(n.min())}
        print(key, res["models"][key])
    # --- Write result ---
    out = CODE / "output" / "harness" / "token_lengths.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
