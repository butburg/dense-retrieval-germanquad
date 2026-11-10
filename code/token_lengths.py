"""Token lengths of the GermanQuAD passages for all nine embedding models (incl. special tokens and doc prefix).

Usage: python token_lengths.py   ->   output/harness/token_lengths.json
Only tokenizers are loaded (pinned revisions from the registry), no model weights and no API calls.

Role in the experiment: supplementary check of input truncation. For every model it records the token-length
distribution of the passages, how many passages exceed 512 tokens (sensitivity analysis --max-seq-length 512) and how
many exceed the model's effective limit in the main run (max_seq_length_effective from
output/harness/<model>/metrics_germanquad.json, section "experiment"), i.e. would actually be truncated there.
Tokenizers: Hugging Face tokenizers for the seven local models; for the two OpenAI models the tiktoken encoding that
tiktoken.encoding_for_model assigns to them (cl100k_base). OpenAI documents no limit in the run protocol, so
effective_limit and n_over_limit are null for these; the counts of tiktoken exclude special tokens because the API
adds none. A model whose tokenizer cannot be loaded is recorded as {"measured": false, "reason": ...}, never estimated.
Independent of the embed_eval.py results; reads output/germanquad/docs.normalized.jsonl and needs the Hugging Face
tokenizers (network or local HF cache) and, for OpenAI, tiktoken (downloads the encoding file once). The resulting JSON
is shown in notebook 05 and its values are quoted in the manuscript (sections 3.3, 4.4, appendix B).
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

LIMIT = 512  # token limit whose exceedance is counted for every model
HF_MODELS = ("e5-large", "bge-m3", "gte-multilingual-base", "jina-v3",
             "qwen3-embedding-0.6b", "arctic-embed-l-v2", "jina-v2-base-de")
OPENAI_MODELS = ("openai-3-small", "openai-3-large")
# Counting rule written to the result file; differs between the two tokenizer families.
COUNTED = ("local models: add_special_tokens=True, doc prefix prepended, no truncation; "
           "OpenAI models (tiktoken cl100k_base): no special tokens (the API adds none), no doc prefix, no truncation")


def effective_limit(key: str) -> int | None:
    """Return max_seq_length_effective of the main run of a model, or None if the run protocol lacks it."""
    protocol = CODE / "output" / "harness" / key / "metrics_germanquad.json"
    if not protocol.exists():
        return None
    return json.loads(protocol.read_text())["experiment"].get("max_seq_length_effective")


def summarize(key: str, base: dict, token_counts: np.ndarray) -> dict:
    """Summary record of one model: base fields plus exceedance counts and length statistics."""
    limit = effective_limit(key)
    # (token_counts > LIMIT) is a boolean mask over all passages, e.g. [300, 700, 520] -> [False, True, True]; .sum() counts
    # the Trues (here 2 passages longer than 512 tokens). median/max/min summarize the length distribution.
    n_over_limit = int((token_counts > limit).sum()) if limit is not None else None
    return {**base, "n_over_512": int((token_counts > LIMIT).sum()), "effective_limit": limit, "n_over_limit": n_over_limit,
            "median": float(np.median(token_counts)), "max": int(token_counts.max()), "min": int(token_counts.min())}


def count_hf_tokens(key: str, texts: list[str]) -> dict:
    """Token counts of one local model: pinned tokenizer, special tokens and doc prefix included, no truncation."""
    from transformers import AutoTokenizer  # noqa: E402
    config = get_config(key)  # model config (name, pinned revision, document prefix)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
    # The tokenizer returns one input_ids list per passage; its length is the token count of that passage.
    encoded = tokenizer([config.doc_prefix + text for text in texts], add_special_tokens=True, truncation=False)["input_ids"]
    token_counts = np.array([len(ids) for ids in encoded])
    return summarize(key, {"model_name": config.model_name, "revision": config.revision}, token_counts)


def count_openai_tokens(key: str, texts: list[str]) -> dict:
    """Token counts of one OpenAI model with its tiktoken encoding (no API call, no truncation)."""
    import tiktoken  # noqa: E402
    config = get_config(key)
    # Source of the model -> encoding mapping: tiktoken.encoding_for_model (text-embedding-3-* -> cl100k_base).
    encoding = tiktoken.encoding_for_model(config.model_name)
    # disallowed_special=() counts special-token strings occurring in passage text as ordinary text instead of raising.
    token_counts = np.array([len(encoding.encode(config.doc_prefix + text, disallowed_special=())) for text in texts])
    base = {"model_name": config.model_name, "revision": None, "tokenizer": f"tiktoken {tiktoken.__version__}, {encoding.name}"}
    return summarize(key, base, token_counts)


def main() -> None:
    """Count passage token lengths for all nine models and write output/harness/token_lengths.json."""
    # --- Load passages; texts use the same title+text composition as the embedding run ---
    docs = load_jsonl(CODE / "output" / "germanquad" / "docs.normalized.jsonl")
    # Indexed text (title + text) of each passage, same composition as in the embedding run.
    texts = [doc_text(d) for d in docs]
    # res: result record; per-model statistics are added below
    res = {"dataset": "germanquad", "n_passages": len(texts), "counted": COUNTED,
           "threshold": LIMIT, "models": {}}
    import transformers  # noqa: E402
    res["transformers"] = transformers.__version__
    # --- Tokenize all passages per model and summarise the length distribution ---
    for key, counter in [(k, count_hf_tokens) for k in HF_MODELS] + [(k, count_openai_tokens) for k in OPENAI_MODELS]:
        try:
            res["models"][key] = counter(key, texts)
        except Exception as error:  # unavailable tokenizer: report, never estimate
            res["models"][key] = {"model_name": get_config(key).model_name, "measured": False,
                                  "reason": f"{type(error).__name__}: {error}"}
        print(key, res["models"][key])
    # --- Write result ---
    out = CODE / "output" / "harness" / "token_lengths.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
