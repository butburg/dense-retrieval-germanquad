"""Token lengths of the GermanQuAD passages per local model tokenizer (incl. special tokens and doc prefix).

Usage: python token_lengths.py   ->   output/harness/token_lengths.json
Only tokenizers are loaded (pinned revisions from the registry), no model weights.
"""
import json, sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
from embed_eval import doc_text  # noqa: E402
from embedders import get_config  # noqa: E402
from local_dataset_io import load_jsonl  # noqa: E402

LIMIT = 512
docs = load_jsonl(CODE / "output" / "germanquad" / "docs.normalized.jsonl")
texts = [doc_text(d) for d in docs]
res = {"dataset": "germanquad", "n_passages": len(texts), "counted": "add_special_tokens=True, doc prefix prepended, no truncation",
       "threshold": LIMIT, "models": {}}
from transformers import AutoTokenizer  # noqa: E402
import transformers  # noqa: E402
res["transformers"] = transformers.__version__
for key in ("e5-large", "bge-m3", "gte-multilingual-base", "jina-v3"):
    c = get_config(key)
    tok = AutoTokenizer.from_pretrained(c.model_name, revision=c.revision)
    n = np.array([len(x) for x in tok([c.doc_prefix + t for t in texts], add_special_tokens=True, truncation=False)["input_ids"]])
    res["models"][key] = {"model_name": c.model_name, "revision": c.revision, "n_over_512": int((n > LIMIT).sum()),
                          "median": float(np.median(n)), "max": int(n.max()), "min": int(n.min())}
    print(key, res["models"][key])
out = CODE / "output" / "harness" / "token_lengths.json"
out.write_text(json.dumps(res, indent=2))
