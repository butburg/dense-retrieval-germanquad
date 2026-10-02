"""Embedder registry and backends (sentence-transformers, OpenAI) for the evaluation harness."""
from __future__ import annotations

import os
import platform
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np

OPENAI_KEY_ENV = "EMBED_OPENAI_API_KEY"


@dataclass(frozen=True)
class EmbedderConfig:
    """Static configuration of one embedding model."""
    key: str
    backend: str  # "sentence-transformers" | "openai"
    model_name: str
    query_prefix: str = ""
    doc_prefix: str = ""
    normalize: bool = True
    batch_size: int = 32
    max_seq_length: int | None = None
    revision: str | None = None  # pinned HF commit hash of the model repo; None = hub HEAD
    trust_remote_code: bool = False
    code_repo: str | None = None  # HF repo providing remote code (auto_map), if any
    code_revision: str | None = None  # expected commit of code_repo; checked against the loaded module dir
    query_task: str | None = None  # sentence-transformers encode(task=...) adapter (jina-v3)
    doc_task: str | None = None
    query_prompt_name: str | None = None  # sentence-transformers encode(prompt_name=...) for queries (Qwen3)
    extra: dict = field(default_factory=dict)


# Registration is not a model selection; entries are candidates only.
REGISTRY: dict[str, EmbedderConfig] = {c.key: c for c in [
    EmbedderConfig("e5-large", "sentence-transformers", "intfloat/multilingual-e5-large",
                   query_prefix="query: ", doc_prefix="passage: ", batch_size=32,
                   revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"),
    # BGE-M3, dense embedding only (no sparse/ColBERT). Model card: https://huggingface.co/BAAI/bge-m3
    # -> no instruction/prefix for queries or passages; max_length 8192 (model default kept).
    EmbedderConfig("bge-m3", "sentence-transformers", "BAAI/bge-m3", batch_size=16,
                   revision="5617a9f61b028005a4858fdac845db406aefb181"),
    # Model card: https://huggingface.co/Alibaba-NLP/gte-multilingual-base
    # -> no prefixes; usage via sentence-transformers with trust_remote_code=True; max 8192 tokens
    # (model default kept). Remote code repo Alibaba-NLP/new-impl is pinned via code_revision (checked after load).
    EmbedderConfig("gte-multilingual-base", "sentence-transformers", "Alibaba-NLP/gte-multilingual-base",
                   batch_size=16, trust_remote_code=True,
                   revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
                   code_repo="Alibaba-NLP/new-impl", code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4"),
    # Model card: https://huggingface.co/jinaai/jina-embeddings-v3
    # -> LoRA task adapters via encode(task=...): "retrieval.query" for queries, "retrieval.passage" for
    # documents (asymmetric retrieval); no text prefix (adapter adds its instruction internally);
    # max 8192 tokens (model default kept). Remote code repo jinaai/xlm-roberta-flash-implementation pinned via code_revision.
    EmbedderConfig("jina-v3", "sentence-transformers", "jinaai/jina-embeddings-v3", batch_size=4,
                   trust_remote_code=True, query_task="retrieval.query", doc_task="retrieval.passage",
                   revision="ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
                   code_repo="jinaai/xlm-roberta-flash-implementation",
                   code_revision="bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"),
    # --- Weitere Modelle der neun-Modelle-Liste (compare_models.MODELS) ---
    # Model card + config_sentence_transformers.json: prompts {"query": "Instruct: Given a web search query,
    # retrieve relevant passages that answer the query\nQuery:", "document": ""} -> prompt_name="query" for queries only.
    # Native context 32k; max_seq_length capped at 8192 (longest GermanQuAD passage ~2.7k tokens -> no truncation).
    EmbedderConfig("qwen3-embedding-0.6b", "sentence-transformers", "Qwen/Qwen3-Embedding-0.6B", batch_size=8,
                   query_prompt_name="query", max_seq_length=8192,
                   revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    # config_sentence_transformers.json: prompts {"query": "query: "}; documents without prefix. Native 8192.
    EmbedderConfig("arctic-embed-l-v2", "sentence-transformers", "Snowflake/snowflake-arctic-embed-l-v2.0",
                   query_prefix="query: ", batch_size=16,
                   revision="ac6544c8a46e00af67e330e85a9028c66b8cfd9a"),
    # Model card: no prefix; native 8192 (ALiBi); remote code jinaai/jina-bert-implementation pinned via code_revision.
    EmbedderConfig("jina-v2-base-de", "sentence-transformers", "jinaai/jina-embeddings-v2-base-de", batch_size=8,
                   trust_remote_code=True, revision="3f9eede875721714945b6a99a3198299243cf2be",
                   code_repo="jinaai/jina-bert-implementation",
                   code_revision="f3ec4cf7de7e561007f27c9efc7148b0bd713f81"),
    EmbedderConfig("openai-3-small", "openai", "text-embedding-3-small", batch_size=128),
    EmbedderConfig("openai-3-large", "openai", "text-embedding-3-large", batch_size=128),
]}


def get_config(key: str) -> EmbedderConfig:
    """Look up a registered config.

    Raises:
        KeyError: If ``key`` is not registered.
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[key]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class Embedder:
    """Encodes texts for one config; prefixes and normalisation applied here."""

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self._model = None
        self._client = None
        self.api_meta: dict = {}  # OpenAI: requested/returned model strings, UTC request window

    def _loaded_code_revision(self) -> str | None:
        """Commit hash of the remote-code repo actually loaded (dynamic-module cache dir name)."""
        if not self.cfg.code_repo:
            return None
        from transformers.utils import HF_MODULES_CACHE
        from pathlib import Path
        base = Path(HF_MODULES_CACHE) / "transformers_modules" / self.cfg.code_repo.replace("-", "_hyphen_").replace(".", "_dot_")
        if not base.is_dir():
            base = Path(HF_MODULES_CACHE) / "transformers_modules" / self.cfg.code_repo
        hashes = sorted(p.name for p in base.iterdir() if p.is_dir() and len(p.name) == 40) if base.is_dir() else []
        return hashes[0] if len(hashes) == 1 else (",".join(hashes) if hashes else "unknown")

    def info(self) -> dict:
        """Backend/library versions for provenance (never includes credentials)."""
        from importlib.metadata import version as pv
        d = {"backend": self.cfg.backend, "model_name": self.cfg.model_name,
             "query_prefix": self.cfg.query_prefix, "doc_prefix": self.cfg.doc_prefix,
             "normalize": self.cfg.normalize, "batch_size": self.cfg.batch_size}
        if self.cfg.revision or self.cfg.query_task:
            d.update(model_revision=self.cfg.revision, trust_remote_code=self.cfg.trust_remote_code,
                     query_task=self.cfg.query_task, doc_task=self.cfg.doc_task)
        if self.cfg.query_prompt_name:
            d["query_prompt_name"] = self.cfg.query_prompt_name
            prompts = getattr(self._model, "prompts", None) or {}
            d["query_prompt_text"] = prompts.get(self.cfg.query_prompt_name)
        d["python"] = platform.python_version()
        if self.cfg.backend == "sentence-transformers":
            import torch
            d.update(sentence_transformers=pv("sentence-transformers"), torch=torch.__version__,
                     transformers=pv("transformers"), device="cpu",
                     max_seq_length_config=self.cfg.max_seq_length,
                     max_seq_length_effective=getattr(self._model, "max_seq_length", None))
            if self.cfg.code_repo:
                d.update(code_repo=self.cfg.code_repo, code_revision_pinned=self.cfg.code_revision,
                         code_revision_loaded=self._loaded_code_revision())
            try:
                from huggingface_hub import HfApi
                d["model_revision_hub_head"] = HfApi().model_info(self.cfg.model_name).sha
            except Exception as ex:  # noqa: BLE001
                d["model_revision_hub_head"] = f"unavailable: {type(ex).__name__}"
        else:
            d["openai"] = pv("openai")
        return d

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Return float32 embeddings (n, dim); L2-normalised if configured."""
        prefix = self.cfg.query_prefix if is_query else self.cfg.doc_prefix
        texts = [prefix + t for t in texts]
        if self.cfg.backend == "sentence-transformers":
            emb = self._encode_st(texts, is_query)
        elif self.cfg.backend == "openai":
            emb = self._encode_openai(texts)
        else:
            raise ValueError(f"unknown backend {self.cfg.backend!r}")
        emb = emb.astype(np.float32)
        return _l2(emb) if self.cfg.normalize else emb

    def _encode_st(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.cfg.model_name, device="cpu", revision=self.cfg.revision,
                trust_remote_code=self.cfg.trust_remote_code)
            if self.cfg.max_seq_length:
                self._model.max_seq_length = self.cfg.max_seq_length
            loaded = self._loaded_code_revision()
            if self.cfg.code_revision and loaded != self.cfg.code_revision:
                raise RuntimeError(f"remote code {self.cfg.code_repo} loaded at {loaded}, "
                                   f"expected pinned {self.cfg.code_revision}")
        task = self.cfg.query_task if is_query else self.cfg.doc_task
        kw = {"task": task} if task else {}
        if is_query and self.cfg.query_prompt_name:
            kw["prompt_name"] = self.cfg.query_prompt_name
        # normalize_embeddings=True matches the pilot (repro) encoding exactly.
        return self._model.encode(texts, batch_size=self.cfg.batch_size, show_progress_bar=False,
                                  normalize_embeddings=self.cfg.normalize, convert_to_numpy=True, **kw)

    def _encode_openai(self, texts: list[str], max_retries: int = 6) -> np.ndarray:
        key = os.environ.get(OPENAI_KEY_ENV)
        if not key:
            raise RuntimeError(f"environment variable {OPENAI_KEY_ENV} is not set")
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=key, max_retries=0)
        out = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i + self.cfg.batch_size]]
            for attempt in range(max_retries):
                try:
                    now = datetime.now(timezone.utc).isoformat()
                    resp = self._client.embeddings.create(model=self.cfg.model_name, input=batch)
                    m = self.api_meta
                    m.setdefault("requested_model", self.cfg.model_name)
                    m.setdefault("first_request_utc", now)
                    m["last_request_utc"] = datetime.now(timezone.utc).isoformat()
                    m["n_requests"] = m.get("n_requests", 0) + 1
                    m.setdefault("returned_models", [])
                    if getattr(resp, "model", None) not in m["returned_models"]:
                        m["returned_models"].append(getattr(resp, "model", None))
                    out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
                    break
                except Exception as ex:  # noqa: BLE001
                    status = getattr(ex, "status_code", None)
                    retryable = status in (None, 408, 409, 429) or (isinstance(status, int) and status >= 500)
                    if not retryable or attempt == max_retries - 1:
                        # Only exception type/status are surfaced, never headers or key.
                        raise RuntimeError(f"OpenAI embedding failed: {type(ex).__name__} status={status}") from None
                    time.sleep(min(2 ** attempt, 30))
        return np.asarray(out, dtype=np.float32)
