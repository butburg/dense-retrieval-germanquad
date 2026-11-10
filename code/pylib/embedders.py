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
    """Static configuration of one embedding model.

    Attributes:
        key: Registry key used on the command line and in result paths, e.g. ``"bge-m3"``.
        backend: ``"sentence-transformers"`` (local, CPU) or ``"openai"`` (API).
        model_name: Hugging Face repo ID or OpenAI model name.
        display_name: Name shown in tables and figures, e.g. ``"multilingual-e5-large"``; the one source
            for these labels (``make_results.py`` and notebook 05 read it). Not part of the cache key.
        query_prefix: Text prepended to every query, e.g. ``"query: "``.
        doc_prefix: Text prepended to every document, e.g. ``"passage: "``.
        normalize: Whether embeddings are L2-normalised (dot product = cosine similarity).
        batch_size: Number of texts per encoding batch or API request.
        max_seq_length: Override of the maximum input length in tokens; ``None`` keeps the
            model default.
        revision: Pinned Hugging Face commit hash of the model repo; ``None`` = hub HEAD.
        trust_remote_code: Allow the model repo to execute its own code on load.
        code_repo: Hugging Face repo providing the remote code (``auto_map``), if any.
        code_revision: Expected commit of ``code_repo``, checked after loading.
        query_task: Adapter name passed as ``encode(task=...)`` for queries (jina-v3).
        doc_task: Adapter name passed as ``encode(task=...)`` for documents (jina-v3).
        query_prompt_name: Prompt name passed as ``encode(prompt_name=...)`` for queries (Qwen3).
        extra: Free-form additional settings (currently unused).
    """
    key: str
    backend: str  # "sentence-transformers" | "openai"
    model_name: str
    display_name: str = ""  # label in tables and figures (single source); empty only for unlabelled candidates
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
# Dict comprehension over the model configs below: builds the lookup table key -> config
# (e.g. "bge-m3" -> its EmbedderConfig) used by get_config and the CLI.
REGISTRY: dict[str, EmbedderConfig] = {c.key: c for c in [
    EmbedderConfig("e5-large", "sentence-transformers", "intfloat/multilingual-e5-large", display_name="multilingual-e5-large",
                   query_prefix="query: ", doc_prefix="passage: ", batch_size=32,
                   revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"),
    # BGE-M3, dense embedding only (no sparse/ColBERT). Model card: https://huggingface.co/BAAI/bge-m3
    # -> no instruction/prefix for queries or passages; max_length 8192 (model default kept).
    EmbedderConfig("bge-m3", "sentence-transformers", "BAAI/bge-m3", display_name="bge-m3", batch_size=16,
                   revision="5617a9f61b028005a4858fdac845db406aefb181"),
    # Model card: https://huggingface.co/Alibaba-NLP/gte-multilingual-base
    # -> no prefixes; usage via sentence-transformers with trust_remote_code=True; max 8192 tokens
    # (model default kept). Remote code repo Alibaba-NLP/new-impl is pinned via code_revision (checked after load).
    EmbedderConfig("gte-multilingual-base", "sentence-transformers", "Alibaba-NLP/gte-multilingual-base", display_name="gte-multilingual-base",
                   batch_size=16, trust_remote_code=True,
                   revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
                   code_repo="Alibaba-NLP/new-impl", code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4"),
    # Model card: https://huggingface.co/jinaai/jina-embeddings-v3
    # -> LoRA task adapters via encode(task=...): "retrieval.query" for queries, "retrieval.passage" for
    # documents (asymmetric retrieval); no text prefix (adapter adds its instruction internally);
    # max 8192 tokens (model default kept). Remote code repo jinaai/xlm-roberta-flash-implementation pinned via code_revision.
    EmbedderConfig("jina-v3", "sentence-transformers", "jinaai/jina-embeddings-v3", display_name="jina-embeddings-v3", batch_size=4,
                   trust_remote_code=True, query_task="retrieval.query", doc_task="retrieval.passage",
                   revision="ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
                   code_repo="jinaai/xlm-roberta-flash-implementation",
                   code_revision="bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"),
    # --- Weitere Modelle der neun-Modelle-Liste (compare_models.MODELS) ---
    # Model card + config_sentence_transformers.json: prompts {"query": "Instruct: Given a web search query,
    # retrieve relevant passages that answer the query\nQuery:", "document": ""} -> prompt_name="query" for queries only.
    # Native context 32k; max_seq_length capped at 8192 (longest GermanQuAD passage ~2.7k tokens -> no truncation).
    EmbedderConfig("qwen3-embedding-0.6b", "sentence-transformers", "Qwen/Qwen3-Embedding-0.6B", display_name="Qwen3-Embedding-0.6B", batch_size=8,
                   query_prompt_name="query", max_seq_length=8192,
                   revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    # config_sentence_transformers.json: prompts {"query": "query: "}; documents without prefix. Native 8192.
    EmbedderConfig("arctic-embed-l-v2", "sentence-transformers", "Snowflake/snowflake-arctic-embed-l-v2.0", display_name="snowflake-arctic-embed-l-v2.0",
                   query_prefix="query: ", batch_size=16,
                   revision="ac6544c8a46e00af67e330e85a9028c66b8cfd9a"),
    # Model card: no prefix; native 8192 (ALiBi); remote code jinaai/jina-bert-implementation pinned via code_revision.
    EmbedderConfig("jina-v2-base-de", "sentence-transformers", "jinaai/jina-embeddings-v2-base-de", display_name="jina-embeddings-v2-base-de", batch_size=8,
                   trust_remote_code=True, revision="3f9eede875721714945b6a99a3198299243cf2be",
                   code_repo="jinaai/jina-bert-implementation",
                   code_revision="f3ec4cf7de7e561007f27c9efc7148b0bd713f81"),
    EmbedderConfig("openai-3-small", "openai", "text-embedding-3-small", display_name="text-embedding-3-small", batch_size=128),
    EmbedderConfig("openai-3-large", "openai", "text-embedding-3-large", display_name="text-embedding-3-large", batch_size=128),
]}


def get_config(key: str) -> EmbedderConfig:
    """Look up the registered configuration of a model.

    Args:
        key: Registry key of the model, e.g. ``"e5-large"`` (one of ``REGISTRY``).

    Returns:
        The frozen :class:`EmbedderConfig` registered under ``key``.

    Raises:
        KeyError: If ``key`` is not registered.
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[key]


def _l2(embeddings: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm; near-zero rows are divided by at least ``1e-12``."""
    # Row-wise L2 norm: axis=1 collapses the embedding dimension, keepdims=True keeps shape (n, 1)
    # so the division below broadcasts per row, e.g. [3, 4] has norm 5 and becomes [0.6, 0.8].
    row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # np.clip lifts norms below 1e-12 to 1e-12 so all-zero rows do not cause a division by zero.
    return embeddings / np.clip(row_norms, 1e-12, None)


def allow_offline_tokenizer_loading() -> None:
    """Work around transformers 4.57.3, which queries the hub API in offline mode and then fails.

    ``PreTrainedTokenizerBase._patch_mistral_regex`` calls ``huggingface_hub.model_info`` to detect Mistral base
    models, even with ``HF_HUB_OFFLINE=1``. The call raises ``OfflineModeIsEnabled``, and sentence-transformers
    reports "Unrecognized processing class". The replacement returns the tokenizer unchanged, which is also what the
    original does for every non-Mistral model (all models of the registry); tokenization and vectors stay the same.
    A transformers version without this method is left alone. Call only in offline mode.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if hasattr(PreTrainedTokenizerBase, "_patch_mistral_regex"):
        PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, tokenizer, *args, **kwargs: tokenizer)


class Embedder:
    """Encodes texts with one configured model; prefixes and normalisation are applied here.

    The model (sentence-transformers) or the API client (OpenAI) is created lazily on the
    first call of :meth:`encode`.

    Attributes:
        cfg: The :class:`EmbedderConfig` this embedder was built from.
        api_meta: OpenAI only: requested and returned model strings, number of requests and
            UTC request window, filled while encoding. Never contains credentials.
    """

    def __init__(self, cfg: EmbedderConfig):
        """Create an embedder; no model is loaded yet.

        Args:
            cfg: Configuration of the model to use, e.g. ``get_config("e5-large")``.
        """
        self.cfg = cfg
        self._model = None
        self._client = None
        self.api_meta: dict = {}  # OpenAI: requested/returned model strings, UTC request window

    def _loaded_code_revision(self) -> str | None:
        """Find the commit hash of the remote-code repo that was actually loaded.

        The hash is read from the dynamic-module cache directory of ``transformers``.

        Returns:
            ``None`` if the config has no ``code_repo``; the 40-character commit hash if
            exactly one is cached; several hashes joined by commas if more than one is
            cached; ``"unknown"`` if none is found.
        """
        if not self.cfg.code_repo:
            return None
        from pathlib import Path
        from transformers.utils import HF_MODULES_CACHE
        modules_root = Path(HF_MODULES_CACHE) / "transformers_modules"
        # transformers escapes "-" and "." in the directory name of the repo.
        repo_dir = modules_root / self.cfg.code_repo.replace("-", "_hyphen_").replace(".", "_dot_")
        if not repo_dir.is_dir():
            repo_dir = modules_root / self.cfg.code_repo
        if repo_dir.is_dir():
            # Generator over the entries of the cached repo directory: keep sub-folder names that look like a
            # 40-character git commit hash; these are the remote-code revisions that were actually downloaded.
            commit_hashes = sorted(entry.name for entry in repo_dir.iterdir()
                                   if entry.is_dir() and len(entry.name) == 40)
        else:
            commit_hashes = []
        if len(commit_hashes) == 1:
            return commit_hashes[0]
        return ",".join(commit_hashes) if commit_hashes else "unknown"

    def info(self) -> dict:
        """Collect backend, model and library versions for provenance.

        Call after encoding so that the effective ``max_seq_length`` and prompt text of a
        loaded sentence-transformers model are included. The Hub HEAD revision is queried
        online and reported as ``"unavailable: <ErrorType>"`` if that fails.

        Returns:
            JSON-serializable dict; never contains credentials.
        """
        from importlib.metadata import version as package_version
        provenance = {"backend": self.cfg.backend, "model_name": self.cfg.model_name,
                      "query_prefix": self.cfg.query_prefix, "doc_prefix": self.cfg.doc_prefix,
                      "normalize": self.cfg.normalize, "batch_size": self.cfg.batch_size}
        if self.cfg.revision or self.cfg.query_task:
            provenance.update(model_revision=self.cfg.revision, trust_remote_code=self.cfg.trust_remote_code,
                              query_task=self.cfg.query_task, doc_task=self.cfg.doc_task)
        if self.cfg.query_prompt_name:
            provenance["query_prompt_name"] = self.cfg.query_prompt_name
            prompts = getattr(self._model, "prompts", None) or {}
            provenance["query_prompt_text"] = prompts.get(self.cfg.query_prompt_name)
        provenance["python"] = platform.python_version()
        if self.cfg.backend == "sentence-transformers":
            import torch
            provenance.update(sentence_transformers=package_version("sentence-transformers"),
                              torch=torch.__version__, transformers=package_version("transformers"),
                              device="cpu", max_seq_length_config=self.cfg.max_seq_length,
                              max_seq_length_effective=getattr(self._model, "max_seq_length", None))
            if self.cfg.code_repo:
                provenance.update(code_repo=self.cfg.code_repo, code_revision_pinned=self.cfg.code_revision,
                                  code_revision_loaded=self._loaded_code_revision())
            try:
                from huggingface_hub import HfApi
                provenance["model_revision_hub_head"] = HfApi().model_info(self.cfg.model_name).sha
            except Exception as error:  # noqa: BLE001
                provenance["model_revision_hub_head"] = f"unavailable: {type(error).__name__}"
        else:
            provenance["openai"] = package_version("openai")
        return provenance

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Embed texts as queries or as documents.

        Args:
            texts: Raw texts, e.g. ``["Wer war Goethe?"]``. The query or document prefix of
                the config is prepended before encoding.
            is_query: True for queries (query prefix, query task or prompt), False for
                documents.

        Returns:
            Array of shape ``(len(texts), dim)`` and dtype float32, L2-normalised if the
            config says so.

        Raises:
            ValueError: If the configured backend is unknown.
            RuntimeError: For OpenAI models if the API key variable is missing or the request
                fails; for models with pinned remote code if another commit was loaded.
        """
        prefix = self.cfg.query_prefix if is_query else self.cfg.doc_prefix
        # For every text: prepend the model's query or passage prefix (e.g. "query: "), as the model card requires.
        prefixed_texts = [prefix + text for text in texts]
        if self.cfg.backend == "sentence-transformers":
            embeddings = self._encode_st(prefixed_texts, is_query)
        elif self.cfg.backend == "openai":
            embeddings = self._encode_openai(prefixed_texts)
        else:
            raise ValueError(f"unknown backend {self.cfg.backend!r}")
        # Cast to float32 so all backends return the same dtype (and memory stays small).
        embeddings = embeddings.astype(np.float32)
        return _l2(embeddings) if self.cfg.normalize else embeddings

    def _encode_st(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed already prefixed texts with sentence-transformers (CPU).

        Loads the model on first use, applies ``max_seq_length`` from the config and checks
        the pinned remote-code revision. Queries additionally use ``query_task`` or
        ``query_prompt_name`` if configured; documents use ``doc_task``.
        """
        if self._model is None:
            from huggingface_hub import constants as hub_constants
            if hub_constants.HF_HUB_OFFLINE:  # set before the first import of huggingface_hub (e.g. by demo_query.py)
                allow_offline_tokenizer_loading()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.cfg.model_name, device="cpu", revision=self.cfg.revision,
                trust_remote_code=self.cfg.trust_remote_code)
            if self.cfg.max_seq_length:
                self._model.max_seq_length = self.cfg.max_seq_length
            loaded_revision = self._loaded_code_revision()
            if self.cfg.code_revision and loaded_revision != self.cfg.code_revision:
                raise RuntimeError(f"remote code {self.cfg.code_repo} loaded at {loaded_revision}, "
                                   f"expected pinned {self.cfg.code_revision}")
        task = self.cfg.query_task if is_query else self.cfg.doc_task
        encode_kwargs = {"task": task} if task else {}
        if is_query and self.cfg.query_prompt_name:
            encode_kwargs["prompt_name"] = self.cfg.query_prompt_name
        # normalize_embeddings=True matches the pilot (repro) encoding exactly.
        return self._model.encode(texts, batch_size=self.cfg.batch_size, show_progress_bar=False,
                                  normalize_embeddings=self.cfg.normalize, convert_to_numpy=True,
                                  **encode_kwargs)

    def _record_api_metadata(self, response, request_started_utc: str) -> None:
        """Update ``api_meta`` after one successful OpenAI request."""
        self.api_meta.setdefault("requested_model", self.cfg.model_name)
        self.api_meta.setdefault("first_request_utc", request_started_utc)
        self.api_meta["last_request_utc"] = datetime.now(timezone.utc).isoformat()
        self.api_meta["n_requests"] = self.api_meta.get("n_requests", 0) + 1
        self.api_meta.setdefault("returned_models", [])
        returned_model = getattr(response, "model", None)
        if returned_model not in self.api_meta["returned_models"]:
            self.api_meta["returned_models"].append(returned_model)

    def _encode_openai(self, texts: list[str], max_retries: int = 6) -> np.ndarray:
        """Embed already prefixed texts via the OpenAI embeddings API in batches.

        Blank texts are replaced by a single space.
        Rate-limit, timeout, conflict and server errors are retried with exponential
        backoff (at most 30 s) up to ``max_retries`` attempts per batch.

        Raises:
            RuntimeError: If ``EMBED_OPENAI_API_KEY`` is not set, or a request fails
                non-retryably or after the last attempt (only exception type and status are
                reported, never headers or key).
        """
        api_key = os.environ.get(OPENAI_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"environment variable {OPENAI_KEY_ENV} is not set")
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, max_retries=0)
        embeddings = []
        # For every batch of texts (batch_start = 0, batch_size, 2*batch_size, ...): one API request,
        # each request carries at most batch_size texts; embeddings accumulate in input order.
        for batch_start in range(0, len(texts), self.cfg.batch_size):
            # Per text in the current batch: replace blank texts by a single space so no empty string is sent
            # to the API.
            batch_texts = [text if text.strip() else " "
                           for text in texts[batch_start:batch_start + self.cfg.batch_size]]
            # Retry loop for this batch: attempt 0, 1, ...; a successful request breaks out, retryable errors wait
            # and try again (exponential backoff), the last failed attempt raises.
            for attempt in range(max_retries):
                try:
                    request_started_utc = datetime.now(timezone.utc).isoformat()
                    response = self._client.embeddings.create(model=self.cfg.model_name, input=batch_texts)
                    self._record_api_metadata(response, request_started_utc)
                    # The API may return items out of order; restore the input order via index.
                    items_in_input_order = sorted(response.data, key=lambda item: item.index)
                    # Generator over the returned items: append each item's vector to the running list, in input
                    # order.
                    embeddings.extend(item.embedding for item in items_in_input_order)
                    break
                except Exception as error:  # noqa: BLE001
                    status = getattr(error, "status_code", None)
                    is_retryable = (status in (None, 408, 409, 429)
                                    or (isinstance(status, int) and status >= 500))
                    if not is_retryable or attempt == max_retries - 1:
                        # Only exception type/status are surfaced, never headers or key.
                        raise RuntimeError(f"OpenAI embedding failed: {type(error).__name__} status={status}") from None
                    time.sleep(min(2 ** attempt, 30))
        # Stack the list of vectors into one (n_texts, dim) float32 matrix.
        return np.asarray(embeddings, dtype=np.float32)
