"""Voyage AI embedding client with status-based retry."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np

from auggie_mcp.config import Config

# Voyage rejects a request above 120,000 tokens. A token spans at least one
# character for mostly ASCII code, so a character budget below it bounds a request.
MAX_BATCH_CHARS = 100_000


def _make_batches(items: list[str], max_count: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in items:
        if current and (len(current) >= max_count or chars + len(text) > MAX_BATCH_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(text)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "RateLimitError",
        "ServerError",
        "ServiceUnavailableError",
        "TryAgain",
        "Timeout",
        "APIConnectionError",
    }
)


class EmbedError(RuntimeError):
    """Raised when embedding fails."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.http_status = status


class MissingApiKeyError(EmbedError):
    """Raised when VOYAGE_API_KEY is not configured."""


class Embedder(Protocol):
    """Protocol for text embedders."""

    @property
    def model_key(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]: ...


def make_model_key(model: str, dimensions: int) -> str:
    """Return the canonical model_key string."""
    return f"{model}:{dimensions}:float32"


def _extract_status(exc: BaseException) -> int | None:
    status = getattr(exc, "http_status", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _is_retryable(exc: BaseException, status: int | None) -> bool:
    if status is not None:
        if status == 429 or 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _RETRYABLE_ERROR_NAMES


class VoyageEmbedder:
    """Voyage AI embedder with parallel batching and status-based retries."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] | None = None,
        max_workers: int = 4,
        base_delay_s: float = 0.5,
        max_delay_s: float = 30.0,
        input_type: str | None = "document",
    ) -> None:
        self._config = config or Config()
        self._api_key = api_key
        self._client = client
        self._sleep = sleep or time.sleep
        self._max_workers = max(1, max_workers)
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._input_type = input_type

    @property
    def model_key(self) -> str:
        emb = self._config.embedding
        return make_model_key(emb.model, emb.dimensions)

    def _create_client(self, api_key: str) -> Any:
        import voyageai

        return voyageai.Client(
            api_key=api_key,
            max_retries=0,
            timeout=self._config.voyage.timeout_s,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self._api_key if self._api_key is not None else os.environ.get("VOYAGE_API_KEY")
        if not key or not key.strip():
            raise MissingApiKeyError(
                "VOYAGE_API_KEY environment variable is not set"
            )
        self._client = self._create_client(key)
        return self._client

    def _embed_batch(
        self, client: Any, batch: list[str], *, input_type: str | None = None
    ) -> list[np.ndarray]:
        emb = self._config.embedding
        max_retries = max(0, self._config.voyage.retries)
        for attempt in range(max_retries + 1):
            try:
                # Enable Voyage per-input context truncation for oversized texts.
                response = client.embed(
                    texts=batch,
                    model=emb.model,
                    input_type=self._input_type if input_type is None else input_type,
                    truncation=True,
                    output_dtype="float",
                    output_dimension=emb.dimensions,
                )
                return [
                    np.asarray(vec, dtype=np.float32) for vec in response.embeddings
                ]
            except Exception as exc:
                status = _extract_status(exc)
                if not _is_retryable(exc, status) or attempt >= max_retries:
                    raise EmbedError(str(exc), status=status) from exc
                delay = min(self._max_delay_s, self._base_delay_s * (2**attempt))
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return self._embed(texts, input_type=self._input_type)

    def embed_query(self, texts: Sequence[str]) -> list[np.ndarray]:
        return self._embed(texts, input_type="query")

    def _embed(self, texts: Sequence[str], *, input_type: str | None) -> list[np.ndarray]:
        items = list(texts)
        if not items:
            return []
        client = self._get_client()
        batches = _make_batches(items, max(1, self._config.embedding.batch_size))
        if len(batches) == 1:
            return self._embed_batch(client, batches[0], input_type=input_type)

        workers = min(self._max_workers, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(
                pool.map(lambda b: self._embed_batch(client, b, input_type=input_type), batches)
            )
        return [vec for group in batch_results for vec in group]
