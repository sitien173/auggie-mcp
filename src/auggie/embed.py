"""
Embedding module for auggie.

Provides Voyage-only text embedding support with token-aware batching.

Environment variables:
    AUGGIE_MODEL: Override default embedding model.
    AUGGIE_DIMENSIONS: Override default embedding dimensions.
    AUGGIE_BATCH_SIZE: Override max texts per Voyage request.
    VOYAGE_API_KEY: Required for Voyage API access.
    AUGGIE_VOYAGE_TIMEOUT: Timeout in seconds.
    AUGGIE_VOYAGE_RETRIES: Max retries.
    AUGGIE_VOYAGE_CHARS_PER_TOKEN: Token estimation ratio.
"""

from __future__ import annotations

import array
import math
import os
import sys
import time
from typing import Literal, overload

from .models import get_max_batch_size, resolve_dimensions, resolve_model

ENV_BATCH_SIZE = "AUGGIE_BATCH_SIZE"
ENV_VOYAGE_API_KEY = "VOYAGE_API_KEY"
ENV_VOYAGE_TIMEOUT = "AUGGIE_VOYAGE_TIMEOUT"
ENV_VOYAGE_RETRIES = "AUGGIE_VOYAGE_RETRIES"
ENV_VOYAGE_CHARS_PER_TOKEN = "AUGGIE_VOYAGE_CHARS_PER_TOKEN"

DEFAULT_VOYAGE_TIMEOUT = 120.0
DEFAULT_VOYAGE_RETRIES = 2
DEFAULT_VOYAGE_CHARS_PER_TOKEN = 1.0
VOYAGE_BATCH_TOKEN_LIMIT = 120000


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector for cosine similarity."""
    s = 0.0
    for x in vec:
        s += x * x
    n = math.sqrt(s) if s > 0 else 1.0
    return [x / n for x in vec]


def _get_voyage_chars_per_token() -> float:
    """Get the chars-per-token ratio for Voyage token estimation."""
    env_val = os.environ.get(ENV_VOYAGE_CHARS_PER_TOKEN)
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return DEFAULT_VOYAGE_CHARS_PER_TOKEN


def _estimate_voyage_tokens(text: str) -> int:
    """Estimate Voyage tokens for a text."""
    if not text:
        return 0
    chars_per_token = _get_voyage_chars_per_token()
    return max(1, int(len(text) / chars_per_token))


def _create_voyage_batches(
    texts: list[str],
    max_tokens: int,
    max_count: int | None = None,
) -> list[list[str]]:
    """Create Voyage batches that respect token and count limits."""
    if not texts:
        return []

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        text_tokens = _estimate_voyage_tokens(text)

        if text_tokens > max_tokens:
            chars_per_token = _get_voyage_chars_per_token()
            max_chars = int(max_tokens * chars_per_token * 0.9)
            text = text[:max_chars] + "\n[...truncated...]"
            text_tokens = _estimate_voyage_tokens(text)

        would_exceed_tokens = (current_tokens + text_tokens) > max_tokens
        would_exceed_count = max_count is not None and len(current_batch) >= max_count

        if current_batch and (would_exceed_tokens or would_exceed_count):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += text_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _embed_voyage(
    texts: list[str],
    model: str,
    input_type: str,
    output_dimension: int | None,
) -> tuple[list[bytes], int]:
    """Embed texts using Voyage AI."""
    try:
        import voyageai
    except ImportError as e:
        raise ImportError(
            "Voyage embeddings require the voyageai package. "
            "Install the dependencies from requirements.txt."
        ) from e

    api_key = os.environ.get(ENV_VOYAGE_API_KEY)
    if not api_key:
        raise ValueError(
            f"{ENV_VOYAGE_API_KEY} environment variable is not set. "
            "Get an API key from https://www.voyageai.com/."
        )

    timeout_str = os.environ.get(ENV_VOYAGE_TIMEOUT)
    timeout = float(timeout_str) if timeout_str else DEFAULT_VOYAGE_TIMEOUT
    retries_str = os.environ.get(ENV_VOYAGE_RETRIES)
    max_retries = int(retries_str) if retries_str else DEFAULT_VOYAGE_RETRIES

    client = voyageai.Client(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )

    kwargs = {
        "texts": texts,
        "model": model,
        "input_type": input_type,
    }
    if output_dimension is not None:
        kwargs["output_dimension"] = output_dimension

    start_time = time.time()
    try:
        result = client.embed(**kwargs)
    except Exception as e:
        error_msg = str(e)
        if "tokens" in error_msg.lower() and "batch" in error_msg.lower():
            raise RuntimeError(
                f"Voyage batch token limit exceeded for {len(texts)} texts. "
                f"Original error: {error_msg}"
            ) from None
        if "rate" in error_msg.lower() or "429" in error_msg:
            raise RuntimeError(
                f"Voyage rate limit reached. Wait and retry.\nOriginal error: {error_msg}"
            ) from None
        if "auth" in error_msg.lower() or "api key" in error_msg.lower() or "401" in error_msg:
            raise RuntimeError(
                f"Voyage authentication failed. Check {ENV_VOYAGE_API_KEY}.\n"
                f"Original error: {error_msg}"
            ) from None
        if any(term in error_msg.lower() for term in ["connect", "timeout", "network", "unreachable"]):
            raise RuntimeError(
                f"Could not connect to Voyage API.\nOriginal error: {error_msg}"
            ) from None
        raise RuntimeError(f"Voyage API error: {error_msg}") from None

    elapsed = time.time() - start_time
    if elapsed > 30:
        print(
            f"Warning: Voyage API request took {elapsed:.1f}s for {len(texts)} texts",
            file=sys.stderr,
        )

    blobs = []
    for emb in result.embeddings:
        normalized = _l2_normalize(emb)
        blobs.append(array.array("f", normalized).tobytes())

    dim = len(result.embeddings[0]) if result.embeddings else 0
    return blobs, dim


def _embed_voyage_batched(
    texts: list[str],
    model: str,
    dimensions: int | None,
    input_type: str,
) -> tuple[list[bytes], int]:
    """Embed texts with Voyage batching."""
    if not texts:
        return [], 0

    max_tokens = int(VOYAGE_BATCH_TOKEN_LIMIT * 0.8)
    max_batch_count = get_max_batch_size(model)

    env_batch = os.environ.get(ENV_BATCH_SIZE)
    if env_batch:
        max_batch_count = min(int(env_batch), max_batch_count)

    batches = _create_voyage_batches(texts, max_tokens=max_tokens, max_count=max_batch_count)

    all_blobs: list[bytes] = []
    dim = 0

    for batch in batches:
        blobs, batch_dim = _embed_voyage(batch, model, input_type, dimensions)
        all_blobs.extend(blobs)
        if dim == 0:
            dim = batch_dim

    return all_blobs, dim


@overload
def embed_texts(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
    *,
    return_timing: Literal[False] = False,
) -> tuple[list[bytes], int]: ...


@overload
def embed_texts(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
    *,
    return_timing: Literal[True],
) -> tuple[list[bytes], int, float]: ...


def embed_texts(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
    *,
    return_timing: bool = False,
) -> tuple[list[bytes], int] | tuple[list[bytes], int, float]:
    """
    Generate Voyage embeddings for a list of texts.

    Args:
        texts: Text strings to embed.
        model: Voyage embedding model or alias.
        dimensions: Optional dimension override.
        return_timing: Include elapsed seconds when True.
    """
    if not texts:
        if return_timing:
            return [], 0, 0.0
        return [], 0

    start_time = time.perf_counter()
    resolved_model = resolve_model(model)
    resolved_dimensions = resolve_dimensions(dimensions, resolved_model)
    blobs, dim = _embed_voyage_batched(
        texts,
        resolved_model,
        resolved_dimensions,
        input_type="document",
    )

    if return_timing:
        return blobs, dim, time.perf_counter() - start_time
    return blobs, dim
