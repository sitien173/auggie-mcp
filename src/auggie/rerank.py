"""
Voyage reranking for improved search result ordering.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

VOYAGE_RERANK_ALIASES = {
    "voyage": "rerank-2.5",
    "voyage:lite": "rerank-2.5-lite",
    "voyage:2": "rerank-2",
}

DEFAULT_RERANK_MODEL = "voyage"
DEFAULT_RERANK_TOPN = int(os.environ.get("AUGGIE_RERANK_TOPN", "50"))

_voyage_rerank_client: Any = None
_captured_warnings: list[str] = []


def _resolve_model_name(model_name: str) -> str:
    """Resolve a rerank alias to its Voyage model name."""
    return VOYAGE_RERANK_ALIASES.get(model_name, model_name)


def is_voyage_rerank_available() -> bool:
    """Check if Voyage reranking is available."""
    try:
        import voyageai  # noqa: F401

        return os.environ.get("VOYAGE_API_KEY") is not None
    except ImportError:
        return False


def get_available_backends() -> dict[str, Any]:
    """Return Voyage rerank availability info."""
    return {
        "voyage_ai": {
            "available": is_voyage_rerank_available(),
            "install": "Install the dependencies from requirements.txt and set VOYAGE_API_KEY.",
            "models": [
                {
                    "alias": "voyage",
                    "name": "rerank-2.5",
                    "context": "32K tokens",
                    "note": "Recommended default.",
                },
                {
                    "alias": "voyage:lite",
                    "name": "rerank-2.5-lite",
                    "context": "32K tokens",
                    "note": "Lower cost.",
                },
                {
                    "alias": "voyage:2",
                    "name": "rerank-2",
                    "context": "16K tokens",
                    "note": "Previous generation.",
                },
            ],
        },
        "default_model": DEFAULT_RERANK_MODEL,
    }


def is_reranker_available(model_name: str | None = None) -> bool:
    """Check if Voyage reranking is available."""
    del model_name
    return is_voyage_rerank_available()


def _clear_reranker_cache() -> None:
    """Clear the cached Voyage reranker."""
    global _voyage_rerank_client, _captured_warnings
    _voyage_rerank_client = None
    _captured_warnings = []


def _clear_all_state() -> None:
    """Clear all cached state."""
    _clear_reranker_cache()


def get_captured_warnings() -> list[str]:
    """Get captured warnings."""
    return _captured_warnings.copy()


def clear_captured_warnings() -> None:
    """Clear captured warnings."""
    global _captured_warnings
    _captured_warnings = []


def _get_voyage_reranker() -> Any:
    """Get or create the cached Voyage client."""
    global _voyage_rerank_client

    if _voyage_rerank_client is not None:
        return _voyage_rerank_client

    try:
        import voyageai
    except ImportError as e:
        raise ImportError(
            "Voyage reranking requires the voyageai package. "
            "Install the dependencies from requirements.txt."
        ) from e

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable is not set.")

    timeout_str = os.environ.get("AUGGIE_VOYAGE_TIMEOUT")
    timeout = float(timeout_str) if timeout_str else 120.0
    retries_str = os.environ.get("AUGGIE_VOYAGE_RETRIES")
    max_retries = int(retries_str) if retries_str else 2

    _voyage_rerank_client = voyageai.Client(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    return _voyage_rerank_client


def _voyage_rerank_predict(query: str, texts: list[str], model: str) -> list[float]:
    """Get reranking scores from Voyage."""
    client = _get_voyage_reranker()
    actual_model = _resolve_model_name(model)

    start_time = time.time()
    result = client.rerank(
        query=query,
        documents=texts,
        model=actual_model,
    )
    elapsed = time.time() - start_time

    if elapsed > 10:
        print(
            f"Warning: Voyage rerank API took {elapsed:.1f}s for {len(texts)} documents",
            file=sys.stderr,
        )

    scores = [0.0] * len(texts)
    for r in result.results:
        scores[r.index] = float(r.relevance_score)
    return scores


def _compute_confidence(score: float, top_score: float) -> str:
    """Compute confidence level for a reranked result."""
    if top_score <= 0:
        return "very_low"

    ratio = score / top_score
    if ratio >= 0.90:
        return "high"
    if ratio >= 0.75:
        return "medium"
    if ratio >= 0.50:
        return "low"
    return "very_low"


@dataclass
class RerankedHit:
    """A search result after reranking."""

    score: float
    path: str
    start_line: int
    end_line: int
    text: str
    chunk_id: int
    chunk_index: int
    confidence: str
    confidence_details: Any = None


def rerank_results(
    query: str,
    hits: list[Any],
    top_n: int | None = None,
    model_name: str | None = None,
    cache_path: Any | None = None,
) -> list[Any]:
    """Rerank search results with Voyage."""
    if not hits:
        return []

    n = top_n if top_n is not None else DEFAULT_RERANK_TOPN
    candidates = hits[:n]
    if not candidates:
        return []

    model = model_name or os.environ.get("AUGGIE_RERANK_MODEL", DEFAULT_RERANK_MODEL)

    cache_con = None
    cache_enabled = cache_path is not None and not os.environ.get("AUGGIE_CACHE_DISABLED")
    chunk_hashes: list[str] = []

    if cache_enabled:
        import hashlib

        from .cache import connect_cache, get_rerank_results, log_cache_event

        cache_con = connect_cache(cache_path)
        chunk_hashes = [hashlib.sha256(hit.text.encode()).hexdigest()[:16] for hit in candidates]
        l3_result = get_rerank_results(cache_con, query, chunk_hashes, model, n)
        if l3_result.hit:
            cached_scores = l3_result.data
            candidate_lookup = {hit.chunk_id: hit for hit in candidates}
            top_score = max(s for _, s in cached_scores) if cached_scores else 0.0

            reranked_hits = []
            for chunk_id, score in cached_scores:
                if chunk_id in candidate_lookup:
                    hit = candidate_lookup[chunk_id]
                    reranked_hits.append(
                        RerankedHit(
                            score=float(score),
                            path=hit.path,
                            start_line=hit.start_line,
                            end_line=hit.end_line,
                            text=hit.text,
                            chunk_id=hit.chunk_id,
                            chunk_index=hit.chunk_index,
                            confidence=_compute_confidence(score, top_score),
                            confidence_details=getattr(hit, "confidence_details", None),
                        )
                    )

            log_cache_event(cache_con, "L3", "hit", time_saved_ms=l3_result.time_saved_ms)
            return reranked_hits

    if not is_voyage_rerank_available():
        raise ImportError("Voyage reranking is unavailable. Set VOYAGE_API_KEY and install dependencies.")

    texts = [hit.text for hit in candidates]
    scores = _voyage_rerank_predict(query, texts, model)

    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    top_score = max(scores) if scores else 0.0
    scored_hits = []
    for hit, score in zip(candidates, scores, strict=False):
        scored_hits.append(
            (
                score,
                RerankedHit(
                    score=float(score),
                    path=hit.path,
                    start_line=hit.start_line,
                    end_line=hit.end_line,
                    text=hit.text,
                    chunk_id=hit.chunk_id,
                    chunk_index=hit.chunk_index,
                    confidence=_compute_confidence(score, top_score),
                    confidence_details=getattr(hit, "confidence_details", None),
                ),
            )
        )

    scored_hits.sort(key=lambda x: x[0], reverse=True)

    if cache_enabled and cache_con is not None:
        from .cache import log_cache_event, set_rerank_results

        results_to_cache = [(hit.chunk_id, score) for score, hit in scored_hits]
        set_rerank_results(cache_con, query, chunk_hashes, model, n, results_to_cache)
        log_cache_event(cache_con, "L3", "miss")

    return [hit for _, hit in scored_hits]
