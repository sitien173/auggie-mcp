"""Unit tests for auggie_mcp.embed."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from auggie_mcp.config import Config, EmbeddingConfig, VoyageConfig
from auggie_mcp.embed import (
    EmbedError,
    MissingApiKeyError,
    VoyageEmbedder,
    make_model_key,
)
from tests.fake_embedder import FakeEmbedder


class _VoyageStubError(Exception):
    def __init__(self, message: str = "", *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class RateLimitError(_VoyageStubError):
    pass


class APIError(_VoyageStubError):
    pass


class Timeout(_VoyageStubError):
    pass


class InvalidRequestError(_VoyageStubError):
    pass


def _make_config(*, batch_size: int = 2, retries: int = 3, dimensions: int = 4) -> Config:
    return Config(
        embedding=EmbeddingConfig(
            model="voyage-code-3",
            dimensions=dimensions,
            batch_size=batch_size,
        ),
        voyage=VoyageConfig(timeout_s=10.0, retries=retries),
    )


def _embed_response(texts: list[str], dims: int = 4) -> SimpleNamespace:
    # Return deterministic vectors keyed by text length/chars so order is verifiable.
    embeddings = []
    for text in texts:
        val = float(sum(ord(c) for c in text))
        embeddings.append([val + i for i in range(dims)])
    return SimpleNamespace(embeddings=embeddings)


class StubClient:
    """Stub Voyage client that executes scripted outcomes per call."""

    def __init__(self, outcomes: list[object] | None = None, dims: int = 4) -> None:
        self._outcomes = list(outcomes or [])
        self._dims = dims
        self.calls: list[dict[str, object]] = []

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        input_type: str | None = None,
        truncation: bool | None = None,
        output_dtype: str | None = None,
        output_dimension: int | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "truncation": truncation,
                "output_dtype": output_dtype,
                "output_dimension": output_dimension,
            }
        )
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return outcome(texts)
            return outcome
        return _embed_response(texts, dims=output_dimension or self._dims)


def test_model_key_format() -> None:
    cfg = _make_config(dimensions=1024)
    embedder = VoyageEmbedder(cfg, client=StubClient())
    assert embedder.model_key == "voyage-code-3:1024:float32"
    assert make_model_key("voyage-3", 512) == "voyage-3:512:float32"


def test_missing_api_key_raises_before_any_call() -> None:
    cfg = _make_config()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(VoyageEmbedder, "_create_client") as mock_create,
        pytest.raises(MissingApiKeyError),
    ):
        embedder = VoyageEmbedder(cfg)
        embedder.embed(["hello"])
        mock_create.assert_not_called()


def test_rate_limit_429_retried_then_success() -> None:
    cfg = _make_config(retries=2)
    sleeps: list[float] = []
    client = StubClient(
        outcomes=[
            RateLimitError("rate limited", http_status=429),
            RateLimitError("still limited", http_status=429),
        ]
    )
    embedder = VoyageEmbedder(cfg, client=client, sleep=sleeps.append)
    result = embedder.embed(["a", "b"])
    assert len(client.calls) == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]
    assert len(result) == 2
    assert result[0].dtype == np.float32


def test_server_error_500_retried_then_success() -> None:
    cfg = _make_config(retries=2)
    sleeps: list[float] = []
    client = StubClient(
        outcomes=[
            APIError("internal error", http_status=500),
        ]
    )
    embedder = VoyageEmbedder(cfg, client=client, sleep=sleeps.append)
    result = embedder.embed(["a"])
    assert len(client.calls) == 2
    assert len(sleeps) == 1
    assert len(result) == 1


def test_timeout_retried_then_success() -> None:
    cfg = _make_config(retries=2)
    sleeps: list[float] = []
    client = StubClient(
        outcomes=[
            Timeout("request timed out"),
        ]
    )
    embedder = VoyageEmbedder(cfg, client=client, sleep=sleeps.append)
    result = embedder.embed(["a"])
    assert len(client.calls) == 2
    assert len(sleeps) == 1
    assert len(result) == 1


def test_400_not_retried_and_raised() -> None:
    cfg = _make_config(retries=3)
    sleeps: list[float] = []
    client = StubClient(
        outcomes=[
            InvalidRequestError("bad request", http_status=400),
        ]
    )
    embedder = VoyageEmbedder(cfg, client=client, sleep=sleeps.append)
    with pytest.raises(EmbedError) as exc_info:
        embedder.embed(["a"])
    assert len(client.calls) == 1
    assert sleeps == []
    assert exc_info.value.status == 400


def test_retries_exhausted_raises_error_carrying_status() -> None:
    cfg = _make_config(retries=2)
    sleeps: list[float] = []
    client = StubClient(
        outcomes=[
            APIError("bad gateway", http_status=502),
            APIError("bad gateway", http_status=502),
            APIError("bad gateway", http_status=502),
        ]
    )
    embedder = VoyageEmbedder(cfg, client=client, sleep=sleeps.append)
    with pytest.raises(EmbedError) as exc_info:
        embedder.embed(["a"])
    assert len(client.calls) == 3
    assert len(sleeps) == 2
    assert exc_info.value.status == 502


def test_order_preserved_across_parallel_batches() -> None:
    cfg = _make_config(batch_size=2, dimensions=4)

    class DelayedStubClient(StubClient):
        def embed(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
            # Even if batch 0 is slower than batch 1, output order must match input.
            if texts[0] == "first":
                import time

                time.sleep(0.03)
            return super().embed(texts, **kwargs)

    client = DelayedStubClient()
    embedder = VoyageEmbedder(cfg, client=client, max_workers=4)
    inputs = ["first", "second", "third", "fourth", "fifth"]
    vectors = embedder.embed(inputs)
    assert len(vectors) == 5
    for idx, text in enumerate(inputs):
        expected_first_coord = float(sum(ord(c) for c in text))
        assert vectors[idx][0] == pytest.approx(expected_first_coord)
        assert vectors[idx].shape == (4,)
        assert vectors[idx].dtype == np.float32


def test_fake_embedder_deterministic_and_counts() -> None:
    fake = FakeEmbedder(model="voyage-code-3", dimensions=8)
    assert fake.model_key == "voyage-code-3:8:float32"
    v1 = fake.embed(["alpha", "beta"])
    v2 = fake.embed(["alpha"])
    assert fake.call_count == 2
    assert fake.text_count == 3
    assert np.array_equal(v1[0], v2[0])
    assert v1[0].dtype == np.float32
    assert v1[0].shape == (8,)


def test_batches_split_by_character_budget() -> None:
    import auggie_mcp.embed as embed_mod

    cfg = _make_config(batch_size=128, dimensions=4)
    client = StubClient()
    embedder = VoyageEmbedder(cfg, client=client, max_workers=1)
    big = "x" * (embed_mod.MAX_BATCH_CHARS // 2 + 1)
    vectors = embedder.embed([big, big, big])
    assert len(vectors) == 3
    assert len(client.calls) == 3
    for call in client.calls:
        assert sum(len(t) for t in call["texts"]) <= embed_mod.MAX_BATCH_CHARS


def test_voyage_embed_explicitly_enables_truncation() -> None:
    client = StubClient()
    embedder = VoyageEmbedder(_make_config(), client=client)

    embedder.embed(["one long input"])

    assert client.calls[0]["truncation"] is True
