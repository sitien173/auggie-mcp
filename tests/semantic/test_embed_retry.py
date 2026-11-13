"""Tests for retrying Voyage errors that the voyageai client does not retry."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from auggie import embed


class APIConnectionError(Exception):
    pass


class ServerError(Exception):
    pass


class InvalidRequestError(Exception):
    pass


def _make_fake_voyageai(errors):
    """Build a fake voyageai module whose embed raises each error in turn, then succeeds."""
    state = {"calls": 0, "client_kwargs": None}
    pending = list(errors)

    class FakeClient:
        def __init__(self, **kwargs):
            state["client_kwargs"] = kwargs

        def embed(self, texts, model, input_type, **kwargs):
            state["calls"] += 1
            if pending:
                raise pending.pop(0)
            return types.SimpleNamespace(embeddings=[[1.0, 0.0] for _ in texts])

    module = types.ModuleType("voyageai")
    module.Client = FakeClient
    return module, state


class EmbedRetryTests(unittest.TestCase):
    def _run(self, fake, extra_env=None):
        env = {"VOYAGE_API_KEY": "test-key", **(extra_env or {})}
        with (
            patch.dict(os.environ, env),
            patch.dict(sys.modules, {"voyageai": fake}),
            patch.object(embed, "_sleep") as sleep,
        ):
            if "AUGGIE_VOYAGE_RETRIES" not in env:
                os.environ.pop("AUGGIE_VOYAGE_RETRIES", None)
            result = embed.embed_texts(["a", "b"])
        return result, sleep

    def test_default_attempts_passed_to_client(self):
        fake, state = _make_fake_voyageai([])
        self._run(fake)
        self.assertEqual(state["client_kwargs"]["max_retries"], 5)

    def test_connection_error_is_retried(self):
        fake, state = _make_fake_voyageai([APIConnectionError("connection reset")])
        (blobs, _), sleep = self._run(fake)
        self.assertEqual(state["calls"], 2)
        self.assertEqual(len(blobs), 2)
        self.assertEqual(sleep.call_count, 1)

    def test_server_error_stops_after_max_attempts(self):
        fake, state = _make_fake_voyageai([ServerError("500")] * 10)
        with self.assertRaises(RuntimeError):
            self._run(fake, {"AUGGIE_VOYAGE_RETRIES": "3"})
        self.assertEqual(state["calls"], 3)

    def test_invalid_request_is_not_retried(self):
        fake, state = _make_fake_voyageai([InvalidRequestError("bad input")])
        with self.assertRaises(RuntimeError):
            self._run(fake)
        self.assertEqual(state["calls"], 1)


if __name__ == "__main__":
    unittest.main()
