"""Tests for parallel Voyage embedding with one shared client."""

from __future__ import annotations

import array
import os
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from auggie import embed

CONCURRENCY = getattr(embed, "VOYAGE_CONCURRENCY", 4)


def _make_fake_voyageai(delay_for=None, fail_on=None):
    """Build a fake voyageai module that records clients and in-flight requests."""
    state = {
        "clients": 0,
        "calls": 0,
        "in_flight": 0,
        "peak": 0,
        "lock": threading.Lock(),
    }

    class FakeClient:
        def __init__(self, **kwargs):
            with state["lock"]:
                state["clients"] += 1

        def embed(self, texts, model, input_type, **kwargs):
            with state["lock"]:
                state["calls"] += 1
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            try:
                if delay_for is not None:
                    time.sleep(delay_for(texts))
                if fail_on is not None and fail_on(texts):
                    raise Exception("429 Too Many Requests")
                embeddings = [[1.0, float(int(t[1:]))] for t in texts]
                return types.SimpleNamespace(embeddings=embeddings)
            finally:
                with state["lock"]:
                    state["in_flight"] -= 1

    module = types.ModuleType("voyageai")
    module.Client = FakeClient
    return module, state


def _decode(blob: bytes) -> list[float]:
    values = array.array("f")
    values.frombytes(blob)
    return list(values)


class EmbedParallelTests(unittest.TestCase):
    def _run(self, texts, fake, batch_size):
        env = {"VOYAGE_API_KEY": "test-key", "AUGGIE_BATCH_SIZE": str(batch_size)}
        with patch.dict(os.environ, env), patch.dict(sys.modules, {"voyageai": fake}):
            return embed.embed_texts(texts)

    def test_one_client_per_call(self):
        fake, state = _make_fake_voyageai()
        texts = [f"t{i}" for i in range(10)]
        self._run(texts, fake, batch_size=2)
        self.assertEqual(state["calls"], 5)
        self.assertEqual(state["clients"], 1)

    def test_single_batch_makes_one_request(self):
        fake, state = _make_fake_voyageai()
        texts = [f"t{i}" for i in range(3)]
        blobs, dim = self._run(texts, fake, batch_size=10)
        self.assertEqual(state["calls"], 1)
        self.assertEqual(state["clients"], 1)
        self.assertEqual(len(blobs), 3)
        self.assertEqual(dim, 2)

    def test_output_order_matches_input_when_later_batches_finish_first(self):
        texts = [f"t{i}" for i in range(8)]
        first_text_of_batch = {t: i for i, t in enumerate(texts[::2])}

        def delay_for(batch):
            # Earlier batches sleep longer, so they finish last.
            return 0.2 - 0.04 * first_text_of_batch[batch[0]]

        fake, _ = _make_fake_voyageai(delay_for=delay_for)
        blobs, dim = self._run(texts, fake, batch_size=2)
        self.assertEqual(dim, 2)
        self.assertEqual(len(blobs), len(texts))
        for i, blob in enumerate(blobs):
            first, second = _decode(blob)
            self.assertAlmostEqual(second / first, float(i), places=4)

    def test_in_flight_requests_are_bounded_and_parallel(self):
        fake, state = _make_fake_voyageai(delay_for=lambda batch: 0.05)
        texts = [f"t{i}" for i in range(24)]
        self._run(texts, fake, batch_size=1)
        self.assertEqual(state["calls"], 24)
        self.assertLessEqual(state["peak"], CONCURRENCY)
        self.assertGreater(state["peak"], 1)

    def test_batch_exception_surfaces_same_runtime_error(self):
        fake, _ = _make_fake_voyageai(fail_on=lambda batch: "t5" in batch)
        texts = [f"t{i}" for i in range(8)]
        with self.assertRaises(RuntimeError) as ctx:
            self._run(texts, fake, batch_size=2)
        self.assertEqual(
            str(ctx.exception),
            "Voyage rate limit reached. Wait and retry.\n"
            "Original error: 429 Too Many Requests",
        )


if __name__ == "__main__":
    unittest.main()
