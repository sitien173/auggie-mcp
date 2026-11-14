"""Deterministic fake embedder for unit tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


class FakeEmbedder:
    """Deterministic hash-based embedder that records calls and text counts."""

    def __init__(
        self,
        model: str = "voyage-code-3",
        dimensions: int = 16,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.model_key = f"{model}:{dimensions}:float32"
        self.calls: list[list[str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def text_count(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        batch = list(texts)
        if not batch:
            return []
        self.calls.append(batch)
        vectors: list[np.ndarray] = []
        for text in batch:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dimensions).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = (vec / norm).astype(np.float32)
            vectors.append(vec)
        return vectors
