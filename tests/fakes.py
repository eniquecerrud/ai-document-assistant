"""Deterministic stand-ins for the embedding and language-model providers.

Nothing in this module opens a socket. The embedder is a small lexical hash so
tests can still assert that the right passage is retrieved.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from folio.models import ScoredChunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    """Bag-of-tokens hashing embedder with a fixed dimension."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float64)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            vector[0] = 1.0
            return vector.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.tolist()


class FakeLLM:
    """Records the context it was given and returns a cited one-line answer."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[ScoredChunk],
        language: str,
    ) -> str:
        self.calls.append({"question": question, "contexts": list(contexts), "language": language})
        if not contexts:
            return "sin contexto"
        first = contexts[0].chunk
        return f"[1] respuesta en {first.source} p.{first.page}"
