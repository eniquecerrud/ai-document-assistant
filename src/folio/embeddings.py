"""Pluggable embedding providers.

The default provider is ``sentence-transformers`` with a multilingual model.
``hashing`` is an offline lexical baseline (used by the container). OpenAI-compatible
HTTP endpoints and a local Ollama server are the networked options. Heavy libraries
are imported only when that provider is actually used.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np

from folio.config import Settings
from folio.errors import ProviderError

_TOKEN = re.compile(r"\w+", re.UNICODE)


class EmbeddingProvider(Protocol):
    """Turns text into dense vectors.

    ``embed_query`` is separate from ``embed_documents`` because some models
    expect a different prefix for queries. The providers shipped here use the
    same endpoint for both.
    """

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document chunks. An empty batch returns ``[]``."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question."""


class HashingEmbeddings:
    """Lexical hashing embedder that does not download a model.

    Words and character trigrams are projected into a fixed vector, so passages
    that share vocabulary sit near each other. Synonyms and paraphrases do not.
    It is the offline baseline used by the container image. Install the
    ``local`` extra and switch the provider to ``sentence-transformers`` for
    semantic search in Spanish and English.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 32:
            raise ProviderError("Hashing embedding dimension must be at least 32.")
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float64)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            vector[0] = 1.0
            return vector.tolist()
        for token in tokens:
            _accumulate(vector, token, weight=1.0)
            padded = f" {token} "
            for index in range(len(padded) - 2):
                _accumulate(vector, padded[index : index + 3], weight=0.25)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.tolist()


class SentenceTransformerEmbeddings:
    """Local embeddings via ``sentence-transformers``.

    The model is downloaded from Hugging Face the first time it is used and
    then reused for the life of the process.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: object | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ProviderError(
                    "sentence-transformers is not installed. "
                    'Install it with: pip install "folio[local]"'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model


class OpenAIEmbeddings:
    """Embeddings on any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self._post("/embeddings", {"model": self.model, "input": list(texts)})
        try:
            rows = payload["data"]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Embedding response did not contain data.") from exc
        ordered = sorted(rows, key=lambda item: item.get("index", 0))
        try:
            return [list(item["embedding"]) for item in ordered]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Embedding response was missing vectors.") from exc

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise ProviderError("Embedding provider returned no vector.")
        return vectors[0]

    def _post(self, path: str, body: dict[str, object]) -> dict[str, object]:
        if not self.api_key:
            raise ProviderError(
                "FOLIO_OPENAI_API_KEY is not set. Add it to .env or the environment."
            )
        response = self._request(path, body)
        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI embeddings failed ({response.status_code}): {_short(response.text)}"
            )
        try:
            parsed = response.json()
        except ValueError as exc:
            raise ProviderError("OpenAI embeddings returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ProviderError("OpenAI embeddings returned an unexpected payload.")
        return parsed

    def _request(self, path: str, body: dict[str, object]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}{path}"
        if self._client is not None:
            return self._client.post(url, headers=headers, json=body)
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(url, headers=headers, json=body)


class OllamaEmbeddings:
    """Embeddings served by the Ollama ``/api/embed`` endpoint (Ollama 0.5+)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._request({"model": self.model, "input": list(texts)})
        if response.status_code >= 400:
            raise ProviderError(
                f"Ollama embeddings failed ({response.status_code}): {_short(response.text)}"
            )
        try:
            payload = response.json()
            vectors = payload["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("Ollama embeddings response was missing vectors.") from exc
        return [list(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise ProviderError("Ollama returned no embedding.")
        return vectors[0]

    def _request(self, body: dict[str, object]) -> httpx.Response:
        url = f"{self.base_url}/api/embed"
        if self._client is not None:
            return self._client.post(url, json=body)
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(url, json=body)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Construct the embedding provider named in settings."""
    name = settings.embedding.provider.strip().lower()
    model = settings.embedding.model
    if name in {"sentence-transformers", "sentence_transformers", "local"}:
        return SentenceTransformerEmbeddings(model, batch_size=settings.embedding.batch_size)
    if name == "hashing":
        return HashingEmbeddings()
    if name == "openai":
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    if name == "ollama":
        return OllamaEmbeddings(model=model, base_url=settings.ollama_base_url)
    raise ProviderError(
        f"Unknown embedding provider '{settings.embedding.provider}'. "
        "Use sentence-transformers, hashing, openai, or ollama."
    )


def _accumulate(vector: np.ndarray, feature: str, weight: float) -> None:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "little") % vector.shape[0]
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[bucket] += sign * weight


def _short(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
