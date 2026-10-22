"""Shared fixtures. Every test stays on disk and on the loopback stack."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.fakes import FakeEmbedder, FakeLLM

from folio.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    ServerConfig,
    Settings,
    VectorStoreConfig,
)
from folio.llm import ExtractiveLLM
from folio.service import FolioService
from folio.vectorstore import LocalVectorStore


@pytest.fixture(autouse=True)
def _block_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if application code opens a real TCP connection."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"Tests must not open network connections (args={args!r}).")

    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        chunking=ChunkingConfig(size=240, overlap=40),
        embedding=EmbeddingConfig(provider="hashing", model="ignored-by-fake"),
        llm=LLMConfig(provider="extractive", model="ignored-by-fake"),
        vectorstore=VectorStoreConfig(provider="local", path=str(tmp_path / "store")),
        retrieval=RetrievalConfig(top_k=3, min_score=0.0),
        server=ServerConfig(host="127.0.0.1", port=8765),
        openai_api_key=None,
        openai_base_url="https://example.test/v1",
        ollama_base_url="http://ollama.test",
    )


@pytest.fixture
def service(settings: Settings) -> Iterator[FolioService]:
    store = LocalVectorStore(settings.vectorstore.path)
    yield FolioService(settings, FakeEmbedder(), store, FakeLLM())


@pytest.fixture
def extractive_service(settings: Settings) -> FolioService:
    """Real offline providers: lexical hashing embeddings and quoted answers."""
    store = LocalVectorStore(Path(settings.vectorstore.path) / "extractive")
    from folio.embeddings import HashingEmbeddings

    return FolioService(settings, HashingEmbeddings(), store, ExtractiveLLM())
