"""HTTP providers are exercised with a fake transport. No API key is required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from folio.config import EmbeddingConfig, LLMConfig, Settings
from folio.embeddings import (
    HashingEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    build_embedder,
)
from folio.errors import ProviderError
from folio.llm import OllamaLLM, OpenAICompatibleLLM, build_llm, system_prompt
from folio.models import Chunk, ScoredChunk
from folio.vectorstore import ChromaVectorStore


def _hit() -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id="doc-0000",
            document_id="doc",
            source="politica.md",
            page=2,
            index=0,
            text="22 dias laborables de vacaciones.",
        ),
        score=0.81,
    )


def test_openai_embeddings_restore_index_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == ["hola", "mundo"]
        assert request.headers["Authorization"] == "Bearer sk-test"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.0]},
                    {"index": 0, "embedding": [0.1, 0.0]},
                ]
            },
        )

    provider = OpenAIEmbeddings(
        model="embed-test",
        api_key="sk-test",
        base_url="https://example.test/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.embed_documents(["hola", "mundo"]) == [[0.1, 0.0], [0.2, 0.0]]
    assert provider.embed_documents([]) == []


def test_openai_embeddings_require_a_key_and_surface_http_errors() -> None:
    missing = OpenAIEmbeddings(model="m", api_key=None, base_url="https://example.test/v1")
    with pytest.raises(ProviderError, match="FOLIO_OPENAI_API_KEY"):
        missing.embed_query("hola")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    provider = OpenAIEmbeddings(
        model="m",
        api_key="sk-test",
        base_url="https://example.test/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError, match="401"):
        provider.embed_query("hola")


def test_ollama_embeddings_parse_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        assert json.loads(request.content)["model"] == "nomic-embed-text"
        return httpx.Response(200, json={"embeddings": [[0.4, 0.2]]})

    provider = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://ollama.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.embed_query("vacaciones") == [0.4, 0.2]


def test_openai_chat_prompt_includes_the_citation() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Son 22 dias laborables [1]."}}]},
        )

    llm = OpenAICompatibleLLM(
        model="gpt-test",
        api_key="sk-test",
        base_url="https://example.test/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    text = llm.generate(question="¿Cuántos días hay?", contexts=[_hit()], language="es")
    assert text == "Son 22 dias laborables [1]."
    body = captured["body"]
    assert isinstance(body, dict)
    messages = body["messages"]
    assert "Spanish" in messages[0]["content"]
    assert "source=politica.md" in messages[1]["content"]
    assert "page=2" in messages[1]["content"]
    assert "22 dias" in messages[1]["content"]


def test_ollama_chat_reads_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "Core hours are 10:00 [1]."}})

    llm = OllamaLLM(
        model="llama3.2",
        base_url="http://ollama.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    text = llm.generate(question="What are the core hours?", contexts=[_hit()], language="en")
    assert "10:00" in text
    assert "English" in system_prompt("en")


def test_unknown_providers_are_rejected(settings: Settings) -> None:
    embedding = settings.model_copy(
        update={"embedding": EmbeddingConfig(provider="nope", model="x")}
    )
    with pytest.raises(ProviderError, match="embedding"):
        build_embedder(embedding)
    language = settings.model_copy(update={"llm": LLMConfig(provider="nope", model="x")})
    with pytest.raises(ProviderError, match="LLM"):
        build_llm(language)


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbeddings()
    first = embedder.embed_query("vacaciones anuales")
    second = embedder.embed_query("vacaciones anuales")
    assert first == second
    assert len(first) == 256
    norm = sum(value * value for value in first) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_optional_backends_import_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("sentence_transformers", None)
    SentenceTransformerEmbeddings("some-model")
    assert "sentence_transformers" not in sys.modules

    real_import = __import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    store = ChromaVectorStore(tmp_path / "chroma")
    with pytest.raises(ProviderError, match="chromadb"):
        store.count()
