"""Language-model backends.

``extractive`` quotes the retrieved passages and needs no network. It is the
default so the project runs immediately. ``openai`` talks to any
OpenAI-compatible chat endpoint. ``ollama`` talks to a local Ollama server.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from folio.config import Settings
from folio.errors import ProviderError
from folio.language import language_name
from folio.models import ScoredChunk
from folio.textutil import excerpt


class LLMProvider(Protocol):
    """Generate an answer from a question and the passages that support it."""

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[ScoredChunk],
        language: str,
    ) -> str:
        """Return the answer text. Citations are attached by the engine."""


class ExtractiveLLM:
    """Offline answerer that shows the retrieved passages with their locations.

    This is not a generative model. It is the honest fallback when no API key
    and no local model are configured, and it is what the test suite uses as a
    reference shape for cited answers.
    """

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[ScoredChunk],
        language: str,
    ) -> str:
        del question  # The passages are the answer; the question selects them.
        intro = (
            "En los documentos indexados aparece lo siguiente:"
            if language == "es"
            else "The indexed documents say the following:"
        )
        blocks = [intro]
        for number, hit in enumerate(contexts, start=1):
            chunk = hit.chunk
            blocks.append(f"[{number}] {chunk.label()}: {excerpt(chunk.text, limit=500)}")
        return "\n\n".join(blocks)


class OpenAICompatibleLLM:
    """Chat completions against ``{base_url}/chat/completions``."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        self._timeout = timeout

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[ScoredChunk],
        language: str,
    ) -> str:
        if not self.api_key:
            raise ProviderError(
                "FOLIO_OPENAI_API_KEY is not set. Add it to .env or the environment."
            )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt(language)},
                {"role": "user", "content": user_prompt(question, contexts)},
            ],
        }
        response = self._request(body)
        if response.status_code >= 400:
            raise ProviderError(
                f"Chat completion failed ({response.status_code}): {_short(response.text)}"
            )
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Chat completion response had no message.") from exc
        return str(content).strip()

    def _request(self, body: dict[str, object]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        if self._client is not None:
            return self._client.post(url, headers=headers, json=body)
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(url, headers=headers, json=body)


class OllamaLLM:
    """Chat against the Ollama ``/api/chat`` endpoint, with streaming disabled."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float = 0.2,
        client: httpx.Client | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self._client = client
        self._timeout = timeout

    def generate(
        self,
        *,
        question: str,
        contexts: Sequence[ScoredChunk],
        language: str,
    ) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt(language)},
                {"role": "user", "content": user_prompt(question, contexts)},
            ],
            "options": {"temperature": self.temperature},
        }
        url = f"{self.base_url}/api/chat"
        if self._client is not None:
            response = self._client.post(url, json=body)
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=body)
        if response.status_code >= 400:
            raise ProviderError(
                f"Ollama chat failed ({response.status_code}): {_short(response.text)}"
            )
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("Ollama chat response had no message.") from exc
        return str(content).strip()


def system_prompt(language: str) -> str:
    """Instruction that keeps the model inside the retrieved context."""
    name = language_name(language)
    return (
        "You are Folio, a careful document assistant. "
        "Answer only with information found in the numbered context blocks. "
        "Do not invent facts, numbers, or citations. "
        f"Write the answer in {name}. "
        "Cite the supporting blocks inline with their bracket numbers, such as [1] or [2]. "
        "If the context is not enough to answer, say so in one sentence and do not guess."
    )


def user_prompt(question: str, contexts: Sequence[ScoredChunk]) -> str:
    """Render the question under the passages the model is allowed to use."""
    blocks: list[str] = []
    for number, hit in enumerate(contexts, start=1):
        chunk = hit.chunk
        blocks.append(f"[{number}] source={chunk.source} page={chunk.page}\n{chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no context)"
    return f"Context:\n{context}\n\nQuestion: {question.strip()}"


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the language-model provider named in settings."""
    name = settings.llm.provider.strip().lower()
    if name == "extractive":
        return ExtractiveLLM()
    if name == "openai":
        return OpenAICompatibleLLM(
            model=settings.llm.model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
    if name == "ollama":
        return OllamaLLM(
            model=settings.llm.model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm.temperature,
        )
    raise ProviderError(
        f"Unknown LLM provider '{settings.llm.provider}'. Use extractive, openai, or ollama."
    )


def _short(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
