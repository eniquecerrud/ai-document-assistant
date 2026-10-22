"""Retrieve passages and ask a language model to answer from them.

The structured citations always come from retrieval, not from whatever the
model decides to mention. A model can forget a marker; the caller still
receives the file and page of every passage that was provided.
"""

from __future__ import annotations

from folio.embeddings import EmbeddingProvider
from folio.errors import FolioError
from folio.language import detect_language, empty_index_message, no_match_message
from folio.llm import LLMProvider
from folio.models import Answer, Citation
from folio.textutil import excerpt
from folio.vectorstore import VectorStore


class RagEngine:
    """One-shot retrieval-augmented question answering.

    Each call is independent. Folio does not keep conversational memory, so a
    follow-up question has to stand on its own.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        """Answer ``question`` from the current index.

        Raises:
            FolioError: The question is empty, or a provider fails.
        """
        cleaned = question.strip()
        if not cleaned:
            raise FolioError("The question is empty.")
        language = detect_language(cleaned)
        if self.store.count() == 0:
            return Answer(text=empty_index_message(language), citations=(), language=language)

        vector = self.embedder.embed_query(cleaned)
        hits = self.store.search(
            vector,
            top_k=top_k or self.top_k,
            min_score=self.min_score,
        )
        if not hits:
            return Answer(text=no_match_message(language), citations=(), language=language)

        text = self.llm.generate(question=cleaned, contexts=hits, language=language)
        citations = tuple(
            Citation(
                source=hit.chunk.source,
                page=hit.chunk.page,
                chunk_id=hit.chunk.id,
                score=round(float(hit.score), 4),
                excerpt=excerpt(hit.chunk.text),
            )
            for hit in hits
        )
        return Answer(text=text.strip(), citations=citations, language=language)
