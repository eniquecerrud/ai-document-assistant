"""Facade used by the CLI, the HTTP API, and the Streamlit UI."""

from __future__ import annotations

from pathlib import Path

from folio.config import Settings, load_settings
from folio.embeddings import EmbeddingProvider, build_embedder
from folio.engine import RagEngine
from folio.errors import IngestError
from folio.ingest import chunk_pages, document_id_for, extract
from folio.llm import LLMProvider, build_llm
from folio.models import Answer, DocumentRecord, IngestResult
from folio.vectorstore import VectorStore, build_store

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class FolioService:
    """Index files and answer questions against one vector store."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.engine = RagEngine(
            embedder=embedder,
            store=store,
            llm=llm,
            top_k=settings.retrieval.top_k,
            min_score=settings.retrieval.min_score,
        )

    def ingest_file(self, path: Path | str) -> IngestResult:
        """Read a path from disk and index it. Only the file name is stored."""
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise IngestError(f"File not found: {file_path}")
        return self.ingest_bytes(file_path.name, file_path.read_bytes())

    def ingest_bytes(self, filename: str, data: bytes) -> IngestResult:
        """Index an upload. Directory components in ``filename`` are discarded."""
        source = _safe_name(filename)
        if len(data) > MAX_UPLOAD_BYTES:
            raise IngestError("The file exceeds the 20 MB limit.")
        pages = extract(source, data)
        document_id = document_id_for(source)
        chunks = chunk_pages(
            pages,
            document_id=document_id,
            source=source,
            size=self.settings.chunking.size,
            overlap=self.settings.chunking.overlap,
        )
        if not chunks:
            raise IngestError(f"No text could be extracted from {source}.")
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise IngestError("The embedding provider returned an unexpected number of vectors.")
        self.store.replace_document(document_id, chunks, vectors)
        return IngestResult(
            document_id=document_id,
            source=source,
            pages=len(pages),
            chunks=len(chunks),
        )

    def query(self, question: str, *, top_k: int | None = None) -> Answer:
        """Answer one question. The call does not read earlier turns."""
        return self.engine.answer(question, top_k=top_k)

    def list_documents(self) -> list[DocumentRecord]:
        """Return the files currently in the index."""
        return self.store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        """Remove one document. Return ``False`` if the id is unknown."""
        return self.store.delete_document(document_id)


def build_service(
    settings: Settings | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    llm: LLMProvider | None = None,
) -> FolioService:
    """Wire the providers selected by ``settings``.

    Tests pass ``embedder``, ``store``, and ``llm`` to avoid network access.
    """
    resolved = settings or load_settings()
    return FolioService(
        resolved,
        embedder or build_embedder(resolved),
        store or build_store(resolved),
        llm or build_llm(resolved),
    )


def _safe_name(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name or name in {".", ".."} or "\x00" in name:
        raise IngestError("A file name is required.")
    return name
