"""HTTP API for indexing documents and asking questions.

Run it with ``folio serve``. The default bind address is localhost. The API
does not authenticate callers; do not expose it on a public network.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from folio import __version__
from folio.errors import FolioError, IngestError
from folio.service import FolioService, build_service


class QueryRequest(BaseModel):
    """A question against the current index."""

    question: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    """One retrieved passage."""

    source: str
    page: int
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    """Answer plus the passages that were retrieved for it."""

    answer: str
    language: str
    citations: list[CitationOut]


class DocumentOut(BaseModel):
    """An indexed file."""

    id: str
    source: str
    chunks: int


class IngestResponse(BaseModel):
    """Result of indexing one upload."""

    document_id: str
    source: str
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    version: str


def create_app(service: FolioService | None = None) -> FastAPI:
    """Build the API.

    Pass ``service`` in tests. When it is omitted, the first request builds a
    service from the process configuration.
    """
    app = FastAPI(
        title="Folio",
        version=__version__,
        summary="Retrieval-augmented document assistant.",
        description=(
            "Index PDF, TXT, and Markdown files, then ask questions. "
            "Answers include the file and page of the passages that were used."
        ),
    )
    if service is not None:
        app.state.service = service

    def get_service() -> FolioService:
        current = getattr(app.state, "service", None)
        if current is None:
            current = build_service()
            app.state.service = current
        return current

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post("/ingest", response_model=IngestResponse, tags=["documents"])
    async def ingest(
        folio: FolioService = Depends(get_service),
        file: UploadFile = File(description="A PDF, TXT, or Markdown file."),
    ) -> IngestResponse:
        data = await file.read()
        try:
            result = folio.ingest_bytes(file.filename or "", data)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FolioError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return IngestResponse(
            document_id=result.document_id,
            source=result.source,
            pages=result.pages,
            chunks=result.chunks,
        )

    @app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
    def documents(folio: FolioService = Depends(get_service)) -> list[DocumentOut]:
        return [
            DocumentOut(id=item.id, source=item.source, chunks=item.chunk_count)
            for item in folio.list_documents()
        ]

    @app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
    def delete_document(
        document_id: str,
        folio: FolioService = Depends(get_service),
    ) -> None:
        if not folio.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found.")

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    def query(
        body: QueryRequest,
        folio: FolioService = Depends(get_service),
    ) -> QueryResponse:
        try:
            answer = folio.query(body.question, top_k=body.top_k)
        except FolioError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return QueryResponse(
            answer=answer.text,
            language=answer.language,
            citations=[
                CitationOut(
                    source=item.source,
                    page=item.page,
                    chunk_id=item.chunk_id,
                    score=item.score,
                    excerpt=item.excerpt,
                )
                for item in answer.citations
            ],
        )

    return app


app = create_app()
