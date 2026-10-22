"""Value objects shared by ingestion, retrieval, and generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Text extracted from one page.

    Plain-text and Markdown files are treated as a single page numbered ``1``.
    PDF page numbers are 1-based and follow the order of the file.
    """

    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit small enough to embed and precise enough to cite."""

    id: str
    document_id: str
    source: str
    page: int
    index: int
    text: str

    def label(self) -> str:
        """Human-readable citation label, for example ``policy.md, p. 1``."""
        return f"{self.source}, p. {self.page}"


@dataclass(frozen=True)
class DocumentRecord:
    """Summary of one indexed file."""

    id: str
    source: str
    chunk_count: int


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk and its similarity to the query, higher is closer."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    """A source passage returned next to an answer."""

    source: str
    page: int
    chunk_id: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class Answer:
    """A grounded answer. ``citations`` are the passages that were retrieved."""

    text: str
    citations: tuple[Citation, ...]
    language: str


@dataclass(frozen=True)
class IngestResult:
    """What happened when a single file was indexed."""

    document_id: str
    source: str
    pages: int
    chunks: int
