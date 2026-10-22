"""Extract text from PDF, TXT, and Markdown files and split it into chunks.

PDF pages keep their page number. Text and Markdown are a single page so every
chunk can still be cited as ``file, p. 1``. Chunks overlap on purpose: a fact
that sits on a boundary remains inside at least one complete window.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from folio.errors import IngestError
from folio.models import Chunk, PageText

_SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}
_PARAGRAPH = re.compile(r"\n\s*\n")


def document_id_for(source: str) -> str:
    """Stable id for a file name.

    Re-ingesting the same file name replaces the previous version. The full
    path is ignored so ``notes/a.md`` and ``other/a.md`` refer to one document.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def extract(filename: str, data: bytes) -> list[PageText]:
    """Return one :class:`PageText` per page.

    Raises:
        IngestError: The type is unsupported, the PDF cannot be read, or the
            file has no extractable text. Scanned PDFs without a text layer
            fail here; Folio does not run OCR.
    """
    source = Path(filename).name
    suffix = Path(source).suffix.lower()
    if suffix not in _SUPPORTED:
        supported = ", ".join(sorted(_SUPPORTED))
        raise IngestError(
            f"Unsupported file type '{suffix or 'none'}'. Supported types: {supported}."
        )
    pages = _pages_from(suffix, data)
    if not any(page.text.strip() for page in pages):
        raise IngestError(f"No text could be extracted from {source}.")
    return pages


def chunk_pages(
    pages: list[PageText],
    *,
    document_id: str,
    source: str,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Split pages into citeable chunks.

    ``size`` and ``overlap`` are character counts. Overlap must be smaller
    than size. Empty pages are skipped. Chunk indexes are global to the file,
    starting at 0.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, size=size, overlap=overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document_id}-{index:04d}",
                    document_id=document_id,
                    source=source,
                    page=page.page,
                    index=index,
                    text=piece,
                )
            )
    return chunks


def split_text(text: str, *, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows, preferring paragraph breaks.

    A trailing overlap fragment is kept only as the prefix of the next piece.
    It is not emitted as a chunk of its own.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    pieces: list[str] = []
    buffer = ""
    # True when ``buffer`` is only the overlap tail of a piece already emitted.
    carry_only = False
    for paragraph in _paragraphs(normalized):
        if len(paragraph) > size:
            if buffer and not carry_only:
                pieces.append(buffer)
            windowed = _window(paragraph, size=size, overlap=overlap)
            pieces.extend(windowed)
            buffer = _overlap_tail(windowed[-1], overlap) if windowed else ""
            carry_only = True
            continue
        if carry_only:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            buffer = candidate if len(candidate) <= size else paragraph
            carry_only = False
            continue
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        tail = _overlap_tail(buffer, overlap)
        merged = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        buffer = merged if len(merged) <= size else paragraph
        carry_only = False
    if buffer and not carry_only:
        pieces.append(buffer)
    return [piece for piece in pieces if piece.strip()]


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in _PARAGRAPH.split(text) if part.strip()]


def _overlap_tail(text: str, overlap: int) -> str:
    """Return the trailing overlap, moved forward to the next word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if 0 < space < len(tail) - 1:
        tail = tail[space + 1 :]
    return tail.strip()


def _window(text: str, *, size: int, overlap: int) -> list[str]:
    """Slide a window across a single over-long paragraph."""
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + size)
        if end < length:
            snap = text.rfind(" ", start + 1, end)
            if snap > start:
                end = snap
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        while next_start < length and text[next_start].isspace():
            next_start += 1
        start = next_start
    return pieces


def _pages_from(suffix: str, data: bytes) -> list[PageText]:
    if suffix == ".pdf":
        return _extract_pdf(data)
    return [PageText(page=1, text=_decode_text(data))]


def _extract_pdf(data: bytes) -> list[PageText]:
    try:
        reader = PdfReader(stream=io.BytesIO(data))
        if reader.is_encrypted:
            raise IngestError("Encrypted PDFs are not supported.")
        return [
            PageText(page=number, text=page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    except IngestError:
        raise
    except (PdfReadError, ValueError, OSError) as exc:
        raise IngestError(f"Could not read PDF: {exc}") from exc


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")
