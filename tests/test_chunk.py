"""Chunking keeps page numbers and repeats text across the overlap."""

from __future__ import annotations

import pytest

from folio.ingest import chunk_pages, split_text
from folio.models import PageText


def test_short_text_is_one_chunk() -> None:
    parts = split_text("Una sola frase sobre vacaciones.", size=200, overlap=20)
    assert parts == ["Una sola frase sobre vacaciones."]


def test_blank_text_yields_nothing() -> None:
    assert split_text("  \n\n\t", size=100, overlap=10) == []


def test_overlap_repeats_the_tail() -> None:
    words = [f"token{index:03d}" for index in range(80)]
    parts = split_text(" ".join(words), size=120, overlap=40)
    assert len(parts) >= 2
    assert all(len(part) <= 120 for part in parts)
    shared = set(parts[0].split()[-3:]) & set(parts[1].split()[:6])
    assert shared, "expected the next window to repeat words from the previous tail"


def test_paragraphs_are_preferred_until_the_window_fills() -> None:
    text = "Primer parrafo corto.\n\nSegundo parrafo corto."
    parts = split_text(text, size=200, overlap=20)
    assert parts == ["Primer parrafo corto.\n\nSegundo parrafo corto."]


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("hola", size=10, overlap=10)


def test_page_numbers_survive_chunking() -> None:
    pages = [
        PageText(page=1, text="uno " * 40),
        PageText(page=2, text=""),
        PageText(page=3, text="tres " * 40),
    ]
    chunks = chunk_pages(pages, document_id="doc", source="nota.pdf", size=50, overlap=10)
    assert chunks
    assert {chunk.page for chunk in chunks} == {1, 3}
    assert all(chunk.source == "nota.pdf" for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].id.startswith("doc-")
    assert all(chunk.document_id == "doc" for chunk in chunks)
