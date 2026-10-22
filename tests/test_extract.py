"""Text extraction for the three supported formats."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.pdfutil import simple_pdf

from folio.errors import IngestError
from folio.ingest import extract

_SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_markdown_is_a_single_page() -> None:
    pages = extract("politica.md", b"# Titulo\n\nVacaciones: 22 dias.")
    assert len(pages) == 1
    assert pages[0].page == 1
    assert "22 dias" in pages[0].text


def test_utf8_bom_is_stripped() -> None:
    pages = extract("nota.txt", "café".encode("utf-8-sig"))
    assert pages[0].text == "café"


def test_unsupported_type() -> None:
    with pytest.raises(IngestError, match="Unsupported"):
        extract("virus.exe", b"MZ")


def test_empty_text_file() -> None:
    with pytest.raises(IngestError, match="No text"):
        extract("vacio.txt", b"   \n")


def test_pdf_pages_are_extracted_separately() -> None:
    payload = simple_pdf(
        [
            "La oficina de Madrid cierra los dias 24 y 31 de diciembre.",
            "El comedor abre de 13:00 a 15:30.",
        ]
    )
    pages = extract("nota.pdf", payload)
    assert [page.page for page in pages] == [1, 2]
    assert "Madrid" in pages[0].text
    assert "13:00" in pages[1].text


def test_damaged_pdf() -> None:
    with pytest.raises(IngestError, match="Could not read PDF"):
        extract("roto.pdf", b"%PDF-1.4 this is not a real pdf")


def test_shipped_samples_have_text() -> None:
    for name in ("politica-vacaciones.md", "handbook.txt", "guia-seguridad.md", "nota.pdf"):
        path = _SAMPLES / name
        pages = extract(path.name, path.read_bytes())
        assert any(page.text.strip() for page in pages), name
