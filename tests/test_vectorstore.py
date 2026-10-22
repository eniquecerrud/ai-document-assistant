"""The local index persists, replaces documents, and ranks by cosine similarity."""

from __future__ import annotations

from pathlib import Path

import pytest

from folio.errors import FolioError
from folio.models import Chunk
from folio.vectorstore import LocalVectorStore


def _chunk(document_id: str, index: int, text: str, source: str = "a.md") -> Chunk:
    return Chunk(
        id=f"{document_id}-{index:04d}",
        document_id=document_id,
        source=source,
        page=index + 1,
        index=index,
        text=text,
    )


def test_search_prefers_the_aligned_vector(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path / "store")
    vacation = _chunk("d1", 0, "vacaciones", source="politica.md")
    security = _chunk("d2", 0, "seguridad", source="guia.md")
    store.replace_document("d1", [vacation], [[1.0, 0.0, 0.0]])
    store.replace_document("d2", [security], [[0.0, 1.0, 0.0]])

    hits = store.search([1.0, 0.1, 0.0], top_k=2)
    assert [hit.chunk.source for hit in hits] == ["politica.md", "guia.md"]
    assert hits[0].score == pytest.approx(1.0, abs=0.01)

    assert store.search([0.0, 0.0, 1.0], top_k=2, min_score=0.5) == []


def test_replace_does_not_duplicate(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path / "store")
    store.replace_document("d1", [_chunk("d1", 0, "viejo")], [[1.0, 0.0]])
    store.replace_document("d1", [_chunk("d1", 0, "nuevo")], [[0.0, 1.0]])
    assert store.count() == 1
    assert store.list_documents()[0].chunk_count == 1
    hits = store.search([0.0, 1.0], top_k=1)
    assert hits[0].chunk.text == "nuevo"


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "store"
    store = LocalVectorStore(path)
    store.replace_document(
        "d1",
        [_chunk("d1", 0, "Madrid cierra en diciembre", source="nota.pdf")],
        [[0.0, 2.0]],
    )
    reloaded = LocalVectorStore(path)
    assert reloaded.count() == 1
    document = reloaded.list_documents()[0]
    assert document.source == "nota.pdf"
    assert document.id == "d1"
    hits = reloaded.search([0.0, 1.0], top_k=1)
    assert hits[0].chunk.page == 1
    assert "Madrid" in hits[0].chunk.text

    manifest = (path / "manifest.json").read_text(encoding="utf-8")
    assert "nota.pdf" in manifest


def test_delete_unknown_and_last_document(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path / "store")
    assert store.delete_document("missing") is False
    store.replace_document("d1", [_chunk("d1", 0, "hola")], [[1.0, 0.0]])
    assert store.delete_document("d1") is True
    assert store.count() == 0
    assert store.list_documents() == []
    store.replace_document("d2", [_chunk("d2", 0, "otra", source="b.txt")], [[0.0, 3.0]])
    assert store.search([0.0, 1.0], top_k=1)[0].chunk.source == "b.txt"


def test_dimension_mismatch_keeps_other_documents(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path / "store")
    store.replace_document("d1", [_chunk("d1", 0, "uno")], [[1.0, 0.0]])
    with pytest.raises(FolioError, match="dimension"):
        store.replace_document("d2", [_chunk("d2", 0, "dos")], [[1.0, 0.0, 0.0]])
    assert store.count() == 1


def test_mismatched_chunk_and_vector_counts(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path / "store")
    with pytest.raises(FolioError, match="exactly one embedding"):
        store.replace_document("d1", [_chunk("d1", 0, "uno")], [])
