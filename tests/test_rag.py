"""Retrieval-augmented answers cite the file and the page."""

from __future__ import annotations

import pytest

from folio.errors import FolioError
from folio.language import detect_language
from folio.service import FolioService


def test_language_heuristic() -> None:
    assert detect_language("¿Cuántos días de vacaciones hay?") == "es"
    assert detect_language("How many vacation days are there?") == "en"
    assert detect_language("Que politica aplica para el equipo") == "es"


def test_empty_index_answers_in_the_question_language(service: FolioService) -> None:
    spanish = service.query("¿Hay documentos cargados?")
    assert spanish.language == "es"
    assert spanish.citations == ()
    assert "No hay documentos" in spanish.text

    english = service.query("Are any documents loaded?")
    assert english.language == "en"
    assert "No documents" in english.text


def test_query_cites_the_matching_source(service: FolioService) -> None:
    service.ingest_bytes(
        "politica-vacaciones.md",
        "Cada persona empleada dispone de 22 dias laborables de vacaciones al año.".encode(),
    )
    service.ingest_bytes(
        "guia-seguridad.md",
        b"Un incidente de seguridad se informa en menos de 24 horas.",
    )
    answer = service.query("¿Cuántos dias de vacaciones tiene una persona empleada?")
    assert answer.language == "es"
    assert answer.citations
    assert answer.citations[0].source == "politica-vacaciones.md"
    assert answer.citations[0].page == 1
    assert "22 dias" in answer.citations[0].excerpt
    assert "[1]" in answer.text
    assert "politica-vacaciones.md" in answer.text

    llm = service.llm
    assert llm.calls  # type: ignore[attr-defined]
    call = llm.calls[0]  # type: ignore[attr-defined]
    assert call["language"] == "es"
    assert call["contexts"][0].chunk.page == 1


def test_reingest_replaces_the_file(service: FolioService) -> None:
    service.ingest_bytes("nota.txt", b"version uno de las vacaciones anuales")
    first = service.list_documents()[0]
    service.ingest_bytes("nota.txt", b"version dos de las vacaciones anuales con mas detalle")
    documents = service.list_documents()
    assert len(documents) == 1
    assert documents[0].id == first.id
    answer = service.query("vacaciones anuales")
    assert "version dos" in answer.citations[0].excerpt


def test_directory_components_are_stripped(service: FolioService) -> None:
    result = service.ingest_bytes("../../notas/politica.txt", b"Texto de la politica interna.")
    assert result.source == "politica.txt"
    assert service.list_documents()[0].source == "politica.txt"


def test_extractive_provider_quotes_spanish_and_english(extractive_service: FolioService) -> None:
    extractive_service.ingest_bytes(
        "politica-vacaciones.md",
        (
            "Cada persona empleada dispone de 22 dias laborables de vacaciones al año. "
            "La solicitud se envia con al menos 15 dias de antelacion."
        ).encode(),
    )
    extractive_service.ingest_bytes(
        "handbook.txt",
        b"People may work remotely up to 3 days per week. Core hours are 10:00 to 15:00.",
    )
    vacation = extractive_service.query(
        "¿Cuantos dias de vacaciones y con que antelacion se solicitan?"
    )
    assert vacation.citations[0].source == "politica-vacaciones.md"
    assert "22 dias" in vacation.text
    assert "p. 1" in vacation.text

    remote = extractive_service.query("How many days per week can people work remotely?")
    assert remote.language == "en"
    assert remote.citations[0].source == "handbook.txt"
    assert "3 days" in remote.text


def test_empty_question_is_rejected(service: FolioService) -> None:
    with pytest.raises(FolioError, match="empty"):
        service.query("   ")
