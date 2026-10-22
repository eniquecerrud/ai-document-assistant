"""CLI commands delegate to the injected service and surface citations."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from folio.cli import app
from folio.service import FolioService


def test_ingest_query_documents_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service: FolioService,
) -> None:
    monkeypatch.setattr("folio.cli.build_service", lambda settings=None: service)
    sample = tmp_path / "nota.txt"
    sample.write_text("Las vacaciones son 22 dias laborables.", encoding="utf-8")
    runner = CliRunner()

    ingested = runner.invoke(app, ["ingest", str(sample)])
    assert ingested.exit_code == 0, ingested.output
    assert "Indexado nota.txt" in ingested.output

    listed = runner.invoke(app, ["documents"])
    assert listed.exit_code == 0
    assert "nota.txt" in listed.output

    asked = runner.invoke(app, ["query", "¿Cuantos dias de vacaciones hay?"])
    assert asked.exit_code == 0, asked.output
    assert "Fuentes:" in asked.output
    assert "nota.txt p.1" in asked.output

    document_id = service.list_documents()[0].id
    removed = runner.invoke(app, ["delete", document_id])
    assert removed.exit_code == 0
    missing = runner.invoke(app, ["delete", document_id])
    assert missing.exit_code == 1

    empty = runner.invoke(app, ["documents"])
    assert "vacío" in empty.output


def test_ingest_reports_unsupported_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service: FolioService,
) -> None:
    monkeypatch.setattr("folio.cli.build_service", lambda settings=None: service)
    bad = tmp_path / "dato.bin"
    bad.write_bytes(b"\x00\x01")
    result = CliRunner().invoke(app, ["ingest", str(bad)])
    assert result.exit_code == 1
    assert "Unsupported" in result.output
