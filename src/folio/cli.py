"""Command-line interface for Folio.

Examples::

    folio ingest samples/politica-vacaciones.md
    folio query "¿Cuántos días de vacaciones hay?"
    folio documents
    folio serve
    folio ui
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from folio.config import load_settings
from folio.errors import FolioError
from folio.service import FolioService, build_service

app = typer.Typer(
    name="folio",
    help="Asistente de documentos con recuperación aumentada.",
    no_args_is_help=True,
)


def _service(config: Path | None) -> FolioService:
    return build_service(load_settings(config))


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(
        ...,
        help="Archivos PDF, TXT o Markdown.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Archivo YAML de configuración."
    ),
) -> None:
    """Indexa uno o más archivos. Volver a cargar el mismo nombre lo reemplaza."""
    service = _service(config)
    failures = 0
    for path in paths:
        try:
            result = service.ingest_file(path)
        except FolioError as exc:
            typer.echo(f"Error en {path.name}: {exc}", err=True)
            failures += 1
            continue
        typer.echo(
            f"Indexado {result.source}: {_count(result.chunks, 'fragmento', 'fragmentos')}, "
            f"{_count(result.pages, 'página', 'páginas')} (id {result.document_id})."
        )
    if failures:
        raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Pregunta sobre los documentos indexados."),
    top_k: int | None = typer.Option(None, "--top-k", min=1, max=20, help="Pasajes a recuperar."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Archivo YAML de configuración."
    ),
) -> None:
    """Responde una pregunta e imprime las citas (archivo y página)."""
    service = _service(config)
    try:
        answer = service.query(question, top_k=top_k)
    except FolioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(answer.text)
    if not answer.citations:
        return
    typer.echo("\nFuentes:")
    for number, citation in enumerate(answer.citations, start=1):
        typer.echo(
            f"  [{number}] {citation.source} p.{citation.page} "
            f"(similitud {citation.score:.2f}) {citation.excerpt}"
        )


@app.command()
def documents(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Archivo YAML de configuración."
    ),
) -> None:
    """Lista los documentos del índice."""
    rows = _service(config).list_documents()
    if not rows:
        typer.echo("El índice está vacío.")
        return
    for row in rows:
        typer.echo(f"{row.id}  {row.source}  {_count(row.chunk_count, 'fragmento', 'fragmentos')}")


@app.command("delete")
def delete(
    document_id: str = typer.Argument(..., help="Identificador devuelto al indexar."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Archivo YAML de configuración."
    ),
) -> None:
    """Elimina un documento del índice."""
    if not _service(config).delete_document(document_id):
        typer.echo(f"No existe el documento {document_id}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Eliminado {document_id}.")


@app.command()
def serve(
    host: str | None = typer.Option(
        None, help="Dirección de escucha. Por defecto, la de la configuración."
    ),
    port: int | None = typer.Option(None, help="Puerto HTTP."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Archivo YAML de configuración."
    ),
) -> None:
    """Arranca la API HTTP."""
    import uvicorn

    settings = load_settings(config)
    service = build_service(settings)
    api = _api(service)
    uvicorn.run(
        api,
        host=host or settings.server.host,
        port=port or settings.server.port,
        log_level="info",
    )


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Dirección de escucha de Streamlit."),
    port: int = typer.Option(8501, help="Puerto de la interfaz."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Archivo YAML de configuración."
    ),
) -> None:
    """Abre la interfaz web para cargar documentos y conversar."""
    ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
    if not ui_path.is_file():
        typer.echo(f"No se encontró la interfaz en {ui_path}.", err=True)
        raise typer.Exit(code=1)
    env = os.environ.copy()
    if config is not None:
        env["FOLIO_CONFIG"] = str(config.resolve())
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    raise typer.Exit(code=subprocess.run(command, env=env, check=False).returncode)


def _count(amount: int, singular: str, plural: str) -> str:
    return f"{amount} {singular if amount == 1 else plural}"


def _api(service: FolioService) -> object:
    from folio.api import create_app

    return create_app(service)
