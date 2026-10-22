"""The HTTP API ingests, lists, queries, and deletes without leaving the process."""

from __future__ import annotations

from fastapi.testclient import TestClient

from folio.api import create_app
from folio.service import FolioService


def test_health_ingest_query_and_delete(service: FolioService) -> None:
    with TestClient(create_app(service)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        empty = client.post("/query", json={"question": "¿Hay documentos?"})
        assert empty.status_code == 200
        assert empty.json()["citations"] == []
        assert empty.json()["language"] == "es"

        uploaded = client.post(
            "/ingest",
            files={
                "file": ("politica.md", "22 dias de vacaciones al año.".encode(), "text/markdown")
            },
        )
        assert uploaded.status_code == 200
        body = uploaded.json()
        assert body["source"] == "politica.md"
        assert body["pages"] == 1
        assert body["chunks"] >= 1

        listed = client.get("/documents").json()
        assert listed == [
            {"id": body["document_id"], "source": "politica.md", "chunks": body["chunks"]}
        ]

        answer = client.post("/query", json={"question": "vacaciones", "top_k": 2})
        assert answer.status_code == 200
        citation = answer.json()["citations"][0]
        assert citation["source"] == "politica.md"
        assert citation["page"] == 1
        assert "vacaciones" in citation["excerpt"]

        missing = client.delete("/documents/does-not-exist")
        assert missing.status_code == 404
        deleted = client.delete(f"/documents/{body['document_id']}")
        assert deleted.status_code == 204
        assert client.get("/documents").json() == []


def test_rejects_unknown_types_and_blank_questions(service: FolioService) -> None:
    with TestClient(create_app(service)) as client:
        rejected = client.post(
            "/ingest",
            files={"file": ("nota.exe", b"nope", "application/octet-stream")},
        )
        assert rejected.status_code == 400
        invalid = client.post("/query", json={"question": ""})
        assert invalid.status_code == 422
        unnamed = client.post("/ingest", files={"file": ("..", b"hola", "text/plain")})
        assert unnamed.status_code == 400
