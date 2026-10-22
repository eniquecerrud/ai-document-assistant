"""YAML defaults and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from folio.config import load_settings
from folio.errors import ConfigError


def _clear_folio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("FOLIO_"):
            monkeypatch.delenv(key, raising=False)


def test_repository_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_folio_env(monkeypatch)
    settings = load_settings(Path("folio.yaml"))
    assert settings.embedding.provider == "sentence-transformers"
    assert "multilingual" in settings.embedding.model
    assert settings.llm.provider == "extractive"
    assert settings.vectorstore.provider == "local"
    assert settings.chunking.overlap < settings.chunking.size
    assert settings.openai_api_key is None


def test_environment_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_folio_env(monkeypatch)
    (tmp_path / "folio.yaml").write_text(
        "embedding:\n  provider: ollama\n  model: nomic-embed-text\n"
        "llm:\n  provider: ollama\n  model: llama3.2\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOLIO_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FOLIO_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("FOLIO_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("FOLIO_OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("FOLIO_TOP_K", "2")
    monkeypatch.setenv("FOLIO_VECTORSTORE_PATH", str(tmp_path / "index"))

    settings = load_settings()
    assert settings.embedding.provider == "ollama"
    assert settings.embedding.model == "nomic-embed-text"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"
    assert settings.openai_api_key == "sk-test"
    assert settings.openai_base_url == "https://example.test/v1"
    assert settings.retrieval.top_k == 2
    assert settings.vectorstore.path == str(tmp_path / "index")


def test_blank_api_key_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_folio_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOLIO_OPENAI_API_KEY", "   ")
    settings = load_settings()
    assert settings.openai_api_key is None
    assert settings.llm.provider == "extractive"


def test_invalid_overlap_and_missing_file(tmp_path: Path) -> None:
    broken = tmp_path / "bad.yaml"
    broken.write_text("chunking:\n  size: 120\n  overlap: 120\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(broken)
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.yaml")
