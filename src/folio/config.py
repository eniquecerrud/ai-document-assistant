"""Load Folio settings from a YAML file and ``FOLIO_*`` environment variables.

Environment variables always win over the file. Secrets (the OpenAI API key)
are read only from the environment or a local ``.env`` file, never from YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from folio.errors import ConfigError

_DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


class ChunkingConfig(BaseModel):
    """Character window used to split extracted text."""

    model_config = ConfigDict(extra="ignore")

    size: int = Field(default=800, ge=100, le=8000)
    overlap: int = Field(default=120, ge=0)

    @field_validator("overlap")
    @classmethod
    def overlap_fits(cls, value: int, info: Any) -> int:
        size = info.data.get("size", 800)
        if value >= size:
            raise ValueError("overlap must be smaller than size")
        return value


class EmbeddingConfig(BaseModel):
    """Which embedding provider and model to use."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "sentence-transformers"
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = Field(default=32, ge=1, le=256)


class LLMConfig(BaseModel):
    """Which language-model provider and model to use."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "extractive"
    model: str = "llama3.2"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=32, le=8192)


class VectorStoreConfig(BaseModel):
    """Where vectors are persisted."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "local"
    path: str = ".folio/store"


class RetrievalConfig(BaseModel):
    """How many passages are passed to the generator."""

    model_config = ConfigDict(extra="ignore")

    top_k: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)


class ServerConfig(BaseModel):
    """Bind address for ``folio serve``."""

    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class FileConfig(BaseModel):
    """Contents of ``folio.yaml``."""

    model_config = ConfigDict(extra="ignore")

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


class EnvSettings(BaseSettings):
    """Environment and ``.env`` overrides. Field names map to ``FOLIO_*``."""

    model_config = SettingsConfigDict(
        env_prefix="FOLIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    ollama_base_url: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    vectorstore_provider: str | None = None
    vectorstore_path: str | None = None
    top_k: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    server_host: str | None = None
    server_port: int | None = None


class Settings(BaseModel):
    """Fully resolved configuration for one Folio process."""

    model_config = ConfigDict(frozen=True)

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrievalConfig
    server: ServerConfig
    openai_api_key: str | None = None
    openai_base_url: str = _DEFAULT_OPENAI_URL
    ollama_base_url: str = _DEFAULT_OLLAMA_URL


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Merge the YAML file with environment variables.

    Search order for the file, when ``config_path`` is omitted:

    1. ``FOLIO_CONFIG``
    2. ``./folio.yaml``
    3. ``./config.yaml``

    A missing default file is fine and means "use built-in defaults". An
    explicit path that does not exist is an error.
    """
    env = EnvSettings()
    path = _resolve_config_path(config_path, env)
    file_config = _read_file(path)
    data = file_config.model_dump()
    _overlay_env(data, env)
    return Settings(
        chunking=ChunkingConfig.model_validate(data["chunking"]),
        embedding=EmbeddingConfig.model_validate(data["embedding"]),
        llm=LLMConfig.model_validate(data["llm"]),
        vectorstore=VectorStoreConfig.model_validate(data["vectorstore"]),
        retrieval=RetrievalConfig.model_validate(data["retrieval"]),
        server=ServerConfig.model_validate(data["server"]),
        openai_api_key=_blank_to_none(env.openai_api_key),
        openai_base_url=(env.openai_base_url or _DEFAULT_OPENAI_URL).rstrip("/"),
        ollama_base_url=(env.ollama_base_url or _DEFAULT_OLLAMA_URL).rstrip("/"),
    )


def _resolve_config_path(config_path: Path | str | None, env: EnvSettings) -> Path | None:
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path
    if env.config:
        path = Path(env.config)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path
    for candidate in (Path("folio.yaml"), Path("config.yaml")):
        if candidate.is_file():
            return candidate
    return None


def _read_file(path: Path | None) -> FileConfig:
    if path is None:
        return FileConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if raw is None:
        return FileConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} must be a mapping.")
    try:
        return FileConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _overlay_env(data: dict[str, Any], env: EnvSettings) -> None:
    if env.embedding_provider:
        data["embedding"]["provider"] = env.embedding_provider
    if env.embedding_model:
        data["embedding"]["model"] = env.embedding_model
    if env.llm_provider:
        data["llm"]["provider"] = env.llm_provider
    if env.llm_model:
        data["llm"]["model"] = env.llm_model
    if env.llm_temperature is not None:
        data["llm"]["temperature"] = env.llm_temperature
    if env.vectorstore_provider:
        data["vectorstore"]["provider"] = env.vectorstore_provider
    if env.vectorstore_path:
        data["vectorstore"]["path"] = env.vectorstore_path
    if env.top_k is not None:
        data["retrieval"]["top_k"] = env.top_k
    if env.chunk_size is not None:
        data["chunking"]["size"] = env.chunk_size
    if env.chunk_overlap is not None:
        data["chunking"]["overlap"] = env.chunk_overlap
    if env.server_host:
        data["server"]["host"] = env.server_host
    if env.server_port is not None:
        data["server"]["port"] = env.server_port


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()
