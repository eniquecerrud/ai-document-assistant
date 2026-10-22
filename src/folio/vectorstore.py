"""Vector stores behind one small interface.

:class:`LocalVectorStore` is the default. It keeps a normalized NumPy matrix
on disk (``vectors.npz`` plus ``manifest.json``) and answers queries with
cosine similarity. :class:`ChromaVectorStore` is an optional alternative that
uses a persistent Chroma collection; install it with ``pip install "folio[chroma]"``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from folio.config import Settings
from folio.errors import FolioError, ProviderError
from folio.models import Chunk, DocumentRecord, ScoredChunk

_MANIFEST = "manifest.json"
_VECTORS = "vectors.npz"
_VERSION = 1


class VectorStore(Protocol):
    """Persistence and nearest-neighbour search for chunks."""

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert ``chunks``, replacing any previous chunks for ``document_id``."""

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks whose similarity is at least ``min_score``."""

    def list_documents(self) -> list[DocumentRecord]:
        """Return one record per indexed file, ordered by file name."""

    def delete_document(self, document_id: str) -> bool:
        """Remove a document. Return ``False`` when the id is unknown."""

    def count(self) -> int:
        """Return the number of stored chunks."""


class LocalVectorStore:
    """Cosine index stored as NumPy arrays. The whole matrix is loaded into memory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._dimension: int | None = None
        self._load()

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise FolioError("Each chunk needs exactly one embedding.")
        if any(chunk.document_id != document_id for chunk in chunks):
            raise FolioError("Refusing to store a chunk under a different document id.")
        matrix = _as_matrix(embeddings)
        with self._lock:
            remaining = self._dimension_after_drop(document_id)
            if matrix is not None and chunks and remaining not in (None, int(matrix.shape[1])):
                raise FolioError(
                    f"Embedding dimension {int(matrix.shape[1])} does not match the index "
                    f"({remaining}). Use the same embedding model or start a new store."
                )
            self._drop(document_id)
            if matrix is not None and chunks:
                self._dimension = int(matrix.shape[1])
                self._chunks.extend(chunks)
                self._vectors = (
                    matrix
                    if self._vectors is None or len(self._vectors) == 0
                    else np.vstack([self._vectors, matrix])
                )
            self._persist()

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._lock:
            if not self._chunks or self._vectors is None:
                return []
            query = np.asarray(embedding, dtype=np.float32)
            if query.ndim != 1 or query.shape[0] != self._dimension:
                got = int(query.shape[0]) if query.ndim == 1 else 0
                raise FolioError(
                    f"Query embedding dimension {got} does not match the index ({self._dimension})."
                )
            scores = _normalize_rows(self._vectors) @ _normalize(query)
            order = np.argsort(-scores)
            found: list[ScoredChunk] = []
            for index in order:
                score = float(scores[int(index)])
                if score < min_score:
                    break
                found.append(ScoredChunk(chunk=self._chunks[int(index)], score=score))
                if len(found) >= top_k:
                    break
            return found

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            counts: dict[str, DocumentRecord] = {}
            for chunk in self._chunks:
                current = counts.get(chunk.document_id)
                count = 1 if current is None else current.chunk_count + 1
                counts[chunk.document_id] = DocumentRecord(
                    id=chunk.document_id,
                    source=chunk.source,
                    chunk_count=count,
                )
            return sorted(counts.values(), key=lambda item: item.source.lower())

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            before = len(self._chunks)
            self._drop(document_id)
            removed = len(self._chunks) != before
            if removed:
                self._persist()
            return removed

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _dimension_after_drop(self, document_id: str) -> int | None:
        """Dimension that will remain once ``document_id`` is removed."""
        if any(chunk.document_id != document_id for chunk in self._chunks):
            return self._dimension
        return None

    def _drop(self, document_id: str) -> None:
        """Remove chunks for ``document_id``. Caller must hold the lock."""
        if not self._chunks:
            return
        keep = [
            index for index, chunk in enumerate(self._chunks) if chunk.document_id != document_id
        ]
        self._chunks = [self._chunks[index] for index in keep]
        if not keep:
            self._vectors = None
            self._dimension = None
            return
        if self._vectors is not None:
            self._vectors = self._vectors[keep]

    def _persist(self) -> None:
        manifest = {
            "version": _VERSION,
            "dimension": self._dimension,
            "chunks": [_chunk_dict(chunk) for chunk in self._chunks],
        }
        manifest_path = self.path / _MANIFEST
        vectors_path = self.path / _VECTORS
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        if self._vectors is None or len(self._chunks) == 0:
            if vectors_path.exists():
                vectors_path.unlink()
        else:
            blob = self.path / "vectors.tmp"
            array = np.asarray(self._vectors, dtype=np.float32)
            with blob.open("wb") as handle:
                np.savez_compressed(handle, embeddings=array)
            blob.replace(vectors_path)
        temporary.replace(manifest_path)

    def _load(self) -> None:
        manifest_path = self.path / _MANIFEST
        if not manifest_path.exists():
            return
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks = [Chunk(**item) for item in payload["chunks"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise FolioError(f"Could not read the vector store at {self.path}: {exc}") from exc
        if payload.get("version") != _VERSION:
            raise FolioError(
                f"Unsupported store version at {self.path}. Remove the directory and re-index."
            )
        vectors_path = self.path / _VECTORS
        vectors: np.ndarray | None = None
        if chunks:
            if not vectors_path.exists():
                raise FolioError(f"Vector file is missing for the store at {self.path}.")
            with np.load(vectors_path) as archive:
                vectors = np.asarray(archive["embeddings"], dtype=np.float32)
            if len(vectors) != len(chunks):
                raise FolioError(f"The store at {self.path} is inconsistent and was not loaded.")
        self._chunks = chunks
        self._vectors = vectors
        self._dimension = int(vectors.shape[1]) if vectors is not None and len(vectors) else None


class ChromaVectorStore:
    """Persistent Chroma collection. ``chromadb`` is imported on first use."""

    def __init__(self, path: Path | str, *, collection: str = "folio") -> None:
        self.path = Path(path).expanduser().resolve()
        self.collection_name = collection
        self._lock = threading.Lock()
        self._collection: object | None = None

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise FolioError("Each chunk needs exactly one embedding.")
        with self._lock:
            collection = self._collection_handle()
            self._delete_where(collection, document_id)
            if not chunks:
                return
            collection.upsert(  # type: ignore[attr-defined]
                ids=[chunk.id for chunk in chunks],
                embeddings=[list(map(float, vector)) for vector in embeddings],
                documents=[chunk.text for chunk in chunks],
                metadatas=[_chroma_meta(chunk) for chunk in chunks],
            )

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._lock:
            collection = self._collection_handle()
            total = int(collection.count())  # type: ignore[attr-defined]
            if total == 0:
                return []
            result = collection.query(  # type: ignore[attr-defined]
                query_embeddings=[list(map(float, embedding))],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
            return _from_chroma(result, min_score)

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            collection = self._collection_handle()
            raw = collection.get(include=["metadatas"])  # type: ignore[attr-defined]
            counts: dict[str, DocumentRecord] = {}
            for meta in raw.get("metadatas") or []:
                if not meta:
                    continue
                document_id = str(meta["document_id"])
                current = counts.get(document_id)
                count = 1 if current is None else current.chunk_count + 1
                counts[document_id] = DocumentRecord(
                    id=document_id,
                    source=str(meta["source"]),
                    chunk_count=count,
                )
            return sorted(counts.values(), key=lambda item: item.source.lower())

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            collection = self._collection_handle()
            before = int(collection.count())  # type: ignore[attr-defined]
            self._delete_where(collection, document_id)
            return int(collection.count()) != before  # type: ignore[attr-defined]

    def count(self) -> int:
        with self._lock:
            return int(self._collection_handle().count())  # type: ignore[attr-defined]

    def _collection_handle(self) -> object:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ProviderError(
                    'chromadb is not installed. Install it with: pip install "folio[chroma]"'
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.path))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @staticmethod
    def _delete_where(collection: object, document_id: str) -> None:
        existing = collection.get(where={"document_id": document_id})  # type: ignore[attr-defined]
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)  # type: ignore[attr-defined]


def build_store(settings: Settings) -> VectorStore:
    """Construct the vector store named in settings."""
    name = settings.vectorstore.provider.strip().lower()
    path = settings.vectorstore.path
    if name == "local":
        return LocalVectorStore(path)
    if name == "chroma":
        return ChromaVectorStore(path)
    raise ProviderError(
        f"Unknown vector store '{settings.vectorstore.provider}'. Use local or chroma."
    )


def _as_matrix(embeddings: Sequence[Sequence[float]]) -> np.ndarray | None:
    if not embeddings:
        return None
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise FolioError("Embeddings must be a non-empty 2-D matrix.")
    return matrix


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _chunk_dict(chunk: Chunk) -> dict[str, object]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "source": chunk.source,
        "page": chunk.page,
        "index": chunk.index,
        "text": chunk.text,
    }


def _chroma_meta(chunk: Chunk) -> dict[str, object]:
    return {
        "document_id": chunk.document_id,
        "source": chunk.source,
        "page": chunk.page,
        "index": chunk.index,
    }


def _from_chroma(result: dict[str, list[list[object]]], min_score: float) -> list[ScoredChunk]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    found: list[ScoredChunk] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances, strict=False):
        if not isinstance(meta, dict):
            continue
        # Chroma's cosine space reports distance as 1 - cosine similarity.
        score = 1.0 - float(distance)
        if score < min_score:
            continue
        chunk = Chunk(
            id=str(chunk_id),
            document_id=str(meta["document_id"]),
            source=str(meta["source"]),
            page=int(meta["page"]),
            index=int(meta["index"]),
            text=str(text),
        )
        found.append(ScoredChunk(chunk=chunk, score=score))
    return found
