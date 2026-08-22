"""Persistent local Chroma retrieval over provenance-preserving knowledge records."""

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    RetrievalHit,
    TemporalLocation,
)
from app.services.storage import storage_root


class VectorStoreError(RuntimeError):
    """Base error for unavailable local indexing or retrieval capabilities."""


class VectorStoreDependencyError(VectorStoreError):
    """Raised when Chroma or sentence-transformers has not been installed."""


class LocalModelUnavailableError(VectorStoreError):
    """Raised when the configured local embedding model is not available."""


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class LocalVectorStore:
    """A local persistent Chroma collection backed by a sentence-transformer model.

    Embeddings are calculated explicitly before calling Chroma. This keeps the
    embedding backend local and avoids depending on Chroma's provider-specific
    embedding-function interfaces.
    """

    def __init__(
        self,
        *,
        persistence_path: Path | None = None,
        collection_name: str = "multimodal_knowledge",
        model_name: str = "all-MiniLM-L6-v2",
        model_directory: Path | None = None,
        local_models_only: bool | None = None,
    ) -> None:
        self.persistence_path = persistence_path or (storage_root() / "chroma")
        self.collection_name = collection_name
        self.model_name = model_name
        self.model_directory = model_directory or (
            storage_root() / "models" / "sentence_transformers"
        )
        self.local_models_only = (
            _env_flag("LOCAL_MODELS_ONLY", default=True)
            if local_models_only is None
            else local_models_only
        )
        self._client: Any | None = None
        self._collection: Any | None = None
        self._model: Any | None = None
        self._lock = threading.RLock()

    def _ensure_initialized(self) -> None:
        if self._collection is not None and self._model is not None:
            return

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise VectorStoreDependencyError(
                "chromadb and sentence-transformers must be installed for local retrieval."
            ) from exc

        self.model_directory.mkdir(parents=True, exist_ok=True)
        try:
            model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.model_directory),
                device="cpu",
                local_files_only=self.local_models_only,
            )
        except Exception as exc:  # The underlying Hugging Face errors vary by version.
            model_hint = (
                "The local embedding model is unavailable. Cache it first, or set "
                "LOCAL_MODELS_ONLY=false once to allow its free initial download."
            )
            raise LocalModelUnavailableError(model_hint) from exc

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persistence_path))
            collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError("Unable to open the local Chroma collection.") from exc

        self._client = client
        self._collection = collection
        self._model = model

    def _embed(self, documents: list[str]) -> list[list[float]]:
        if self._model is None:
            raise VectorStoreError("The embedding model has not been initialized.")
        try:
            vectors = self._model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as exc:
            raise VectorStoreError("Local embedding generation failed.") from exc

    @staticmethod
    def _document_for(record: ExtractedKnowledgeBase) -> str:
        if isinstance(record.content, str):
            return record.content
        return json.dumps(record.content, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _metadata_for(record: ExtractedKnowledgeBase) -> dict[str, str | int | float]:
        metadata: dict[str, str | int | float] = {
            "modality": record.modality.value,
            "source": record.source,
            "confidence": record.confidence,
            "source_id": str(record.source_id),
            "content_kind": "text" if isinstance(record.content, str) else "json",
        }
        if record.parent_knowledge_id is not None:
            metadata["parent_knowledge_id"] = str(record.parent_knowledge_id)
        if record.timestamp is not None:
            if record.timestamp.start_seconds is not None:
                metadata["timestamp_start"] = record.timestamp.start_seconds
            if record.timestamp.end_seconds is not None:
                metadata["timestamp_end"] = record.timestamp.end_seconds
            if record.timestamp.page_number is not None:
                metadata["page_number"] = record.timestamp.page_number
        return metadata

    @staticmethod
    def _record_id(record: ExtractedKnowledgeBase) -> str:
        payload = json.dumps(
            record.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ingest(self, records: list[ExtractedKnowledgeBase]) -> int:
        """Upsert extracted records and their provenance into the local collection."""
        if not records:
            return 0

        with self._lock:
            self._ensure_initialized()
            documents = [self._document_for(record) for record in records]
            self._collection.upsert(
                ids=[self._record_id(record) for record in records],
                documents=documents,
                metadatas=[self._metadata_for(record) for record in records],
                embeddings=self._embed(documents),
            )
        return len(records)

    @staticmethod
    def _record_from_result(
        document: str,
        metadata: dict[str, Any],
    ) -> ExtractedKnowledgeBase:
        if metadata.get("content_kind") == "json":
            try:
                content: str | dict[str, Any] = json.loads(document)
            except json.JSONDecodeError:
                content = document
        else:
            content = document

        timestamp_values: dict[str, int | float] = {}
        if "timestamp_start" in metadata:
            timestamp_values["start_seconds"] = float(metadata["timestamp_start"])
        if "timestamp_end" in metadata:
            timestamp_values["end_seconds"] = float(metadata["timestamp_end"])
        if "page_number" in metadata:
            timestamp_values["page_number"] = int(metadata["page_number"])

        parent_value = metadata.get("parent_knowledge_id")
        return ExtractedKnowledgeBase(
            content=content,
            modality=metadata["modality"],
            timestamp=TemporalLocation(**timestamp_values) if timestamp_values else None,
            source=str(metadata["source"]),
            confidence=float(metadata["confidence"]),
            source_id=UUID(str(metadata["source_id"])),
            parent_knowledge_id=UUID(str(parent_value)) if parent_value else None,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return the nearest source-linked records across every indexed modality."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        with self._lock:
            self._ensure_initialized()
            result_count = min(top_k, self._collection.count())
            if result_count == 0:
                return []
            raw_results = self._collection.query(
                query_embeddings=self._embed([normalized_query]),
                n_results=result_count,
                include=["documents", "metadatas", "distances"],
            )

        identifiers = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        hits: list[RetrievalHit] = []
        for record_id, document, metadata, distance in zip(
            identifiers, documents, metadatas, distances, strict=True
        ):
            if document is None or metadata is None:
                continue
            numeric_distance = max(0.0, float(distance))
            hits.append(
                RetrievalHit(
                    record_id=str(record_id),
                    score=1.0 / (1.0 + numeric_distance),
                    record=self._record_from_result(document, metadata),
                )
            )
        return hits


_store_instance: LocalVectorStore | None = None
_store_instance_lock = threading.Lock()


def get_vector_store() -> LocalVectorStore:
    """Return the process-wide local persistent index without eager model loading."""
    global _store_instance
    with _store_instance_lock:
        if _store_instance is None:
            _store_instance = LocalVectorStore()
        return _store_instance


async def index_records(records: list[ExtractedKnowledgeBase]) -> int:
    """Offload synchronous local embedding and Chroma writes from the API loop."""
    return await asyncio.to_thread(get_vector_store().ingest, records)
