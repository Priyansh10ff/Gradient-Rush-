"""Persistent local Chroma retrieval over provenance-preserving knowledge records."""

import asyncio
import hashlib
import json
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    KnowledgeNode,
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


class VectorStore:
    """Persistent ChromaDB storage for :class:`KnowledgeNode` instances.

    Chroma's default embedding function creates embeddings from the assembled
    document.  Node metadata is normalized to Chroma's scalar-only metadata
    format, with nested values JSON encoded so source provenance is retained.
    """

    def __init__(
        self,
        persistence_path: str | Path = "./chroma_db",
        collection_name: str = "multimodal_knowledge",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreDependencyError(
                "chromadb must be installed to use VectorStore."
            ) from exc

        self.persistence_path = Path(persistence_path)
        self.collection_name = collection_name
        self.text_only_collection_name = f"{collection_name}_text_only"
        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persistence_path))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.text_only_collection = self.client.get_or_create_collection(
                name=self.text_only_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError("Unable to initialize the ChromaDB collection.") from exc

    @staticmethod
    def _json_value(value: Any) -> str:
        """Serialize nested Pydantic/UUID values consistently for metadata."""
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude_none=True)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def _metadata_for_node(cls, node: KnowledgeNode) -> dict[str, str | int | float | bool]:
        """Convert all supplied node attributes to Chroma-compatible metadata."""
        raw = node.model_dump(mode="python", exclude_none=True)
        metadata: dict[str, str | int | float | bool] = {}
        for key, value in raw.items():
            if key == "id" or value is None:
                continue
            if isinstance(value, Enum):
                metadata[key] = str(value.value)
            elif isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = cls._json_value(value)
        # Include the assembled text fields in metadata so callers can display
        # transcripts and visual summaries without parsing the document.
        return metadata

    @staticmethod
    def _content_text(value: str | dict[str, Any] | None) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    @classmethod
    def _document_for_node(cls, node: KnowledgeNode) -> str:
        """Build the embedding document from text and optional visual context."""
        parts = (
            cls._content_text(node.content).strip(),
            (node.transcript or "").strip(),
            (node.visual_summary or "").strip(),
        )
        document = "\n\n".join(part for part in parts if part)
        # Chroma accepts an empty document, but a stable non-empty fallback
        # produces clearer embeddings for metadata-only image/frame nodes.
        return document or "Multimodal knowledge item"

    def add_nodes(self, nodes: list[KnowledgeNode]) -> None:
        """Upsert nodes, embedding their text while preserving all metadata.

        ``upsert`` makes repeated ingestion of a caller-provided ``node.id``
        idempotent; absent IDs are generated with UUID4.
        """
        if not nodes:
            return
        try:
            ids = [str(node.id) if node.id is not None else str(uuid4()) for node in nodes]
            documents = [self._document_for_node(node) for node in nodes]
            metadatas = [self._metadata_for_node(node) for node in nodes]
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            transcript_entries = [
                (node_id, node.transcript.strip(), metadata)
                for node_id, node, metadata in zip(ids, nodes, metadatas, strict=True)
                if node.transcript and node.transcript.strip()
            ]
            if transcript_entries:
                self.text_only_collection.upsert(
                    ids=[entry[0] for entry in transcript_entries],
                    documents=[entry[1] for entry in transcript_entries],
                    metadatas=[entry[2] for entry in transcript_entries],
                )
        except Exception as exc:
            raise VectorStoreError("Unable to add nodes to the ChromaDB collection.") from exc

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return nearest knowledge nodes with provenance and similarity scores."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        try:
            available = self.collection.count()
            if available == 0:
                return []
            results = self.collection.query(
                query_texts=[normalized_query],
                n_results=min(limit, available),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to search the ChromaDB collection.") from exc

        hits: list[dict[str, Any]] = []
        for node_id, document, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            node_metadata = metadata or {}
            numeric_distance = float(distance)
            hits.append(
                {
                    "id": str(node_id),
                    "document": document,
                    "metadata": node_metadata,
                    "timestamp": node_metadata.get("timestamp"),
                    "frame_path": node_metadata.get("frame_path"),
                    "transcript": node_metadata.get("transcript"),
                    "source": node_metadata.get("source"),
                    "distance": numeric_distance,
                    "similarity_score": max(0.0, 1.0 - numeric_distance),
                }
            )
        return hits

    def search_text_only(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search only embeddings made from raw transcripts, never visual summaries."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        try:
            available = self.text_only_collection.count()
            if available == 0:
                return []
            results = self.text_only_collection.query(
                query_texts=[normalized_query],
                n_results=min(limit, available),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to perform text-only ChromaDB search.") from exc

        hits: list[dict[str, Any]] = []
        for node_id, transcript, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            node_metadata = metadata or {}
            numeric_distance = float(distance)
            hits.append(
                {
                    "id": str(node_id),
                    "document": transcript,
                    "metadata": node_metadata,
                    "timestamp": node_metadata.get("timestamp"),
                    "frame_path": node_metadata.get("frame_path"),
                    "transcript": transcript,
                    "source": node_metadata.get("source"),
                    "distance": numeric_distance,
                    "similarity_score": max(0.0, 1.0 - numeric_distance),
                }
            )
        return hits


_knowledge_store_instance: VectorStore | None = None
_knowledge_store_lock = threading.Lock()


def get_knowledge_vector_store() -> VectorStore:
    """Return the process-wide Chroma store used for ``KnowledgeNode`` objects."""
    global _knowledge_store_instance
    with _knowledge_store_lock:
        if _knowledge_store_instance is None:
            _knowledge_store_instance = VectorStore()
        return _knowledge_store_instance


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
