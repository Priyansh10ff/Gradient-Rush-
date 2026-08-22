"""Persistent local Chroma retrieval over provenance-preserving knowledge records."""

import json
import threading
import asyncio
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.schemas.knowledge import (
    KnowledgeNode,
    ExtractedKnowledgeBase,
)
from app.services.storage import storage_root


class VectorStoreError(RuntimeError):
    """Base error for unavailable local indexing or retrieval capabilities."""


class VectorStoreDependencyError(VectorStoreError):
    """Raised when Chroma's local embedding dependencies are unavailable."""


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
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
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
            self.embedding_function = DefaultEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function,
            )
            self.text_only_collection = self.client.get_or_create_collection(
                name=self.text_only_collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function,
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

async def index_records(records: list[ExtractedKnowledgeBase]) -> int:
    """Index legacy extracted records through the local KnowledgeNode store."""
    nodes = [
        KnowledgeNode(
            content=record.content,
            modality=record.modality,
            timestamp=record.timestamp,
            source=record.source,
            provenance={
                "source_id": str(record.source_id),
                "parent_knowledge_id": (
                    str(record.parent_knowledge_id)
                    if record.parent_knowledge_id is not None
                    else None
                ),
            },
        )
        for record in records
    ]
    await asyncio.to_thread(get_knowledge_vector_store().add_nodes, nodes)
    return len(nodes)
