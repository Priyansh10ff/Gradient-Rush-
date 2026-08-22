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
        """Build the embedding document from text and optional visual context.

        Short segments (e.g. 2–8 word audio lyrics) produce noisy embeddings
        that score similarly against every query.  We enrich them by prepending
        a source-context prefix so the embedding has enough signal to
        discriminate between topics:

            [audio | Edd_Sheeran.mp3 | 03:32 - 03:34]
            Come on be my baby

        This pattern dramatically improves retrieval precision for short clips
        while not hurting longer PDF/image documents where the body text is
        already sufficient.
        """
        transcript = (node.transcript or "").strip()
        content = cls._content_text(node.content).strip()
        visual = (node.visual_summary or "").strip()

        # Build a context prefix that anchors short segments in embedding space.
        modality_str = (
            node.modality.value
            if hasattr(node.modality, "value")
            else str(node.modality)
        )
        source_str = (node.source or "").strip()
        timestamp_str = str(node.timestamp or "").strip()
        prefix_parts = [p for p in [modality_str, source_str, timestamp_str] if p]
        prefix = "[" + " | ".join(prefix_parts) + "]" if prefix_parts else ""

        # Primary text: prefer transcript (speech/OCR), fall back to content.
        primary_text = transcript or content

        body_parts = [p for p in [prefix, primary_text, visual] if p]
        document = "\n".join(body_parts)
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

    # Minimum cosine-similarity score a result must exceed to be returned.
    # Results at or below this threshold are statistically indistinguishable
    # from random noise for MiniLM-L6 embeddings and should be suppressed.
    _MIN_SCORE = 0.30

    # Maximum number of results allowed from a single source file.  Prevents
    # many short segments from one audio/video file flooding the top-k list.
    _MAX_PER_SOURCE = 2

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float | None = None,
        max_per_source: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest knowledge nodes with provenance and similarity scores.

        Results below ``min_score`` are filtered out so the caller never
        receives irrelevant noise hits.  Source-level deduplication
        (``max_per_source``) prevents one audio file's many short segments
        from dominating the entire result list.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        effective_min_score = self._MIN_SCORE if min_score is None else min_score
        effective_max_per_source = (
            self._MAX_PER_SOURCE if max_per_source is None else max_per_source
        )

        try:
            available = self.collection.count()
            if available == 0:
                return []
            # Fetch more candidates than requested so deduplication and
            # threshold filtering still leave at least ``limit`` survivors.
            fetch_n = min(limit * 4, available)
            results = self.collection.query(
                query_texts=[normalized_query],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to search the ChromaDB collection.") from exc

        hits: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        for node_id, document, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            node_metadata = metadata or {}
            numeric_distance = float(distance)
            score = max(0.0, 1.0 - numeric_distance)

            # --- Relevance threshold ---
            if score < effective_min_score:
                continue

            # --- Source-level deduplication ---
            source_key = str(node_metadata.get("source") or "unknown")
            if source_counts.get(source_key, 0) >= effective_max_per_source:
                continue
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

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
                    "similarity_score": score,
                }
            )
            if len(hits) >= limit:
                break

        return hits

    def search_text_only(
        self,
        query: str,
        limit: int = 5,
        min_score: float | None = None,
        max_per_source: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search only embeddings made from raw transcripts, never visual summaries.

        Applies the same relevance threshold and source-level deduplication
        as ``search()`` so the baseline result is also meaningful.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        effective_min_score = self._MIN_SCORE if min_score is None else min_score
        effective_max_per_source = (
            self._MAX_PER_SOURCE if max_per_source is None else max_per_source
        )

        try:
            available = self.text_only_collection.count()
            if available == 0:
                return []
            fetch_n = min(limit * 4, available)
            results = self.text_only_collection.query(
                query_texts=[normalized_query],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to perform text-only ChromaDB search.") from exc

        hits: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        for node_id, transcript, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            node_metadata = metadata or {}
            numeric_distance = float(distance)
            score = max(0.0, 1.0 - numeric_distance)

            if score < effective_min_score:
                continue

            source_key = str(node_metadata.get("source") or "unknown")
            if source_counts.get(source_key, 0) >= effective_max_per_source:
                continue
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

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
                    "similarity_score": score,
                }
            )
            if len(hits) >= limit:
                break

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
    """Index legacy extracted records through the local KnowledgeNode store.

    Converts ``TemporalLocation`` timestamps to clean human-readable strings
    (e.g. ``"01:15 - 01:30"`` or ``"Page 3"``) so the UI never receives raw
    JSON dict objects in the timestamp field.  Sets ``transcript`` from
    ``content`` so every record lands in both the multimodal and text-only
    collections.
    """

    def _fmt_seconds(seconds: float) -> str:
        mins, secs = divmod(max(0, int(seconds)), 60)
        return f"{mins:02d}:{secs:02d}"

    def _format_timestamp(loc: object) -> str | None:
        if loc is None:
            return None
        # TemporalLocation pydantic model
        page = getattr(loc, "page_number", None)
        if page is not None:
            return f"Page {page}"
        start = getattr(loc, "start_seconds", None)
        end = getattr(loc, "end_seconds", None)
        if start is not None and end is not None:
            return f"{_fmt_seconds(float(start))} - {_fmt_seconds(float(end))}"
        if start is not None:
            return _fmt_seconds(float(start))
        return None

    nodes = []
    for record in records:
        content_text = str(record.content) if record.content is not None else ""
        nodes.append(
            KnowledgeNode(
                content=content_text,
                # transcript must be set so the text-only search index
                # receives this segment's text (add_nodes only indexes
                # to text_only_collection when transcript is non-empty).
                transcript=content_text,
                modality=record.modality,
                timestamp=_format_timestamp(record.timestamp),
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
        )
    await asyncio.to_thread(get_knowledge_vector_store().add_nodes, nodes)
    return len(nodes)

