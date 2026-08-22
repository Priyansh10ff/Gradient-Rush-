"""Schemas that retain provenance and hierarchy for extracted multimodal data."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MediaModality(str, Enum):
    """Supported source media categories."""

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"


class TemporalLocation(BaseModel):
    """Where a datum occurs inside its original source.

    Video and audio normally use ``start_seconds``/``end_seconds``.  PDFs may
    use ``page_number``; images can leave this value as ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_time_range(self) -> "TemporalLocation":
        """Prevent invalid media ranges from reaching the knowledge store."""
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return self


class SourceAsset(BaseModel):
    """The uploaded parent asset to which all derived knowledge is linked."""

    model_config = ConfigDict(extra="forbid")

    source_id: UUID = Field(default_factory=uuid4)
    filename: str = Field(min_length=1)
    modality: MediaModality
    content_type: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractedKnowledgeBase(BaseModel):
    """Mandatory, provenance-preserving representation of extracted content.

    ``timestamp`` is intentionally present for every item but may be ``None``
    when a modality has no temporal location (for example, a whole image).
    ``source`` keeps the human-readable original filename or external ID.
    """

    model_config = ConfigDict(extra="forbid")

    content: str | dict[str, Any] = Field(
        ..., description="Extracted text or structured data from the source."
    )
    modality: MediaModality
    timestamp: TemporalLocation | None = Field(
        ...,
        description="Temporal or page location in the original source, if applicable.",
    )
    source: str = Field(
        ..., min_length=1, description="Original file name or source identifier."
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_id: UUID = Field(
        ..., description="Stable identifier of the parent SourceAsset."
    )
    parent_knowledge_id: UUID | None = Field(
        default=None, description="Containing extracted node, if this is a child segment."
    )


class ExtractedKnowledge(ExtractedKnowledgeBase):
    """An extracted node with explicit links to its source and parent node.

    These IDs mean transcript snippets, OCR blocks, captions, and PDF sections
    remain connected to both their original asset and any containing segment.
    """

    knowledge_id: UUID = Field(default_factory=uuid4)
    segment_index: int | None = Field(default=None, ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class UploadAccepted(BaseModel):
    """Acknowledgement returned now; a worker can process it in a later phase."""

    model_config = ConfigDict(extra="forbid")

    upload_id: UUID
    status: Literal["accepted"] = "accepted"
    source_asset: SourceAsset
    extraction_started: bool = False


class UploadProcessingResult(BaseModel):
    """Synchronous processing response containing provenance-linked output."""

    model_config = ConfigDict(extra="forbid")

    upload_id: UUID
    status: Literal["completed", "partial"] = "completed"
    source_asset: SourceAsset
    extracted_knowledge: list[ExtractedKnowledgeBase] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    indexed_count: int = Field(default=0, ge=0)


class RetrievalHit(BaseModel):
    """A semantically ranked knowledge record returned by the local index."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    score: float = Field(ge=0.0, le=1.0)
    record: ExtractedKnowledgeBase


class QueryRequest(BaseModel):
    """Input accepted by the cross-modal retrieval endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    """Cross-modal semantic search output with source provenance."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RetrievalHit] = Field(default_factory=list)
