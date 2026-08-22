"""KnowledgeNode semantic retrieval endpoint."""

import json
import logging
import re
from typing import Any

from anyio import to_thread
from fastapi import APIRouter, HTTPException, status

from app.schemas.knowledge import (
    KnowledgeQueryRequest,
    KnowledgeComparisonResponse,
    KnowledgeQueryResponse,
    KnowledgeQueryResult,
)
from app.services.vector_store import VectorStoreError, get_knowledge_vector_store

_log = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])

# Matches JSON-serialized TemporalLocation dicts like
# '{"end_seconds": 10.0, "page_number": null, "start_seconds": 5.0}'
_JSON_TIMESTAMP_RE = re.compile(r"^\{.*\}$", re.DOTALL)


def _fmt_seconds(seconds: float) -> str:
    mins, secs = divmod(max(0, int(seconds)), 60)
    return f"{mins:02d}:{secs:02d}"


def _sanitize_timestamp(raw: Any) -> str | None:
    """Return a clean human-readable timestamp string.

    Handles:
    - Already-clean strings like ``"01:15 - 01:20"`` or ``"Page 3"`` → pass through.
    - Raw JSON dict strings like ``'{"start_seconds": 5.0, "end_seconds": 10.0}'``
      → parse and reformat as ``"MM:SS - MM:SS"``.
    - ``TemporalLocation``-like objects → convert attributes.
    - ``None`` or empty → return ``None``.
    """
    if raw is None:
        return None

    # Already a clean string (not JSON)?
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        if _JSON_TIMESTAMP_RE.match(stripped):
            # Try to parse and reformat it.
            try:
                obj = json.loads(stripped)
                page = obj.get("page_number")
                if page is not None:
                    return f"Page {page}"
                start = obj.get("start_seconds")
                end = obj.get("end_seconds")
                if start is not None and end is not None:
                    return f"{_fmt_seconds(float(start))} - {_fmt_seconds(float(end))}"
                if start is not None:
                    return _fmt_seconds(float(start))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return stripped  # already clean

    # Numeric fallback
    if isinstance(raw, (int, float)):
        return _fmt_seconds(float(raw))

    return None


def _sanitize_frame_path(raw: Any) -> str | None:
    """Return a valid URL path string or ``None``.

    Guards against broken sentinel values that callers must never pass to
    ``st.image()``:  ``0``, ``"0"``, empty strings, and non-string types.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped or stripped == "0":
        return None
    return stripped


def _to_query_result(hit: dict[str, Any]) -> KnowledgeQueryResult:
    """Flatten Chroma metadata into the stable demo-facing response contract."""
    metadata = hit.get("metadata") or {}

    # Transcript: prefer the dedicated field; fall back to content so the UI
    # never shows "No transcript available" when text exists in content.
    transcript_raw = hit.get("transcript") or metadata.get("transcript")
    content_raw = metadata.get("content") or hit.get("document")
    transcript = transcript_raw or content_raw or None

    # Visual summary — only meaningful for visual modalities.
    visual_summary = metadata.get("visual_summary") or None

    # Timestamp: must always be a clean human-readable string.
    timestamp_raw = hit.get("timestamp") or metadata.get("timestamp")
    timestamp = _sanitize_timestamp(timestamp_raw)

    # Frame path: must be a valid URL string or None — never "0" or falsy garbage.
    frame_path_raw = hit.get("frame_path") or metadata.get("frame_path")
    frame_path = _sanitize_frame_path(frame_path_raw)

    return KnowledgeQueryResult(
        transcript=transcript,
        visual_summary=visual_summary,
        timestamp=timestamp,
        frame_path=frame_path,
        source=hit.get("source") or metadata.get("source"),
        modality=metadata.get("modality"),
        similarity_score=float(hit["similarity_score"]),
        distance=float(hit["distance"]),
    )


@router.post("/query", response_model=KnowledgeQueryResponse)
async def query_knowledge(request: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
    """Search the persistent Chroma KnowledgeNode collection."""
    try:
        hits = await to_thread.run_sync(
            lambda: get_knowledge_vector_store().search(request.query, request.limit)
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return KnowledgeQueryResponse(
        query=request.query,
        results=[_to_query_result(hit) for hit in hits],
    )


@router.post("/query/compare", response_model=KnowledgeComparisonResponse)
async def compare_query(request: KnowledgeQueryRequest) -> KnowledgeComparisonResponse:
    """Compare combined multimodal retrieval with transcript-only retrieval."""
    try:
        multimodal_hits, baseline_hits = await to_thread.run_sync(
            lambda: (
                get_knowledge_vector_store().search(request.query, request.limit),
                get_knowledge_vector_store().search_text_only(request.query, request.limit),
            )
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return KnowledgeComparisonResponse(
        query=request.query,
        multimodal_result=_to_query_result(multimodal_hits[0]) if multimodal_hits else None,
        text_only_baseline_result=_to_query_result(baseline_hits[0]) if baseline_hits else None,
    )
