"""KnowledgeNode semantic retrieval endpoint."""

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


router = APIRouter(tags=["retrieval"])


def _to_query_result(hit: dict[str, Any]) -> KnowledgeQueryResult:
    """Flatten Chroma metadata into the stable demo-facing response contract."""
    metadata = hit.get("metadata") or {}
    return KnowledgeQueryResult(
        transcript=hit.get("transcript") or metadata.get("transcript"),
        visual_summary=metadata.get("visual_summary"),
        timestamp=hit.get("timestamp") or metadata.get("timestamp"),
        frame_path=hit.get("frame_path") or metadata.get("frame_path"),
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
