"""Cross-modal semantic retrieval endpoint."""

import asyncio

from fastapi import APIRouter, HTTPException, status

from app.schemas.knowledge import QueryRequest, QueryResponse
from app.services.vector_store import VectorStoreError, get_vector_store


router = APIRouter(tags=["retrieval"])


@router.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest) -> QueryResponse:
    """Search all locally indexed modalities and retain provenance in every hit."""
    try:
        results = await asyncio.to_thread(
            get_vector_store().search,
            request.query,
            request.top_k,
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return QueryResponse(query=request.query, results=results)
