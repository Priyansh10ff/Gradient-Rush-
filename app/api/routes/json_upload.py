"""Structured JSON ingestion endpoint (single record or array of records)."""

from anyio import to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes._upload import acknowledge_upload
from app.schemas.knowledge import KnowledgeUploadResponse, MediaModality
from app.services.json_processor import JsonProcessingError, process_json
from app.services.storage import persist_upload
from app.services.vector_store import VectorStoreError, get_knowledge_vector_store


router = APIRouter(prefix="/upload", tags=["uploads"])


@router.post("/json", response_model=KnowledgeUploadResponse)
async def upload_json(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    """Index a .json file (object/array of records) or a plain .txt note as KnowledgeNodes."""
    receipt = acknowledge_upload(file, MediaModality.JSON)
    source_path = await persist_upload(file, receipt.source_asset.source_id)
    try:
        nodes = await to_thread.run_sync(process_json, str(source_path))
        await to_thread.run_sync(lambda: get_knowledge_vector_store().add_nodes(nodes))
    except JsonProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return KnowledgeUploadResponse(
        success=True, processed_nodes=len(nodes), source=receipt.source_asset.filename
    )
