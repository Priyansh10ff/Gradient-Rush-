"""Standalone PNG/JPEG ingestion endpoint."""

from pathlib import Path

from anyio import to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes._upload import acknowledge_upload
from app.schemas.knowledge import KnowledgeUploadResponse, MediaModality
from app.services.image_processor import ImageProcessingError, process_image
from app.services.storage import persist_upload
from app.services.vector_store import VectorStoreError, get_knowledge_vector_store


router = APIRouter(prefix="/upload", tags=["uploads"])


@router.post("/image", response_model=KnowledgeUploadResponse)
async def upload_image(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    """Analyze a PNG/JPEG image with vision and persist its KnowledgeNode."""
    if Path(file.filename or "").suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PNG and JPEG image uploads are supported.",
        )
    receipt = acknowledge_upload(file, MediaModality.IMAGE)
    source_path = await persist_upload(file, receipt.source_asset.source_id)
    try:
        node = await to_thread.run_sync(process_image, str(source_path))
        await to_thread.run_sync(lambda: get_knowledge_vector_store().add_nodes([node]))
    except ImageProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return KnowledgeUploadResponse(success=True, processed_nodes=1, source=node.source or "")
