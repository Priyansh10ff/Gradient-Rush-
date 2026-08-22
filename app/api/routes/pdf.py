"""PDF ingestion endpoint with page-wise KnowledgeNode indexing."""

from anyio import to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes._upload import acknowledge_upload
from app.schemas.knowledge import KnowledgeUploadResponse, MediaModality
from app.services.pdf_processor import PdfProcessingError, process_pdf
from app.services.storage import persist_upload
from app.services.vector_store import VectorStoreError, get_knowledge_vector_store


router = APIRouter(prefix="/upload", tags=["uploads"])


@router.post("/pdf", response_model=KnowledgeUploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    """Extract page text and visual artifacts from a PDF, then index every page."""
    receipt = acknowledge_upload(file, MediaModality.PDF)
    source_path = await persist_upload(file, receipt.source_asset.source_id)
    try:
        nodes = await to_thread.run_sync(process_pdf, str(source_path))
        await to_thread.run_sync(lambda: get_knowledge_vector_store().add_nodes(nodes))
    except PdfProcessingError as exc:
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
