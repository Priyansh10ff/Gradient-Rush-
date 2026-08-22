"""PDF upload route."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes._upload import acknowledge_upload
from app.schemas.knowledge import MediaModality, UploadProcessingResult
from app.services.pdf_processor import (
    PdfDependencyUnavailableError,
    PdfProcessingError,
    process_pdf,
)
from app.services.storage import derivative_directory, persist_upload
from app.services.vector_store import VectorStoreError, index_records


router = APIRouter(prefix="/upload", tags=["uploads"])


@router.post("/pdf", response_model=UploadProcessingResult)
async def upload_pdf(file: UploadFile = File(...)) -> UploadProcessingResult:
    """Persist a PDF, extract page-aware knowledge, and index its contents."""
    receipt = acknowledge_upload(file, MediaModality.PDF)
    source_path = await persist_upload(file, receipt.source_asset.source_id)
    try:
        result = await process_pdf(
            source_path,
            receipt.source_asset,
            derivative_directory(receipt.source_asset.source_id),
        )
    except PdfDependencyUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except PdfProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    warnings = list(result.warnings)
    try:
        indexed_count = await index_records(result.knowledge)
    except VectorStoreError as exc:
        indexed_count = 0
        warnings.append(f"Local vector indexing was skipped: {exc}")

    return UploadProcessingResult(
        upload_id=receipt.upload_id,
        status="partial" if warnings else "completed",
        source_asset=receipt.source_asset,
        extracted_knowledge=result.knowledge,
        warnings=warnings,
        indexed_count=indexed_count,
    )
