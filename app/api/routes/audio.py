"""Audio upload route."""

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes._upload import acknowledge_upload
from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    MediaModality,
    TemporalLocation,
    UploadProcessingResult,
)
from app.services.audio_processor import AudioProcessingError, process_audio
from app.services.storage import persist_upload
from app.services.vector_store import VectorStoreError, index_records


router = APIRouter(prefix="/upload", tags=["uploads"])


@router.post("/audio", response_model=UploadProcessingResult)
async def upload_audio(file: UploadFile = File(...)) -> UploadProcessingResult:
    """Persist audio and asynchronously produce timestamped transcript knowledge."""
    receipt = acknowledge_upload(file, MediaModality.AUDIO)
    source_path = await persist_upload(file, receipt.source_asset.source_id)
    try:
        segments = await asyncio.to_thread(process_audio, str(source_path))
        extracted_knowledge = [
            ExtractedKnowledgeBase(
                content=str(segment["text"]),
                modality=MediaModality.AUDIO,
                timestamp=TemporalLocation(
                    start_seconds=float(segment["start_time"]),
                    end_seconds=float(segment["end_time"]),
                ),
                source=receipt.source_asset.filename,
                confidence=0.5,
                source_id=receipt.source_asset.source_id,
            )
            for segment in segments
        ]
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    warnings: list[str] = []
    try:
        indexed_count = await index_records(extracted_knowledge)
    except VectorStoreError as exc:
        indexed_count = 0
        warnings.append(f"Local vector indexing was skipped: {exc}")

    return UploadProcessingResult(
        upload_id=receipt.upload_id,
        status="partial" if warnings else "completed",
        source_asset=receipt.source_asset,
        extracted_knowledge=extracted_knowledge,
        warnings=warnings,
        indexed_count=indexed_count,
    )
