"""Audio upload route — builds KnowledgeNodes with formatted timestamps."""

import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes._upload import acknowledge_upload
from app.schemas.knowledge import (
    KnowledgeNode,
    KnowledgeUploadResponse,
    MediaModality,
)
from app.services.audio_processor import AudioProcessingError, process_audio
from app.services.storage import persist_upload
from app.services.vector_store import VectorStoreError, get_knowledge_vector_store

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["uploads"])


def _fmt_time(seconds: float) -> str:
    """Format a non-negative second offset as MM:SS."""
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


@router.post("/audio", response_model=KnowledgeUploadResponse)
async def upload_audio(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    """Persist audio, transcribe with Groq/Whisper, and index timestamped KnowledgeNodes.

    Nodes are added to BOTH the multimodal ChromaDB collection and the
    text-only baseline collection so every search endpoint can find audio.
    Both ``content`` and ``transcript`` are populated from segment text so
    no consumer ever gets "No transcript available" for a segment with speech.
    """
    receipt = acknowledge_upload(file, MediaModality.AUDIO)
    source_path = await persist_upload(file, receipt.source_asset.source_id)

    try:
        segments = await asyncio.to_thread(process_audio, str(source_path))
    except AudioProcessingError as exc:
        _log.exception("Audio transcription failed for %s", receipt.source_asset.filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    nodes: list[KnowledgeNode] = []
    for segment in segments:
        text = str(segment["text"]).strip()
        if not text:
            continue
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        # Format timestamp as a clean human-readable string — never a raw
        # dict, JSON blob, or TemporalLocation object.
        timestamp_str = f"{_fmt_time(start)} - {_fmt_time(end)}"
        nodes.append(
            KnowledgeNode(
                # Both content AND transcript must be populated so the
                # text-only search collection receives this segment's text.
                content=text,
                transcript=text,
                modality=MediaModality.AUDIO,
                timestamp=timestamp_str,
                source=receipt.source_asset.filename,
                provenance={
                    "start_seconds": start,
                    "end_seconds": end,
                    "source_id": str(receipt.source_asset.source_id),
                },
            )
        )

    warnings: list[str] = []
    indexed_count = 0
    if nodes:
        try:
            await asyncio.to_thread(get_knowledge_vector_store().add_nodes, nodes)
            indexed_count = len(nodes)
        except VectorStoreError as exc:
            warnings.append(f"Local vector indexing was skipped: {exc}")

    return KnowledgeUploadResponse(
        success=True,
        processed_nodes=indexed_count,
        source=receipt.source_asset.filename,
    )
