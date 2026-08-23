"""Shared upload validation and acknowledgement helpers.

This module intentionally does not persist files or run extraction.  It only
creates a source-asset record that later pipeline stages can use as their
stable parent reference.
"""

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.schemas.knowledge import MediaModality, SourceAsset, UploadAccepted


_EXTENSIONS: dict[MediaModality, set[str]] = {
    MediaModality.VIDEO: {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    MediaModality.AUDIO: {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"},
    MediaModality.IMAGE: {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tiff", ".bmp"},
    MediaModality.PDF: {".pdf"},
    MediaModality.JSON: {".json", ".txt"},
}


def _matches_modality(file: UploadFile, modality: MediaModality) -> bool:
    """Accept a known extension or an appropriate MIME type supplied by a client."""
    extension = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()

    if extension in _EXTENSIONS[modality]:
        return True
    if modality is MediaModality.PDF:
        return content_type == "application/pdf"
    if modality is MediaModality.JSON:
        return content_type in {"application/json", "text/json", "text/plain"}
    return content_type.startswith(f"{modality.value}/")


def acknowledge_upload(file: UploadFile, modality: MediaModality) -> UploadAccepted:
    """Validate metadata and create the source asset passed to processors."""
    filename = file.filename
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a filename.",
        )
    if not _matches_modality(file, modality):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a {modality.value} file.",
        )

    safe_filename = Path(filename).name
    if not safe_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a valid filename.",
        )

    source_asset = SourceAsset(
        source_id=uuid4(),
        filename=safe_filename,
        modality=modality,
        content_type=file.content_type,
    )
    return UploadAccepted(upload_id=uuid4(), source_asset=source_asset)
