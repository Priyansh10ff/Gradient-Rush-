"""Video ingestion endpoint backed by OpenCV/Gemini processing and ChromaDB."""

from pathlib import Path

from anyio import open_file, to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.knowledge import VideoUploadResponse
from app.services.video_processor import MediaProcessingError, process_video
from app.services.vector_store import VectorStoreError, get_knowledge_vector_store


router = APIRouter(prefix="/upload", tags=["uploads"])

_UPLOAD_CHUNK_SIZE = 1024 * 1024


async def _save_video(file: UploadFile) -> Path:
    """Stream an MP4 into the local raw-media directory without loading it in RAM."""
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only MP4 video uploads are supported.",
        )

    raw_directory = Path.cwd() / "data" / "raw"
    destination = raw_directory / filename
    try:
        raw_directory.mkdir(parents=True, exist_ok=True)
        await file.seek(0)
        async with await open_file(destination, "wb") as output:
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                await output.write(chunk)
        await file.seek(0)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save the uploaded video.",
        ) from exc
    return destination


@router.post("/video", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    """Save an MP4, build timestamp-aligned knowledge nodes, and index them."""
    saved_path = await _save_video(file)
    try:
        nodes = await to_thread.run_sync(process_video, str(saved_path))
        # Keep Chroma's synchronous persistence work off the request event loop.
        await to_thread.run_sync(lambda: get_knowledge_vector_store().add_nodes(nodes))
    except MediaProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return VideoUploadResponse(
        success=True,
        processed_nodes=len(nodes),
        source=Path(file.filename or saved_path.name).name,
    )
