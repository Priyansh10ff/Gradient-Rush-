"""Asynchronous FFmpeg-based video extraction and timeline-aware mapping."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    MediaModality,
    SourceAsset,
    TemporalLocation,
)
from app.services.audio_processor import ASRService, AudioProcessingError, process_audio


class MediaProcessingError(RuntimeError):
    """Raised when FFmpeg cannot produce a requested video derivative."""


class FfmpegUnavailableError(MediaProcessingError):
    """Raised when the configured FFmpeg executable cannot be started."""


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A sampled frame and its approximate position on the source timeline."""

    path: Path
    index: int
    timestamp_seconds: float


@dataclass(frozen=True, slots=True)
class VideoProcessingResult:
    """Artifacts and knowledge derived from one uploaded video source."""

    audio_path: Path | None
    frames: list[VideoFrame]
    frame_knowledge: list[ExtractedKnowledgeBase]
    transcript_knowledge: list[ExtractedKnowledgeBase]
    warnings: list[str]

    @property
    def extracted_knowledge(self) -> list[ExtractedKnowledgeBase]:
        return [*self.frame_knowledge, *self.transcript_knowledge]


async def _run_ffmpeg(
    *arguments: str,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 300.0,
) -> None:
    """Run FFmpeg without blocking FastAPI's event loop or invoking a shell."""
    try:
        process = await asyncio.create_subprocess_exec(
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            *arguments,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise FfmpegUnavailableError(
            "FFmpeg is required for video processing but was not found on PATH."
        ) from exc

    communicate_task = asyncio.create_task(process.communicate())
    try:
        _, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=timeout_seconds
        )
    except TimeoutError as exc:
        with suppress(ProcessLookupError):
            process.kill()
        await communicate_task
        raise MediaProcessingError("FFmpeg exceeded the processing time limit.") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise MediaProcessingError(detail or "FFmpeg failed to process the video.")


async def extract_audio_track(
    video_path: Path,
    output_path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 300.0,
) -> Path:
    """Extract the first video audio stream as 16 kHz mono WAV for ASR."""
    if not video_path.is_file():
        raise MediaProcessingError(f"Video source does not exist: {video_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    await _run_ffmpeg(
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MediaProcessingError("The video has no extractable audio track.")
    return output_path


async def sample_video_frames(
    video_path: Path,
    output_directory: Path,
    *,
    interval_seconds: float = 1.0,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 300.0,
) -> list[VideoFrame]:
    """Sample JPEG frames at a fixed interval, beginning at the video start."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")
    if not video_path.is_file():
        raise MediaProcessingError(f"Video source does not exist: {video_path.name}")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_pattern = output_directory / "frame_%06d.jpg"
    await _run_ffmpeg(
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds:g}",
        "-q:v",
        "2",
        str(output_pattern),
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )

    frame_paths = sorted(output_directory.glob("frame_*.jpg"))
    if not frame_paths:
        raise MediaProcessingError("FFmpeg did not produce any video frames.")
    return [
        VideoFrame(
            path=frame_path,
            index=index,
            timestamp_seconds=index * interval_seconds,
        )
        for index, frame_path in enumerate(frame_paths)
    ]


def map_video_frames(
    frames: list[VideoFrame],
    *,
    source_asset: SourceAsset,
    artifact_root: Path,
    parent_knowledge_id: UUID | None = None,
) -> list[ExtractedKnowledgeBase]:
    """Represent every sampled frame as a video-timeline knowledge item."""
    root = artifact_root.resolve()
    knowledge: list[ExtractedKnowledgeBase] = []
    for frame in frames:
        try:
            artifact_key = frame.path.resolve().relative_to(root).as_posix()
        except ValueError:
            artifact_key = frame.path.name
        knowledge.append(
            ExtractedKnowledgeBase(
                content={
                    "kind": "video_frame",
                    "artifact": artifact_key,
                    "frame_index": frame.index,
                },
                modality=MediaModality.VIDEO,
                timestamp=TemporalLocation(
                    start_seconds=frame.timestamp_seconds,
                    end_seconds=frame.timestamp_seconds,
                ),
                source=source_asset.filename,
                confidence=1.0,
                source_id=source_asset.source_id,
                parent_knowledge_id=parent_knowledge_id,
            )
        )
    return knowledge


async def process_video(
    video_path: Path,
    source_asset: SourceAsset,
    artifact_directory: Path,
    *,
    frame_interval_seconds: float = 1.0,
    asr_service: ASRService | None = None,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 300.0,
) -> VideoProcessingResult:
    """Extract video audio and frames, then retain a single source relationship.

    Transcript chunks deliberately use ``MediaModality.VIDEO`` because their
    parent source is the uploaded video, even though their bytes came from an
    intermediate audio artifact.
    """
    frames = await sample_video_frames(
        video_path,
        artifact_directory / "frames",
        interval_seconds=frame_interval_seconds,
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )
    frame_knowledge = map_video_frames(
        frames,
        source_asset=source_asset,
        artifact_root=artifact_directory,
    )
    audio_path: Path | None = None
    transcript_knowledge: list[ExtractedKnowledgeBase] = []
    warnings: list[str] = []
    try:
        audio_path = await extract_audio_track(
            video_path,
            artifact_directory / "audio.wav",
            ffmpeg_binary=ffmpeg_binary,
            timeout_seconds=timeout_seconds,
        )
        transcript_knowledge = await process_audio(
            audio_path,
            source_asset=source_asset,
            asr_service=asr_service,
            modality=MediaModality.VIDEO,
        )
    except (MediaProcessingError, AudioProcessingError) as exc:
        # Frame extraction already succeeded. Keep that useful source-linked
        # output instead of discarding it for a silent video or ASR outage.
        if isinstance(exc, FfmpegUnavailableError):
            raise
        warnings.append(f"Audio/transcript extraction was skipped: {exc}")

    return VideoProcessingResult(
        audio_path=audio_path,
        frames=frames,
        frame_knowledge=frame_knowledge,
        transcript_knowledge=transcript_knowledge,
        warnings=warnings,
    )
