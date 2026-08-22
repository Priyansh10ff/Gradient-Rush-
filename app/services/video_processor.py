"""OpenCV/Gemini processing for timestamp-aligned video knowledge nodes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services.audio_processor import AudioProcessingError, process_audio
from app.services.image_processor import ImageProcessingError, analyze_image

# Gemini calls are network-bound, so describing several sampled frames at
# once cuts ingestion latency roughly linearly with worker count instead of
# paying full round-trip latency once per frame.
_VISION_WORKERS = 4


class MediaProcessingError(RuntimeError):
    """Raised when a video cannot be decoded, sampled, or described."""


class FfmpegUnavailableError(MediaProcessingError):
    """Retained for backwards compatibility with the upload route."""


def _format_time(seconds: float) -> str:
    """Format a non-negative second offset as the requested ``MM:SS`` form."""
    minutes, seconds_part = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds_part:02d}"


def _frame_filename(timestamp_seconds: float) -> str:
    """Create a collision-free, timestamp-readable JPEG filename."""
    hours, remainder = divmod(max(0, int(timestamp_seconds)), 3600)
    minutes, seconds_part = divmod(remainder, 60)
    prefix = f"{hours:02d}_" if hours else ""
    return f"frame_{prefix}{minutes:02d}_{seconds_part:02d}.jpg"


def _describe_frame(frame_path: Path, client: Any | None) -> str:
    """Ask Gemini Flash for a compact, retrieval-oriented visual description.

    Returns a human-readable fallback string instead of raising so that a
    single Gemini failure does not abort the entire video ingestion pipeline.
    """
    try:
        summary = analyze_image(frame_path, client=client)["visual_summary"]
    except ImageProcessingError as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Gemini vision analysis failed for %s: %s", frame_path.name, exc
        )
        return "[Visual description unavailable]"
    if not summary:
        return "[Visual description unavailable]"
    return summary


def _transcript_in_window(
    segments: list[dict[str, float | str]], start_seconds: float, end_seconds: float
) -> str:
    """Join transcript segments that overlap a sampled frame's time window."""
    return " ".join(
        str(segment["text"])
        for segment in segments
        if float(segment["end_time"]) > start_seconds
        and float(segment["start_time"]) < end_seconds
    ).strip()


def process_video(
    file_path: str,
    frame_interval_seconds: int = 3,
    *,
    gemini_client: Any | None = None,
    groq_client: Any | None = None,
) -> list[KnowledgeNode]:
    """Sample a video and combine Gemini visual context with aligned Whisper text."""
    if frame_interval_seconds <= 0:
        raise ValueError("frame_interval_seconds must be greater than zero")
    video_path = Path(file_path)
    if not video_path.is_file():
        raise MediaProcessingError(f"Video source does not exist: {video_path}")
    try:
        import cv2
    except ImportError as exc:
        raise MediaProcessingError("OpenCV is required to process video files.") from exc

    try:
        transcript_segments = process_audio(str(video_path), client=groq_client)
    except AudioProcessingError as exc:
        raise MediaProcessingError("Unable to transcribe the video audio track.") from exc

    output_directory = Path.cwd() / "data" / "frames"
    output_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise MediaProcessingError(f"Unable to open video: {video_path.name}")

    # Pass 1: sample frames and compute their transcript windows. This is
    # fast, local, and sequential because OpenCV's capture cursor can only
    # move one frame at a time.
    sampled: list[dict[str, Any]] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps if fps > 0 else 0.0
        timestamp_seconds = 0.0
        while timestamp_seconds <= duration_seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            success, frame = capture.read()
            if not success:
                break
            frame_path = output_directory / _frame_filename(timestamp_seconds)
            if not cv2.imwrite(str(frame_path), frame):
                raise MediaProcessingError(f"Unable to save extracted frame: {frame_path.name}")
            window_end = min(timestamp_seconds + frame_interval_seconds, duration_seconds)
            transcript = _transcript_in_window(
                transcript_segments, timestamp_seconds, window_end
            )
            sampled.append(
                {
                    "frame_path": frame_path,
                    "timestamp_seconds": timestamp_seconds,
                    "window_end": window_end,
                    "transcript": transcript,
                }
            )
            timestamp_seconds += frame_interval_seconds
    finally:
        capture.release()

    # Pass 2: describe every sampled frame with Gemini concurrently instead
    # of paying one full round-trip per frame in sequence.
    with ThreadPoolExecutor(max_workers=_VISION_WORKERS) as executor:
        visual_summaries = list(
            executor.map(
                lambda item: _describe_frame(item["frame_path"], gemini_client), sampled
            )
        )

    nodes: list[KnowledgeNode] = []
    for item, visual_summary in zip(sampled, visual_summaries, strict=True):
        frame_path: Path = item["frame_path"]
        nodes.append(
            KnowledgeNode(
                content=item["transcript"],
                transcript=item["transcript"],
                visual_summary=visual_summary,
                timestamp=(
                    f"{_format_time(item['timestamp_seconds'])} - "
                    f"{_format_time(item['window_end'])}"
                ),
                # Store a URL path served by the /frames static mount, not
                # an absolute filesystem path (the frontend can't resolve
                # the latter into a loadable image URL).
                frame_path=f"/frames/{frame_path.name}",
                modality=MediaModality.VIDEO,
                source=video_path.name,
                provenance={
                    "frame_timestamp_seconds": item["timestamp_seconds"],
                    "window_end_seconds": item["window_end"],
                },
            )
        )
    return nodes
