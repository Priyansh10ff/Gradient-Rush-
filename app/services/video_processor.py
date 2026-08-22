"""OpenCV/Gemini processing for timestamp-aligned video knowledge nodes."""

from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services.audio_processor import AudioProcessingError, process_audio
from app.services.image_processor import ImageProcessingError, analyze_image


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
    """Ask Gemini Flash for a compact, retrieval-oriented visual description."""
    try:
        summary = analyze_image(frame_path, client=client)["visual_summary"]
    except ImageProcessingError as exc:
        raise MediaProcessingError(f"Gemini vision analysis failed for {frame_path.name}.") from exc
    if not summary:
        raise MediaProcessingError(f"Gemini vision returned no description for {frame_path.name}.")
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
    frame_interval_seconds: int = 10,
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

    nodes: list[KnowledgeNode] = []
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
            nodes.append(
                KnowledgeNode(
                    content=transcript,
                    transcript=transcript,
                    visual_summary=_describe_frame(frame_path, gemini_client),
                    timestamp=f"{_format_time(timestamp_seconds)} - {_format_time(window_end)}",
                    frame_path=str(frame_path),
                    modality=MediaModality.VIDEO,
                    source=video_path.name,
                    provenance={
                        "frame_timestamp_seconds": timestamp_seconds,
                        "window_end_seconds": window_end,
                    },
                )
            )
            timestamp_seconds += frame_interval_seconds
    finally:
        capture.release()
    return nodes
