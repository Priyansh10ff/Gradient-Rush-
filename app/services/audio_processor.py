"""Free-tier Groq Whisper transcription with a local faster-whisper fallback."""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    """Raised when an input audio file cannot be transcribed."""


def _response_value(value: object, field: str, default: Any = None) -> Any:
    """Read a field from either an SDK response object or a dictionary."""
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _load_environment() -> None:
    """Load local .env values when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _groq_client() -> Any:
    """Create a Groq client after GEMINI-independent environment loading."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise AudioProcessingError("Groq support requires the groq package.") from exc
    try:
        return Groq(api_key=os.environ["GROQ_API_KEY"])
    except (KeyError, ValueError) as exc:
        raise AudioProcessingError("GROQ_API_KEY is not configured.") from exc


def _normalize_segments(response: object) -> list[dict[str, float | str]]:
    """Normalize Groq's verbose JSON response into the public segment shape."""
    transcripts: list[dict[str, float | str]] = []
    for raw_segment in _response_value(response, "segments", []) or []:
        try:
            start_time = float(_response_value(raw_segment, "start"))
            end_time = float(_response_value(raw_segment, "end"))
        except (TypeError, ValueError) as exc:
            raise AudioProcessingError(
                "Whisper returned a segment without valid start/end timestamps."
            ) from exc
        if end_time < start_time:
            raise AudioProcessingError("Whisper returned a segment with an invalid time range.")
        text = str(_response_value(raw_segment, "text", "")).strip()
        if text:
            transcripts.append(
                {"start_time": start_time, "end_time": end_time, "text": text}
            )
    return transcripts


def _transcribe_with_groq(path: Path, client: Any) -> list[dict[str, float | str]]:
    """Request timestamped segments from Groq's hosted Whisper model."""
    try:
        with path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(path.name, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
    except Exception as exc:  # SDK/provider exceptions vary between releases.
        # Log the full traceback so Groq failures are never silently swallowed.
        _log.exception("Groq Whisper transcription failed for %s: %s", path.name, exc)
        raise AudioProcessingError("Groq Whisper transcription failed.") from exc
    return _normalize_segments(response)


def _transcribe_locally(path: Path) -> list[dict[str, float | str]]:
    """Use CPU faster-whisper when no Groq API key has been configured."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AudioProcessingError(
            "Install faster-whisper for local transcription or set GROQ_API_KEY."
        ) from exc
    try:
        model = WhisperModel(
            os.getenv("FASTER_WHISPER_MODEL", "base"),
            device="cpu",
            compute_type="int8",
        )
        segments, _ = model.transcribe(str(path), vad_filter=True)
        return [
            {
                "start_time": float(segment.start),
                "end_time": float(segment.end),
                "text": str(segment.text).strip(),
            }
            for segment in segments
            if str(segment.text).strip()
        ]
    except Exception as exc:
        raise AudioProcessingError("Local faster-whisper transcription failed.") from exc


def process_audio(
    file_path: str, *, client: Any | None = None
) -> list[dict[str, float | str]]:
    """Transcribe audio/video with Groq, falling back to local faster-whisper.

    A supplied ``client`` is treated as a Groq-compatible client, which keeps
    the function straightforward to test without network access.
    """
    path = Path(file_path)
    if not path.is_file():
        raise AudioProcessingError(f"Audio source does not exist: {path}")
    _load_environment()
    if client is not None:
        return _transcribe_with_groq(path, client)
    if os.getenv("GROQ_API_KEY"):
        return _transcribe_with_groq(path, _groq_client())
    return _transcribe_locally(path)
