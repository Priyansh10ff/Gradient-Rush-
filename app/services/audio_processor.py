"""Timestamp-aware audio transcription interfaces and knowledge mapping."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    MediaModality,
    SourceAsset,
    TemporalLocation,
)


class AudioProcessingError(RuntimeError):
    """Raised when an ASR provider cannot produce valid timestamped segments."""


@dataclass(frozen=True, slots=True)
class TimestampedTranscript:
    """A provider-neutral ASR segment expressed in seconds."""

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


class ASRService(Protocol):
    """Contract implemented by local or hosted automatic-speech-recognition clients."""

    async def transcribe(self, audio_path: Path) -> Sequence[TimestampedTranscript]:
        """Return ordered, timestamped transcript chunks for one audio file."""


class MockTimestampedASRService:
    """A deterministic placeholder until a real ASR provider is configured.

    Its confidence is deliberately zero, making it impossible to mistake this
    placeholder for a real transcription result.
    """

    async def transcribe(self, audio_path: Path) -> Sequence[TimestampedTranscript]:
        await asyncio.sleep(0)
        label = audio_path.stem.replace("_", " ") or "audio"
        return [
            TimestampedTranscript(
                text=f"[Mock ASR] Transcription pending for {label}.",
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.0,
            )
        ]


class OpenAIWhisperASRService:
    """Optional adapter for an injected AsyncOpenAI-compatible Whisper client.

    The OpenAI SDK is intentionally not a required dependency.  Construct this
    adapter in application wiring when credentials and a client are available.
    """

    def __init__(self, client: Any, model: str = "whisper-1") -> None:
        self._client = client
        self._model = model

    async def transcribe(self, audio_path: Path) -> Sequence[TimestampedTranscript]:
        try:
            with audio_path.open("rb") as audio_file:
                response = await self._client.audio.transcriptions.create(
                    model=self._model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
        except Exception as exc:  # Provider exceptions vary by SDK version.
            raise AudioProcessingError("The Whisper ASR request failed.") from exc

        raw_segments = _get_value(response, "segments", []) or []
        return [_to_transcript_segment(segment) for segment in raw_segments]


def _get_value(value: object, field: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _to_transcript_segment(segment: object) -> TimestampedTranscript:
    """Normalize either dict-based or SDK-object Whisper segment output."""
    try:
        start_seconds = float(_get_value(segment, "start"))
        end_seconds = float(_get_value(segment, "end"))
    except (TypeError, ValueError) as exc:
        raise AudioProcessingError("ASR returned a segment without valid timestamps.") from exc

    raw_confidence = _get_value(segment, "confidence")
    if raw_confidence is None:
        no_speech_probability = _get_value(segment, "no_speech_prob")
        raw_confidence = (
            1.0 - float(no_speech_probability)
            if no_speech_probability is not None
            else 0.5
        )

    try:
        confidence = min(1.0, max(0.0, float(raw_confidence)))
    except (TypeError, ValueError) as exc:
        raise AudioProcessingError("ASR returned an invalid confidence value.") from exc

    return TimestampedTranscript(
        text=str(_get_value(segment, "text", "")).strip(),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        confidence=confidence,
    )


async def process_audio(
    audio_path: Path,
    source_asset: SourceAsset,
    *,
    asr_service: ASRService | None = None,
    modality: MediaModality | None = None,
    parent_knowledge_id: UUID | None = None,
) -> list[ExtractedKnowledgeBase]:
    """Transcribe audio and map every ASR segment to provenance-linked knowledge."""
    provider = asr_service or MockTimestampedASRService()
    segments = await provider.transcribe(audio_path)
    return map_transcript_chunks(
        segments,
        source_asset=source_asset,
        modality=modality or source_asset.modality,
        parent_knowledge_id=parent_knowledge_id,
    )


def map_transcript_chunks(
    segments: Sequence[TimestampedTranscript],
    *,
    source_asset: SourceAsset,
    modality: MediaModality,
    parent_knowledge_id: UUID | None = None,
) -> list[ExtractedKnowledgeBase]:
    """Map ASR output without severing its source and time relationships."""
    knowledge: list[ExtractedKnowledgeBase] = []
    for segment in segments:
        if segment.end_seconds < segment.start_seconds:
            raise AudioProcessingError("ASR returned a segment ending before it starts.")
        if not segment.text:
            continue
        knowledge.append(
            ExtractedKnowledgeBase(
                content=segment.text,
                modality=modality,
                timestamp=TemporalLocation(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                ),
                source=source_asset.filename,
                confidence=segment.confidence,
                source_id=source_asset.source_id,
                parent_knowledge_id=parent_knowledge_id,
            )
        )
    return knowledge
