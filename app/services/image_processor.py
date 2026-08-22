"""Local image metadata and OCR extraction with provenance-preserving mapping."""

import asyncio
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence
from uuid import UUID

from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    MediaModality,
    SourceAsset,
    TemporalLocation,
)
from app.services.storage import storage_root


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be read or a supplied OCR engine fails."""


class ImageDependencyUnavailableError(ImageProcessingError):
    """Raised when Pillow, EasyOCR, or local OCR model weights are unavailable."""


@dataclass(frozen=True, slots=True)
class OCRRegion:
    """One OCR text region with its image-space bounding polygon."""

    text: str
    confidence: float
    bounding_box: list[list[float]]


@dataclass(frozen=True, slots=True)
class ImageProcessingResult:
    """Knowledge and non-fatal OCR notices from one image."""

    knowledge: list[ExtractedKnowledgeBase]
    warnings: list[str]


class OCREngine(Protocol):
    """Synchronous local OCR contract, run off FastAPI's event loop."""

    def read(self, image_path: Path) -> Sequence[OCRRegion]:
        """Return recognized text regions for one image."""


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def _create_easyocr_reader(
    languages: tuple[str, ...], model_directory: str, allow_model_downloads: bool
) -> object:
    """Initialize a CPU-only EasyOCR reader once per local model configuration."""
    try:
        import easyocr
    except ImportError as exc:
        raise ImageDependencyUnavailableError(
            "EasyOCR is not installed. Install project requirements to enable OCR."
        ) from exc

    try:
        return easyocr.Reader(
            list(languages),
            gpu=False,
            model_storage_directory=model_directory,
            download_enabled=allow_model_downloads,
            verbose=False,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ImageDependencyUnavailableError(
            "EasyOCR model weights are unavailable locally. Place them in the local "
            "model directory or explicitly enable EASYOCR_ALLOW_MODEL_DOWNLOADS."
        ) from exc


@dataclass(frozen=True, slots=True)
class EasyOCREngine:
    """CPU-only EasyOCR adapter with local-model-only behavior by default."""

    languages: tuple[str, ...] = ("en",)
    model_directory: Path | None = None
    allow_model_downloads: bool = False

    def read(self, image_path: Path) -> Sequence[OCRRegion]:
        model_directory = self.model_directory or (
            storage_root() / "models" / "easyocr"
        )
        reader = _create_easyocr_reader(
            self.languages,
            str(model_directory),
            self.allow_model_downloads,
        )
        try:
            raw_regions = reader.readtext(str(image_path), detail=1, paragraph=False)
        except Exception as exc:  # EasyOCR has several backend-specific error types.
            raise ImageProcessingError("EasyOCR failed to analyze the image.") from exc

        regions: list[OCRRegion] = []
        for raw_region in raw_regions:
            if len(raw_region) < 3:
                continue
            bounding_box, text, confidence = raw_region[:3]
            normalized_text = str(text).strip()
            if not normalized_text:
                continue
            try:
                normalized_box = [
                    [float(coordinate) for coordinate in point] for point in bounding_box
                ]
                normalized_confidence = min(1.0, max(0.0, float(confidence)))
            except (TypeError, ValueError):
                continue
            regions.append(
                OCRRegion(
                    text=normalized_text,
                    confidence=normalized_confidence,
                    bounding_box=normalized_box,
                )
            )
        return regions


def extract_image_knowledge(
    image_path: Path,
    source_asset: SourceAsset,
    *,
    modality: MediaModality = MediaModality.IMAGE,
    timestamp: TemporalLocation | None = None,
    parent_knowledge_id: UUID | None = None,
    ocr_engine: OCREngine | None = None,
) -> ImageProcessingResult:
    """Extract image metadata and OCR regions into source-linked knowledge records."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ImageDependencyUnavailableError(
            "Pillow is not installed. Install project requirements to process images."
        ) from exc

    try:
        with Image.open(image_path) as image:
            image.load()
            visual_metadata = {
                "kind": "image_metadata",
                "format": image.format or "unknown",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "frame_count": getattr(image, "n_frames", 1),
                "has_alpha": "A" in image.getbands(),
            }
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ImageProcessingError(f"Unable to load image: {image_path.name}") from exc

    warnings: list[str] = []
    engine = ocr_engine or EasyOCREngine(
        allow_model_downloads=_env_flag("EASYOCR_ALLOW_MODEL_DOWNLOADS")
    )
    try:
        regions = engine.read(image_path)
        visual_metadata["ocr_status"] = "completed"
    except ImageDependencyUnavailableError as exc:
        # Preserve visual context even when local OCR weights have not been provisioned.
        regions = []
        visual_metadata["ocr_status"] = "unavailable"
        warnings.append(str(exc))

    knowledge = [
        ExtractedKnowledgeBase(
            content=visual_metadata,
            modality=modality,
            timestamp=timestamp,
            source=source_asset.filename,
            confidence=1.0,
            source_id=source_asset.source_id,
            parent_knowledge_id=parent_knowledge_id,
        )
    ]
    for region in regions:
        knowledge.append(
            ExtractedKnowledgeBase(
                content={
                    "kind": "ocr_region",
                    "text": region.text,
                    "bounding_box": region.bounding_box,
                },
                modality=modality,
                timestamp=timestamp,
                source=source_asset.filename,
                confidence=region.confidence,
                source_id=source_asset.source_id,
                parent_knowledge_id=parent_knowledge_id,
            )
        )
    return ImageProcessingResult(knowledge=knowledge, warnings=warnings)


async def process_image(
    image_path: Path,
    source_asset: SourceAsset,
    *,
    modality: MediaModality = MediaModality.IMAGE,
    timestamp: TemporalLocation | None = None,
    parent_knowledge_id: UUID | None = None,
    ocr_engine: OCREngine | None = None,
) -> ImageProcessingResult:
    """Run local image processing without blocking FastAPI's event loop."""
    return await asyncio.to_thread(
        extract_image_knowledge,
        image_path,
        source_asset,
        modality=modality,
        timestamp=timestamp,
        parent_knowledge_id=parent_knowledge_id,
        ocr_engine=ocr_engine,
    )
