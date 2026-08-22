"""Page-aware local PDF text and embedded-image extraction."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import UUID

from app.schemas.knowledge import (
    ExtractedKnowledgeBase,
    MediaModality,
    SourceAsset,
    TemporalLocation,
)
from app.services.image_processor import (
    ImageProcessingError,
    ImageProcessingResult,
    OCREngine,
    extract_image_knowledge,
)


class PdfProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed by pypdf."""


class PdfDependencyUnavailableError(PdfProcessingError):
    """Raised when pypdf is not installed."""


@dataclass(frozen=True, slots=True)
class PdfProcessingResult:
    """All page-level knowledge and non-fatal extraction notices."""

    knowledge: list[ExtractedKnowledgeBase]
    warnings: list[str]


def _embedded_images(page: object) -> Sequence[object]:
    """Read pypdf's page image collection while retaining compatibility guards."""
    try:
        return list(getattr(page, "images", []))
    except Exception as exc:
        raise PdfProcessingError("Unable to inspect embedded PDF images.") from exc


def _persist_embedded_image(
    image_file: object, output_directory: Path, page_number: int, image_index: int
) -> Path | None:
    data = getattr(image_file, "data", None)
    if not isinstance(data, bytes):
        return None

    original_name = Path(str(getattr(image_file, "name", ""))).name
    suffix = Path(original_name).suffix or ".img"
    output_path = output_directory / (
        f"page_{page_number:04d}_image_{image_index:03d}{suffix.lower()}"
    )
    output_path.write_bytes(data)
    return output_path


def extract_pdf_knowledge(
    pdf_path: Path,
    source_asset: SourceAsset,
    artifact_directory: Path,
    *,
    parent_knowledge_id: UUID | None = None,
    ocr_engine: OCREngine | None = None,
) -> PdfProcessingResult:
    """Extract selectable page text plus OCR/metadata from embedded PDF images."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PdfDependencyUnavailableError(
            "pypdf is not installed. Install project requirements to process PDFs."
        ) from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PdfProcessingError(f"Unable to open PDF: {pdf_path.name}") from exc

    image_directory = artifact_directory / "pdf_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    knowledge: list[ExtractedKnowledgeBase] = []
    warnings: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_location = TemporalLocation(page_number=page_number)
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            warnings.append(f"Page {page_number}: text extraction failed ({exc}).")
            text = ""

        if text:
            knowledge.append(
                ExtractedKnowledgeBase(
                    content=text,
                    modality=MediaModality.PDF,
                    timestamp=page_location,
                    source=source_asset.filename,
                    confidence=1.0,
                    source_id=source_asset.source_id,
                    parent_knowledge_id=parent_knowledge_id,
                )
            )

        try:
            embedded_images = _embedded_images(page)
        except PdfProcessingError as exc:
            warnings.append(f"Page {page_number}: {exc}")
            continue

        for image_index, image_file in enumerate(embedded_images, start=1):
            image_path = _persist_embedded_image(
                image_file, image_directory, page_number, image_index
            )
            if image_path is None:
                warnings.append(
                    f"Page {page_number}, image {image_index}: image bytes were unavailable."
                )
                continue
            try:
                image_result: ImageProcessingResult = extract_image_knowledge(
                    image_path,
                    source_asset,
                    modality=MediaModality.PDF,
                    timestamp=page_location,
                    parent_knowledge_id=parent_knowledge_id,
                    ocr_engine=ocr_engine,
                )
            except ImageProcessingError as exc:
                warnings.append(f"Page {page_number}, image {image_index}: {exc}")
                continue
            knowledge.extend(image_result.knowledge)
            warnings.extend(
                f"Page {page_number}, image {image_index}: {warning}"
                for warning in image_result.warnings
            )

    return PdfProcessingResult(knowledge=knowledge, warnings=warnings)


async def process_pdf(
    pdf_path: Path,
    source_asset: SourceAsset,
    artifact_directory: Path,
    *,
    parent_knowledge_id: UUID | None = None,
    ocr_engine: OCREngine | None = None,
) -> PdfProcessingResult:
    """Run PDF processing off the event loop because parsing and OCR are CPU-bound."""
    return await asyncio.to_thread(
        extract_pdf_knowledge,
        pdf_path,
        source_asset,
        artifact_directory,
        parent_knowledge_id=parent_knowledge_id,
        ocr_engine=ocr_engine,
    )
