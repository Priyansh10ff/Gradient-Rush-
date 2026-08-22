"""Page-wise PDF extraction with optional Gemini Flash image analysis."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services.image_processor import analyze_image


logger = logging.getLogger(__name__)
_EMBEDDED_IMAGE_FALLBACK = "[Embedded image analysis skipped or unreadable]"


class PdfProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened or read page by page."""


def _persist_page_image(
    image: object, pdf_stem: str, page_number: int, image_index: int
) -> Path | None:
    """Persist an embedded PDF image as a browser-safe JPEG frame artifact."""
    data = getattr(image, "data", None)
    if not isinstance(data, bytes):
        return None
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as source_image:
            image_rgb = source_image.convert("RGB")
            directory = Path.cwd() / "data" / "frames"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (
                f"{pdf_stem}_page_{page_number}_img_{image_index}.jpg"
            )
            image_rgb.save(path, format="JPEG")
            return path
    except Exception:
        return None


def process_pdf(file_path: str, *, client: Any | None = None) -> list[KnowledgeNode]:
    """Extract one KnowledgeNode per PDF page and describe embedded diagrams/images."""
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        raise PdfProcessingError(f"PDF source does not exist: {pdf_path}")
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
    except ImportError as exc:
        raise PdfProcessingError("pypdf is required to process PDF files.") from exc
    except Exception as exc:
        raise PdfProcessingError(f"Unable to open PDF: {pdf_path.name}") from exc

    nodes: list[KnowledgeNode] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = (page.extract_text() or "").strip()
            embedded_images = list(getattr(page, "images", []))
        except Exception as exc:
            raise PdfProcessingError(f"Unable to extract PDF page {page_number}.") from exc

        summaries: list[str] = []
        entities: list[str] = []
        frame_path: str | None = None
        for image_index, image in enumerate(embedded_images, start=1):
            extracted_path = _persist_page_image(
                image, pdf_path.stem, page_number, image_index
            )
            if extracted_path is None:
                continue
            try:
                analysis = analyze_image(extracted_path, client=client)
                summaries.append(analysis["visual_summary"])
                entities.extend(analysis["entities"])
            except Exception as e:
                logger.warning(f"Warning: Failed to analyze image on page {page_number}: {e}")
                summaries.append(_EMBEDDED_IMAGE_FALLBACK)
            frame_path = frame_path or f"/frames/{extracted_path.name}"

        nodes.append(
            KnowledgeNode(
                content=page_text,
                transcript=page_text,
                visual_summary="\n".join(summary for summary in summaries if summary),
                timestamp=f"Page {page_number}",
                frame_path=frame_path,
                modality=MediaModality.PDF,
                source=pdf_path.name,
                entities=list(dict.fromkeys(entities)),
                provenance={"page_number": page_number, "embedded_image_count": len(embedded_images)},
            )
        )
    return nodes
