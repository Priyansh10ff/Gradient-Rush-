"""Page-wise PDF extraction with optional Gemini Flash image analysis."""

from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services.image_processor import ImageProcessingError, analyze_image


class PdfProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened or read page by page."""


def _persist_page_image(image: object, page_number: int, image_index: int) -> Path | None:
    """Persist an embedded pypdf image in the demo-accessible frames directory."""
    data = getattr(image, "data", None)
    if not isinstance(data, bytes):
        return None
    suffix = Path(str(getattr(image, "name", ""))).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        return None
    directory = Path.cwd() / "data" / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pdf_page_{page_number:04d}_image_{image_index:03d}{suffix}"
    path.write_bytes(data)
    return path


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
            extracted_path = _persist_page_image(image, page_number, image_index)
            if extracted_path is None:
                continue
            try:
                analysis = analyze_image(extracted_path, client=client)
            except ImageProcessingError as exc:
                raise PdfProcessingError(
                    f"Unable to analyze image {image_index} on page {page_number}."
                ) from exc
            summaries.append(analysis["visual_summary"])
            entities.extend(analysis["entities"])
            frame_path = frame_path or str(extracted_path)

        nodes.append(
            KnowledgeNode(
                content=page_text,
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
