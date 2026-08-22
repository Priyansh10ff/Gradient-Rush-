"""Page-wise PDF extraction with optional Gemini Flash image analysis."""

import logging
from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services.image_processor import ImageProcessingError, analyze_image

_log = logging.getLogger(__name__)


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


def _render_page_image(page: object, page_number: int) -> Path | None:
    """Render a PDF page to a JPEG using pymupdf (fitz) if available.

    This gives every page a visual representation even when it has no
    embedded images, allowing Gemini to describe charts, diagrams, and
    formatted layouts that pypdf's text extraction misses.
    Returns ``None`` silently when pymupdf is not installed.
    """
    try:
        import fitz  # type: ignore[import-untyped]  # pymupdf
    except ImportError:
        return None
    try:
        # ``page`` is a fitz.Page when the caller uses fitz directly.
        # If it came from pypdf we have no fitz page — caller should pass
        # the fitz page, or we skip rendering.
        if not hasattr(page, "get_pixmap"):
            return None
        pixmap = page.get_pixmap(dpi=150)
        directory = Path.cwd() / "data" / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"pdf_page_{page_number:04d}_render.jpg"
        pixmap.save(str(path))
        return path
    except Exception as exc:
        _log.warning("Could not render PDF page %d to image: %s", page_number, exc)
        return None


def process_pdf(file_path: str, *, client: Any | None = None) -> list[KnowledgeNode]:
    """Extract one KnowledgeNode per PDF page and describe embedded diagrams/images."""
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        raise PdfProcessingError(f"PDF source does not exist: {pdf_path}")

    # Prefer pymupdf (fitz) for both text extraction AND page rendering
    # because it gives us full-page visual snapshots. Fall back to pypdf
    # for text-only extraction when fitz is not installed.
    fitz_doc: Any = None
    try:
        import fitz  # type: ignore[import-untyped]
        fitz_doc = fitz.open(str(pdf_path))
    except ImportError:
        pass
    except Exception as exc:
        _log.warning("pymupdf failed to open %s, falling back to pypdf: %s", pdf_path.name, exc)

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

        # --- Try full-page render via pymupdf first (best visual coverage) ---
        if fitz_doc is not None:
            try:
                fitz_page = fitz_doc[page_number - 1]
                rendered = _render_page_image(fitz_page, page_number)
                if rendered is not None and rendered.is_file():
                    frame_path = f"/frames/{rendered.name}"
                    try:
                        analysis = analyze_image(rendered, client=client)
                        if analysis["visual_summary"]:
                            summaries.append(analysis["visual_summary"])
                        entities.extend(analysis["entities"])
                    except ImageProcessingError as exc:
                        _log.warning(
                            "Gemini vision failed for page %d render of %s: %s",
                            page_number, pdf_path.name, exc,
                        )
                        summaries.append("[Visual description unavailable]")
            except Exception as exc:
                _log.warning("Unexpected error rendering page %d: %s", page_number, exc)

        # --- Fall back to embedded images when fitz render is unavailable ---
        if not summaries:
            for image_index, image in enumerate(embedded_images, start=1):
                extracted_path = _persist_page_image(image, page_number, image_index)
                if extracted_path is None:
                    continue
                # Record the first valid image path for the UI thumbnail.
                if frame_path is None:
                    frame_path = f"/frames/{extracted_path.name}"
                try:
                    analysis = analyze_image(extracted_path, client=client)
                except ImageProcessingError as exc:
                    # One bad image must not crash the whole page/PDF.
                    _log.warning(
                        "Gemini vision failed for image %d on page %d of %s: %s",
                        image_index, page_number, pdf_path.name, exc,
                    )
                    summaries.append("[Visual description unavailable]")
                    continue
                summaries.append(analysis["visual_summary"])
                entities.extend(analysis["entities"])

        visual_summary = "\n".join(s for s in summaries if s)

        nodes.append(
            KnowledgeNode(
                # Both content AND transcript must be populated so the text-only
                # search collection can index this page's text.
                content=page_text,
                transcript=page_text,
                visual_summary=visual_summary or None,
                timestamp=f"Page {page_number}",
                # Ensure frame_path is always a valid URL string or None —
                # never an integer, empty string, or "0".
                frame_path=frame_path if frame_path else None,
                modality=MediaModality.PDF,
                source=pdf_path.name,
                entities=list(dict.fromkeys(entities)),
                provenance={"page_number": page_number, "embedded_image_count": len(embedded_images)},
            )
        )

    if fitz_doc is not None:
        try:
            fitz_doc.close()
        except Exception:
            pass

    return nodes
