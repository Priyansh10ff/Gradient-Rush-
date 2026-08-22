"""Gemini Flash Vision analysis for standalone images and PDF artifacts."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be read or described by Gemini."""


_IMAGE_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_UNAVAILABLE_IMAGE_CONTENT = "[Image uploaded - visual description unavailable]"
logger = logging.getLogger(__name__)


def _gemini_model() -> Any:
    """Configure and return Gemini Flash using ``GEMINI_API_KEY`` from .env."""
    try:
        from dotenv import load_dotenv
        import google.generativeai as genai
    except ImportError as exc:
        raise ImageProcessingError(
            "Gemini support requires google-generativeai and python-dotenv."
        ) from exc
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ImageProcessingError("GEMINI_API_KEY is not configured.")
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as exc:
        raise ImageProcessingError("Unable to initialize Gemini Flash.") from exc


def analyze_image(image_path: Path, *, client: Any | None = None) -> dict[str, Any]:
    """Return detailed visual summary, OCR text, and entities via Gemini Flash."""
    if not image_path.is_file():
        raise ImageProcessingError(f"Image source does not exist: {image_path}")
    media_type = _IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())
    if media_type is None:
        raise ImageProcessingError("Only PNG and JPEG images are supported.")

    prompt = (
        "Analyze this image for retrieval. Return JSON with exactly these keys: "
        "visual_summary (detailed description of visual elements, diagrams and layout), "
        "ocr_text (all legible text), and entities (an array of named people, products, "
        "systems, or concepts)."
    )
    try:
        response = (client or _gemini_model()).generate_content(
            [prompt, {"mime_type": media_type, "data": image_path.read_bytes()}],
            generation_config={"response_mime_type": "application/json"},
        )
        analysis = json.loads(response.text or "{}")
    except Exception as exc:  # Provider and malformed-image errors vary by backend.
        logger.warning("Gemini image analysis failed for %s: %s", image_path.name, exc)
        ocr_text = _fallback_ocr(image_path)
        return {
            "visual_summary": _UNAVAILABLE_IMAGE_CONTENT,
            "ocr_text": ocr_text or _UNAVAILABLE_IMAGE_CONTENT,
            "entities": [],
        }

    entities = analysis.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    return {
        "visual_summary": str(analysis.get("visual_summary", "")).strip(),
        "ocr_text": str(analysis.get("ocr_text", "")).strip(),
        "entities": [str(entity) for entity in entities if str(entity).strip()],
    }


def _fallback_ocr(image_path: Path) -> str:
    """Extract readable text locally when Gemini Vision is unavailable."""
    try:
        import easyocr

        model_directory = Path.cwd() / "data" / "models" / "easyocr"
        allow_downloads = os.getenv("EASYOCR_ALLOW_MODEL_DOWNLOADS", "false").strip().lower()
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=str(model_directory),
            download_enabled=allow_downloads in {"1", "true", "yes", "on"},
            verbose=False,
        )
        regions = reader.readtext(str(image_path), detail=1, paragraph=False)
        return " ".join(
            str(region[1]).strip()
            for region in regions
            if len(region) >= 2 and str(region[1]).strip()
        )
    except Exception as exc:
        logger.warning("Local OCR fallback failed for %s: %s", image_path.name, exc)
        return ""


def process_image(file_path: str, *, client: Any | None = None) -> KnowledgeNode:
    """Create one image knowledge node from a local PNG or JPEG upload."""
    image_path = Path(file_path)
    analysis = analyze_image(image_path, client=client)
    return KnowledgeNode(
        content=analysis["ocr_text"],
        visual_summary=analysis["visual_summary"],
        modality=MediaModality.IMAGE,
        source=image_path.name,
        frame_path=str(image_path),
        entities=analysis["entities"],
        provenance={"kind": "standalone_image"},
    )
