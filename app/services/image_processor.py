"""Gemini Flash Vision analysis for standalone images and PDF artifacts."""

import json
import os
from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services.storage import storage_root


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be read or described by Gemini."""


_IMAGE_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

# gemini-1.5-flash and the whole Gemini 1.x family have been fully retired
# by Google (requests now 404). gemini-2.5-flash is the current stable,
# GA vision-capable model as of writing this.
_GEMINI_MODEL = "gemini-2.5-flash"


def _gemini_client() -> Any:
    """Build a Gen AI SDK client using ``GEMINI_API_KEY`` from .env.

    Uses the ``google-genai`` package (``from google import genai``), which
    is Google's current, supported SDK. The older ``google-generativeai``
    package this project originally used is deprecated upstream.
    """
    try:
        from dotenv import load_dotenv
        from google import genai
    except ImportError as exc:
        raise ImageProcessingError(
            "Gemini support requires google-genai and python-dotenv."
        ) from exc
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ImageProcessingError("GEMINI_API_KEY is not configured.")
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise ImageProcessingError("Unable to initialize the Gemini client.") from exc


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
        from google.genai import types

        active_client = client or _gemini_client()
        response = active_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=media_type),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        analysis = json.loads(response.text or "{}")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ImageProcessingError(f"Gemini vision analysis failed for {image_path.name}.") from exc
    except Exception as exc:  # Provider exceptions vary by installed SDK version.
        raise ImageProcessingError(f"Gemini vision analysis failed for {image_path.name}.") from exc

    entities = analysis.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    return {
        "visual_summary": str(analysis.get("visual_summary", "")).strip(),
        "ocr_text": str(analysis.get("ocr_text", "")).strip(),
        "entities": [str(entity) for entity in entities if str(entity).strip()],
    }


def _servable_frame_path(image_path: Path) -> str:
    """Map a persisted upload path to the URL the /uploads mount serves it at.

    Falls back to the bare filename if the path lives outside the storage
    root (e.g. in tests), which keeps callers from crashing on an edge case
    that never happens in the running service.
    """
    try:
        relative = image_path.resolve().relative_to(storage_root() / "uploads")
    except ValueError:
        return f"/uploads/{image_path.name}"
    return f"/uploads/{relative.as_posix()}"


def process_image(file_path: str, *, client: Any | None = None) -> KnowledgeNode:
    """Create one image knowledge node from a local PNG or JPEG upload."""
    image_path = Path(file_path)
    analysis = analyze_image(image_path, client=client)
    ocr_text = analysis["ocr_text"]
    return KnowledgeNode(
        # Both content AND transcript must be populated so every downstream
        # consumer (text-only index, multimodal index, UI) can find the text.
        content=ocr_text,
        transcript=ocr_text,
        visual_summary=analysis["visual_summary"],
        modality=MediaModality.IMAGE,
        source=image_path.name,
        frame_path=_servable_frame_path(image_path),
        entities=analysis["entities"],
        provenance={"kind": "standalone_image"},
    )
