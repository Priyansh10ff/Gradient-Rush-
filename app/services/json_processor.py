"""JSON/plain-text ingestion: turn structured or free-text records into KnowledgeNodes.

This is the fourth input modality alongside video/audio/image/pdf. It exists
so that pre-extracted or externally-sourced knowledge (e.g. ticket data,
chat logs, metadata catalogs, transcripts already produced by another tool,
or a plain .txt note) can be indexed into the same cross-modal collection
without re-running OCR/ASR/vision on it.

Accepted inputs:

1. A JSON array of records, each optionally providing:
     {
       "text": "...",              // or "content" / "transcript"
       "visual_summary": "...",    // optional, e.g. a described chart/diagram
       "locator": "Section 2",     // optional, or "timestamp" / "page"
       "source": "override name",  // optional, defaults to the filename
       "entities": ["..."]         // optional
     }
   -> one KnowledgeNode per record.

2. A single JSON object -> treated as one record using the same field
   names, or, if none of the known text fields are present, the whole
   object is pretty-printed and used as the node's content.

3. A plain-text (.txt, or a .json file that isn't valid JSON) file -> the
   whole file becomes one KnowledgeNode's content/transcript, so nothing is
   silently dropped just because it wasn't JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality


class JsonProcessingError(RuntimeError):
    """Raised when an uploaded file has no indexable content at all."""


_TEXT_KEYS = ("text", "content", "transcript", "body", "description")
_LOCATOR_KEYS = ("locator", "timestamp", "page", "section")


def _record_to_node(record: dict[str, Any], *, default_source: str) -> KnowledgeNode:
    text = ""
    for key in _TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if not text:
        # Nothing recognizable -- keep the record instead of dropping it.
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)

    locator = None
    for key in _LOCATOR_KEYS:
        value = record.get(key)
        if value is not None:
            locator = str(value)
            break

    entities = record.get("entities")
    if not isinstance(entities, list):
        entities = []

    return KnowledgeNode(
        content=text,
        transcript=text,
        visual_summary=str(record.get("visual_summary") or "").strip() or None,
        modality=MediaModality.JSON,
        timestamp=locator,
        source=str(record.get("source") or default_source),
        entities=[str(e) for e in entities if str(e).strip()],
        provenance={"kind": "json_record"},
    )


def _plain_text_node(text: str, *, source: str) -> KnowledgeNode:
    return KnowledgeNode(
        content=text,
        transcript=text,
        modality=MediaModality.JSON,
        source=source,
        provenance={"kind": "plain_text_note"},
    )


def process_json(file_path: str) -> list[KnowledgeNode]:
    """Parse an uploaded JSON or plain-text file into one or more KnowledgeNodes."""
    path = Path(file_path)
    if not path.is_file():
        raise JsonProcessingError(f"Source file does not exist: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise JsonProcessingError(f"Unable to read {path.name} as UTF-8 text.") from exc

    default_source = path.name
    is_json_extension = path.suffix.lower() == ".json"

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        # .txt files (or a .json file that isn't actually valid JSON) still
        # get indexed as plain text instead of being rejected outright.
        stripped = raw_text.strip()
        if not stripped:
            raise JsonProcessingError(f"{path.name} is empty; nothing to index.")
        return [_plain_text_node(stripped, source=default_source)]

    if isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]
        if not records:
            if is_json_extension:
                raise JsonProcessingError(
                    f"{path.name} is a JSON array but contains no object records to index."
                )
            return [_plain_text_node(raw_text.strip(), source=default_source)]
        return [_record_to_node(r, default_source=default_source) for r in records]

    if isinstance(payload, dict):
        return [_record_to_node(payload, default_source=default_source)]

    # Valid JSON but a bare scalar (e.g. a lone string or number) -- index
    # its text form rather than rejecting the upload.
    return [_plain_text_node(str(payload), source=default_source)]

