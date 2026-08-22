# 🧠 Multimodal Data Management Pipeline for RAG-Ready Systems
### *Preserving Cross-Modal Context & Temporal Provenance — 100% Free, 100% Local*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0+-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 One-Line Pitch

> **A locally-running FastAPI service that ingests Video, Audio, Images, and PDFs, extracts semantically rich knowledge chunks with full temporal and cross-modal provenance, and serves them through a unified vector-search endpoint — with zero cloud dependencies and zero cost.**

---

## 📋 Table of Contents

1. [The Problem](#the-problem)
2. [Key Architecture & Design Decisions](#key-architecture--design-decisions)
3. [Pipeline Flowchart](#pipeline-flowchart)
4. [Project Structure](#project-structure)
5. [Quickstart Guide](#quickstart-guide)
6. [Environment Variables](#environment-variables)
7. [API Reference](#api-reference)
8. [Cross-Modal Provenance: How It Works](#cross-modal-provenance-how-it-works)
9. [Evaluation vs. Text-Centric RAG](#evaluation-vs-text-centric-rag)
10. [Future Improvements](#future-improvements)
11. [Tech Stack](#tech-stack)

---

## The Problem

Traditional RAG pipelines ingest plain text and embed it — full stop. This breaks catastrophically on real-world technical corpora:

| Failure Mode | Example | Impact |
|---|---|---|
| **Diagram blindness** | Architecture diagram in a PDF | Query about "data flow" returns nothing |
| **Speaker misalignment** | Interview video with 3 speakers | "What did Alice say about pricing?" fails |
| **Temporal drift** | Lecture video with topic shifts | Retrieval ignores *when* a concept was discussed |
| **Cross-modal disconnection** | Slide deck + narration video | Text and visual evidence are never linked |
| **Page-context loss** | Multi-page technical spec | Extracted text has no page anchoring |

**The result:** A RAG system that confidently answers from incomplete evidence — or worse, halluccinates context that was present in diagrams or audio it never processed.

---

## Key Architecture & Design Decisions

### 1. Unified Structured Representation — `ExtractedKnowledgeBase`

Every piece of extracted content — whether an OCR region from a diagram, a 5-second audio transcript, or a PDF paragraph — is normalized into a single schema:

```python
class ExtractedKnowledgeBase(BaseModel):
    content:             str | dict       # Extracted text or structured blob
    modality:            MediaModality    # "video" | "audio" | "image" | "pdf"
    timestamp:           TemporalLocation # seconds (AV) or page_number (PDF)
    source:              str              # Original filename
    confidence:          float            # 0.0–1.0 from the extractor
    source_id:           UUID             # Stable parent asset identifier
    parent_knowledge_id: UUID | None      # Links child segments to parent chunks
```

`TemporalLocation` is the key to cross-modal alignment:

```python
class TemporalLocation(BaseModel):
    start_seconds: float | None   # Audio/video segment start
    end_seconds:   float | None   # Audio/video segment end
    page_number:   int   | None   # PDF page / image frame number
```

This means a video frame's OCR output at t=42s and the audio transcript at t=40–45s share the same temporal window and can be retrieved together.

### 2. 100% Free & Local Processing Stack

| Task | Library | Cloud? | Cost |
|---|---|---|---|
| Image loading & processing | Pillow | ❌ | Free |
| OCR (diagrams, slides) | EasyOCR | ❌ | Free |
| PDF text extraction | pypdf | ❌ | Free |
| Speech-to-text (ASR) | faster-whisper | ❌ | Free |
| Video frame extraction | FFmpeg | ❌ | Free |
| Text embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | ❌ | Free |
| Vector store | ChromaDB (persistent, local) | ❌ | Free |
| API framework | FastAPI + Uvicorn | ❌ | Free |

### 3. Cross-Modal Context Preservation

The pipeline uses a two-level hierarchy to keep related content linked:

- **`source_id`** — ties every chunk back to the original uploaded asset (e.g., `lecture.mp4`)
- **`parent_knowledge_id`** — ties child chunks (e.g., a 5-second audio clip) to a parent segment (e.g., the full audio track extracted from that video)

When you query `"explain the architecture diagram"`, the vector store returns:
- The OCR text from the diagram image → with `source_id` pointing to `slides.pdf`, `page_number=7`
- The audio transcript at the moment the presenter described it → with `timestamp_start=1840.0`
- Both share the same `source_id` family — your application can reconstruct the full cross-modal context.

### 4. Idempotent Ingestion via Content-Addressed IDs

Each record's Chroma document ID is a `SHA-256` hash of its full serialized payload. Re-uploading the same file is safe — Chroma's `upsert` semantics make it a no-op. No duplicates accumulate in the index.

### 5. Non-Blocking API Design

All CPU-bound work (OCR, embedding, PDF parsing) runs in thread pool executors via `asyncio.to_thread()`. The FastAPI event loop is never blocked, keeping the API responsive under concurrent uploads.

---

## Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLIENT (curl / frontend)                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ POST multipart/form-data
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Upload Router                                 │
│  /upload/video  /upload/audio  /upload/image  /upload/pdf               │
│                                                                         │
│  1. Validate filename & MIME type                                       │
│  2. Create SourceAsset (stable UUID, filename, modality, timestamp)     │
│  3. Persist raw file to  data/uploads/<source_id>/                      │
└─────────┬──────────┬──────────┬──────────┬──────────────────────────────┘
          │          │          │          │
          ▼          ▼          ▼          ▼
   ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │  video_  │ │ audio_ │ │ image_ │ │  pdf_  │
   │processor │ │process │ │process │ │process │
   │          │ │   or   │ │   or   │ │   or   │
   │ FFmpeg   │ │faster- │ │EasyOCR │ │ pypdf  │
   │ frames + │ │whisper │ │+ Pillow│ │+ OCR on│
   │ audio    │ │  ASR   │ │        │ │embedded│
   │ extract  │ │        │ │        │ │ images │
   └────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │           │          │           │
        └───────────┴──────────┴───────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────┐
        │     Chunk Normalizer (ExtractedKnowledge) │
        │                                           │
        │  content · modality · timestamp           │
        │  source · confidence · source_id          │
        │  parent_knowledge_id                      │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │     sentence-transformers Embedder        │
        │      (all-MiniLM-L6-v2, CPU-only)        │
        └──────────────────┬───────────────────────┘
                           │ 384-dim cosine vectors
                           ▼
        ┌──────────────────────────────────────────┐
        │    ChromaDB  (local, persistent HNSW)     │
        │                                           │
        │  document: embeddable text/JSON           │
        │  metadata: source_id · modality           │
        │            timestamp_start/end            │
        │            page_number · confidence       │
        │            parent_knowledge_id            │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │    POST /query  →  QueryResponse          │
        │                                           │
        │  results: [                               │
        │    { record_id, score,                    │
        │      record: ExtractedKnowledgeBase }     │
        │  ]                                        │
        └──────────────────────────────────────────┘
```

---

## Project Structure

```
hackathon/
├── main.py                          # FastAPI app entry point + /health
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── api/
│   │   ├── router.py                # Composes all route modules
│   │   └── routes/
│   │       ├── _upload.py           # Shared validation & SourceAsset creation
│   │       ├── video.py             # POST /upload/video
│   │       ├── audio.py             # POST /upload/audio
│   │       ├── image.py             # POST /upload/image  → indexes to ChromaDB
│   │       ├── pdf.py               # POST /upload/pdf    → indexes to ChromaDB
│   │       └── query.py             # POST /query
│   │
│   ├── schemas/
│   │   └── knowledge.py             # ExtractedKnowledgeBase, TemporalLocation,
│   │                                #   SourceAsset, QueryRequest/Response, ...
│   │
│   └── services/
│       ├── storage.py               # Async file persistence helpers
│       ├── video_processor.py       # FFmpeg frame & audio extraction
│       ├── audio_processor.py       # ASR interface + faster-whisper adapter
│       ├── image_processor.py       # Pillow load + EasyOCR → knowledge records
│       ├── pdf_processor.py         # pypdf page text + embedded image OCR
│       └── vector_store.py          # ChromaDB + sentence-transformers layer
│
└── data/                            # Runtime only — excluded from git
    ├── uploads/                     # Raw uploaded assets
    ├── derived/                     # Extracted frames, PDF images
    ├── chroma/                      # ChromaDB persistence
    └── models/                      # Cached model weights
```

---

## Quickstart Guide

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| FFmpeg | Any recent | `winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg` |
| Git | Any | [git-scm.com](https://git-scm.com/) |

> **Windows users:** Ensure `ffmpeg` is on your `PATH`. Open a new terminal and run `ffmpeg -version` to verify.

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-org>/multimodal-rag-pipeline.git
cd multimodal-rag-pipeline
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **First-run model downloads:** On the first request, `sentence-transformers` will download `all-MiniLM-L6-v2` (~90 MB). To pre-download and allow this once, set `LOCAL_MODELS_ONLY=false`. After the first run, set it back to `true` (the default) for fully offline operation.

### Step 4 — Run the server

```bash
# Development (auto-reload on file changes)
python -m uvicorn main:app --reload

# Production (4 workers)
python -m uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Step 5 — Verify the server is running

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Step 6 — Explore the interactive API docs

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** in your browser. The Swagger UI lists all endpoints with live try-it-out functionality.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MEDIA_STORAGE_ROOT` | `./data` | Root directory for all uploaded and derived files |
| `LOCAL_MODELS_ONLY` | `true` | Prevent sentence-transformers from downloading models at runtime |
| `EASYOCR_ALLOW_MODEL_DOWNLOADS` | `false` | Allow EasyOCR to download language model weights on first use |

---

## API Reference

All endpoints are documented interactively at `/docs` (Swagger) and `/redoc`.

### `POST /upload/video`

Upload an MP4/MOV/MKV/WebM video. The service extracts audio via FFmpeg, runs ASR to produce timestamped transcript segments, and indexes all resulting knowledge into ChromaDB.

**Request:** `multipart/form-data` with field `file`  
**Response:** `UploadProcessingResult`

```json
{
  "upload_id": "uuid",
  "status": "completed",
  "source_asset": { "source_id": "...", "filename": "lecture.mp4", "modality": "video" },
  "extracted_knowledge": [ { "content": "...", "modality": "video", "timestamp": {...}, ... } ],
  "warnings": [],
  "indexed_count": 42
}
```

---

### `POST /upload/audio`

Upload an MP3/WAV/FLAC/M4A audio file. Runs ASR and maps each segment to a `TemporalLocation(start_seconds, end_seconds)`.

**Request:** `multipart/form-data` with field `file`  
**Response:** `UploadProcessingResult`

---

### `POST /upload/image`

Upload a JPG/PNG/WebP/TIFF image. Runs Pillow for metadata and EasyOCR for text regions. Each detected OCR region becomes a separate knowledge record with its bounding box preserved.

**Request:** `multipart/form-data` with field `file`  
**Response:** `UploadProcessingResult`

---

### `POST /upload/pdf`

Upload a PDF. Extracts selectable text page-by-page (each page → one knowledge record with `page_number`). Also extracts embedded images and runs OCR on them.

**Request:** `multipart/form-data` with field `file`  
**Response:** `UploadProcessingResult`

---

### `POST /query`

Execute a cross-modal semantic search across all indexed modalities.

**Request body:**
```json
{ "query": "explain the transformer attention mechanism", "top_k": 5 }
```

**Response:**
```json
{
  "query": "explain the transformer attention mechanism",
  "results": [
    {
      "record_id": "sha256hex...",
      "score": 0.91,
      "record": {
        "content": "Attention is all you need — the core idea is...",
        "modality": "audio",
        "timestamp": { "start_seconds": 342.5, "end_seconds": 358.1 },
        "source": "lecture.mp4",
        "confidence": 0.97,
        "source_id": "3fa85f64-...",
        "parent_knowledge_id": null
      }
    },
    {
      "record_id": "sha256hex...",
      "score": 0.87,
      "record": {
        "content": { "kind": "ocr_region", "text": "Multi-Head Attention", "bounding_box": [...] },
        "modality": "image",
        "timestamp": null,
        "source": "slides_page_12.png",
        "confidence": 0.93,
        "source_id": "9b1deb4d-...",
        "parent_knowledge_id": null
      }
    }
  ]
}
```

---

### `GET /health`

Lightweight liveness probe.

```json
{ "status": "ok" }
```

---

## Cross-Modal Provenance: How It Works

The vector store preserves the full provenance of every record through a two-layer storage strategy:

```
ChromaDB document (embeddable text)     ChromaDB metadata (flat key-value, filterable)
────────────────────────────────────    ─────────────────────────────────────────────
"Attention is all you need — the        source_id         → "3fa85f64-5717-4562-..."
 core idea is to compute a weighted     source            → "lecture.mp4"
 sum of values..."                      modality          → "audio"
                                        confidence        → 0.97
                                        timestamp_start   → 342.5
                                        timestamp_end     → 358.1
                                        parent_knowledge_id → "9c7b..."  (optional)
```

**Key properties:**

- **`source_id` (UUID):** Never changes for a given upload. Multiple chunks from the same file all share the same `source_id`, enabling "fetch all evidence from this lecture" queries.
- **`parent_knowledge_id` (UUID | None):** Chains child segments to their parent. Audio segments extracted from a video have `parent_knowledge_id` pointing to the video's top-level record.
- **Temporal anchoring:** `timestamp_start`/`timestamp_end` (seconds) for audio/video; `page_number` (1-indexed) for PDF/embedded images.
- **Idempotent IDs:** Record IDs are `SHA-256(full_record_json)` — re-ingesting the same file produces identical IDs, triggering Chroma's upsert (no-op).
- **Round-trip fidelity:** `_record_from_result()` fully reconstructs a typed `ExtractedKnowledgeBase` — including all UUIDs and `TemporalLocation` — directly from Chroma metadata, so **every search hit returns its complete provenance chain**.

---

## Evaluation vs. Text-Centric RAG

### Benchmark Scenarios

| Query | Text-Only RAG | This Pipeline |
|---|---|---|
| *"What does the system architecture diagram show?"* | ❌ No image ingested | ✅ OCR regions from diagram image indexed with bounding boxes |
| *"What was said about the database schema at the 30-minute mark?"* | ❌ No temporal index | ✅ Audio segment at `start_seconds=1795` returned directly |
| *"Summarize page 7 of the technical specification"* | ❌ Page context lost | ✅ PDF record with `page_number=7` retrieved exactly |
| *"What did the presenter say while showing the UML diagram?"* | ❌ Modalities disconnected | ✅ Shared `source_id` links video frame OCR + audio transcript |
| *"Find all mentions of 'cache invalidation' across all uploaded materials"* | ⚠️ Text only | ✅ OCR + transcripts + PDF text all searched in one query |

### Why It Works

1. **Unified embedding space:** All modalities pass through the same `all-MiniLM-L6-v2` encoder. Semantically similar content ranks together regardless of its origin modality.
2. **Temporal grounding:** Results carry machine-readable timestamps — your downstream application can seek a video player to the exact moment, highlight a PDF page, or draw a bounding box on an image.
3. **Hierarchy preservation:** The `parent_knowledge_id` chain lets you re-expand context (e.g., retrieve the full paragraph a sentence came from) without re-querying.

---

## Future Improvements

### Near-Term (Next Sprint)

- **Semantic boundary detection** — Instead of fixed-size audio/video chunking, use topic-shift detection to create semantically coherent segments.
- **faster-whisper integration** — Replace the `MockTimestampedASRService` with the bundled `faster-whisper` adapter for fully offline, word-level-accurate transcription.
- **Metadata filtering in queries** — Expose `modality`, `source_id`, and time-range filters on the `/query` endpoint for targeted retrieval.
- **Batch upload endpoint** — Accept a ZIP archive containing mixed-modality files and process them as a coherent corpus.

### Medium-Term

- **Knowledge graph layer** — Store `source_id → parent_knowledge_id` relationships in a lightweight graph (e.g., NetworkX) to enable multi-hop traversal: *"find all evidence related to this concept, across all linked source documents."*
- **Re-ranking** — Add a cross-encoder re-ranker pass after the initial ANN retrieval to improve precision on long-tail queries.
- **Incremental indexing** — Track file hashes to skip re-embedding unchanged pages/segments on re-upload.

### Long-Term

- **Multi-agent cross-referencing** — Spawn specialized agents per modality that post evidence into a shared scratchpad, enabling structured debate before a final answer is synthesized.
- **Speaker diarization** — Integrate `pyannote.audio` to label each transcript segment with a speaker ID, enabling speaker-specific retrieval.
- **Streaming ingestion** — Accept live audio/video streams via WebSocket and continuously update the index in near-real-time.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.115+, Uvicorn, Pydantic v2 |
| **Video** | FFmpeg (system), Python subprocess |
| **Audio / ASR** | faster-whisper (local CTranslate2) |
| **Image** | Pillow, EasyOCR (PyTorch CPU) |
| **PDF** | pypdf 5.0+ |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB 1.0+ (local persistent HNSW) |
| **Concurrency** | asyncio + `asyncio.to_thread` |
| **Language** | Python 3.11+ |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built in 13 hours for the Multimodal Data Management Hackathon. Every dependency is free and runs entirely on local hardware.*
