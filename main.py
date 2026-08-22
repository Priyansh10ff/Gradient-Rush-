"""FastAPI entry point for the multimodal ingestion service."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router


app = FastAPI(
    title="Multimodal Data Management Pipeline",
    version="0.1.0",
    description="Upload entry points for RAG-ready multimodal source assets.",
)

# The dashboard is static HTML/CSS/JS served same-origin, but CORS is kept
# open for local demo convenience (e.g. hitting the API from a notebook).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frames_directory = Path.cwd() / "data" / "frames"
_frames_directory.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(_frames_directory)), name="frames")

# Standalone image uploads are persisted under data/uploads/{source_id}/...
# and referenced by the same relative URL scheme, so they need their own
# mount to be viewable in the dashboard.
_uploads_directory = Path.cwd() / "data" / "uploads"
_uploads_directory.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_uploads_directory)), name="uploads")

_static_directory = Path(__file__).parent / "static"
if _static_directory.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_directory)), name="static")

app.include_router(api_router)


@app.get("/", tags=["system"], include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the terminal-brutalist dashboard as the app's root page."""
    return FileResponse(str(_static_directory / "index.html"))


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Small readiness endpoint for local development and deployment probes."""
    return {"status": "ok"}
