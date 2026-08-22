"""FastAPI entry point for the multimodal ingestion service."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router


app = FastAPI(
    title="Multimodal Data Management Pipeline",
    version="0.1.0",
    description="Upload entry points for RAG-ready multimodal source assets.",
)

_frames_directory = Path.cwd() / "data" / "frames"
_frames_directory.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(_frames_directory)), name="frames")

app.include_router(api_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Small readiness endpoint for local development and deployment probes."""
    return {"status": "ok"}
