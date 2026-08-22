"""Top-level API router composition."""

from fastapi import APIRouter

from app.api.routes import audio, image, pdf, query, video


api_router = APIRouter()
api_router.include_router(video.router)
api_router.include_router(audio.router)
api_router.include_router(image.router)
api_router.include_router(pdf.router)
api_router.include_router(query.router)
