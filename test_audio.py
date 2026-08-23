import asyncio
import logging
from app.services.audio_processor import process_audio
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

async def test():
    # find shapeofyou.mp3
    uploads_dir = Path("data/uploads")
    for p in uploads_dir.rglob("shapeofyou.mp3"):
        print("Testing file:", p)
        try:
            segments = await asyncio.to_thread(process_audio, str(p))
            print("Success")
        except Exception as e:
            print("Error processing audio:", e)
        break

asyncio.run(test())
