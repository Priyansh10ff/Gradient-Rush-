"""Upload every file in test_data/samples/ through the running API.

This exercises the REAL pipeline (Whisper/Gemini/pypdf/OpenCV, whichever are
configured) rather than writing nodes directly into Chroma, so it's the
right script to run once you have `.env` set up and `uvicorn main:app`
running.

If you don't have API keys configured yet, use
`test_data/seed_cross_modal_demo.py` instead -- it seeds equivalent
knowledge directly, no keys or running server required.

Usage:
    uvicorn main:app --reload &
    python test_data/upload_samples.py
    python test_data/upload_samples.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests --break-system-packages")
    sys.exit(1)


SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# (filename, endpoint, content-type)
UPLOADS = [
    # Scenario 1: checkout timeout incident
    ("incident-review.mp4", "/upload/video", "video/mp4"),
    ("standup-notes.mp3", "/upload/audio", "audio/mpeg"),
    ("grafana-after-fix.png", "/upload/image", "image/png"),
    ("checkout-timeout-postmortem.pdf", "/upload/pdf", "application/pdf"),
    ("support-tickets.json", "/upload/json", "application/json"),
    ("engineering-notes.txt", "/upload/json", "text/plain"),
    # Scenario 2: onboarding permission-flow redesign
    ("onboarding-walkthrough.mp4", "/upload/video", "video/mp4"),
    ("design-standup.mp3", "/upload/audio", "audio/mpeg"),
    ("onboarding-ab-results.png", "/upload/image", "image/png"),
    ("onboarding-design-spec.pdf", "/upload/pdf", "application/pdf"),
    ("user-feedback-survey.json", "/upload/json", "application/json"),
    ("design-notes.txt", "/upload/json", "text/plain"),
]

DEMO_QUERIES = [
    "How did the team fix the checkout timeout issue and how do we know it worked?",
    "Why did we redesign the onboarding permissions flow, and did it actually improve completion rate?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    print(f"Uploading samples from {SAMPLES_DIR} to {args.base_url}\n")
    for filename, endpoint, content_type in UPLOADS:
        path = SAMPLES_DIR / filename
        if not path.is_file():
            print(f"SKIP {filename}: not found at {path}")
            continue
        with path.open("rb") as fh:
            response = requests.post(
                f"{args.base_url}{endpoint}",
                files={"file": (filename, fh, content_type)},
                timeout=180,
            )
        status = "OK" if response.ok else "FAIL"
        print(f"{status} {endpoint:14s} {filename:36s} -> {response.status_code} {response.text[:200]}")

    for query in DEMO_QUERIES:
        print(f"\nRunning demo query: {query!r}\n")
        response = requests.post(
            f"{args.base_url}/query",
            json={"query": query, "limit": 5},
            timeout=60,
        )
        if not response.ok:
            print(f"Query failed: {response.status_code} {response.text}")
            continue

        payload = response.json()
        answer = payload.get("answer") or {}
        print("Answer:", answer.get("answer"))
        print("Grounded:", answer.get("grounded"), "| method:", answer.get("method"))
        print("Evidence used:")
        for src in answer.get("sources", []):
            print(f"  - {src.get('modality'):<6} {src.get('source'):<36} {src.get('locator')}")

        print("Raw retrieval results:")
        for r in payload.get("results", []):
            print(f"  - {r.get('modality'):<6} {r.get('source'):<36} score={r.get('similarity_score'):.3f}")


if __name__ == "__main__":
    main()
