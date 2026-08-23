# test_data/samples/

Real, ready-to-use sample files for every input modality — bundled directly
in the repo so anyone can clone and test the full pipeline without
downloading or recording anything themselves.

Two independent scenarios are included, each with **every** modality
(video, audio, image, pdf, json, txt), so you always have more than one
real example per uploader to test against.

## Scenario 1 — checkout timeout incident

Evidence is deliberately split across modalities so a correct answer to the
demo question requires connecting more than one of them, not just doing
vector search over one big text blob.

| File | Modality | Endpoint | What it contains |
| --- | --- | --- | --- |
| `incident-review.mp4` | video | `POST /upload/video` | 25s clip: real synthesized narration (espeak) explaining the timeout root cause, cut over two different slide frames — the problem statement, then the backoff/circuit-breaker fix diagram — at different timestamps. |
| `standup-notes.mp3` | audio | `POST /upload/audio` | Unrelated standup narration, used as a distractor to prove retrieval discriminates rather than returning "everything recent." |
| `grafana-after-fix.png` | image | `POST /upload/image` | A rendered dashboard-style screenshot with real OCR-able text showing the p99 latency drop (4200ms → 310ms) and error-rate drop after the fix shipped. |
| `checkout-timeout-postmortem.pdf` | pdf | `POST /upload/pdf` | 4-page real PDF (Summary / Root Cause / Remediation / Result) with the exact numeric retry policy (`base_delay_ms=200, multiplier=2.0, max_retries=5`, circuit breaker thresholds). |
| `support-tickets.json` | json | `POST /upload/json` | 3 structured support-ticket records — two related (before/after the fix), one unrelated distractor — proving the JSON uploader indexes structured records, not just blobs of text. |
| `engineering-notes.txt` | txt | `POST /upload/json` | Plain-text note tying the other files together — proves `.txt` uploads are indexed too, not just `.json`. |
| `frame_problem.png` / `frame_fix_diagram.png` | image | (source frames) | The two slide images baked into `incident-review.mp4`, kept standalone in case you want to test `/upload/image` on the individual diagrams too. |

**Demo question:**
```
How did the team fix the checkout timeout issue and how do we know it worked?
```
- **why** it broke → spoken narration in `incident-review.mp4` (first slide)
- **what** the fix looks like → the diagram in `incident-review.mp4` (second slide)
- **exact policy numbers** → `checkout-timeout-postmortem.pdf`, page 3
- **proof it worked** → OCR'd text + visual description of `grafana-after-fix.png`
- **customer-facing confirmation** → `support-tickets.json`

## Scenario 2 — onboarding permission-flow redesign

A second, unrelated scenario so you can test discrimination between topics,
not just retrieval within one topic.

| File | Modality | Endpoint | What it contains |
| --- | --- | --- | --- |
| `onboarding-walkthrough.mp4` | video | `POST /upload/video` | 26s clip: real narration explaining the permissions-screen drop-off, cut over two slides — the funnel problem, then the new split-screen flow diagram. |
| `design-standup.mp3` | audio | `POST /upload/audio` | Related but tangential standup note (illustrations + legal review), good for testing partial-relevance ranking. |
| `onboarding-ab-results.png` | image | `POST /upload/image` | A results screenshot with real OCR-able text: completion rate 39% → 74%, contacts-permission requests down 90%. |
| `onboarding-design-spec.pdf` | pdf | `POST /upload/pdf` | 4-page real PDF (Background / Proposed Flow / Success Metrics / Result) with the target metrics and the actual A/B test outcome. |
| `user-feedback-survey.json` | json | `POST /upload/json` | 3 structured survey-response records — before/after sentiment plus one unrelated distractor (dark mode request). |
| `design-notes.txt` | txt | `POST /upload/json` | Plain-text note cross-referencing the PDF and video. |
| `frame_onboarding_problem.png` / `frame_onboarding_fix.png` | image | (source frames) | The two slide images baked into `onboarding-walkthrough.mp4`. |

**Demo question:**
```
Why did we redesign the onboarding permissions flow, and did it actually improve completion rate?
```
- **why** it changed → funnel narration in `onboarding-walkthrough.mp4` (first slide)
- **what** changed → the new flow diagram in `onboarding-walkthrough.mp4` (second slide)
- **target vs actual metrics** → `onboarding-design-spec.pdf`, pages 3-4
- **visual proof** → OCR'd text in `onboarding-ab-results.png`
- **user sentiment confirmation** → `user-feedback-survey.json`

## Try it in under a minute

**Option A — through the real pipeline (needs your `.env` API keys, server running):**

```bash
cp .env.example .env    # fill in GEMINI_API_KEY and GROQ_API_KEY
uvicorn main:app --reload &
python test_data/upload_samples.py
```

This uploads every sample (both scenarios, 12 files) through the actual
`/upload/*` endpoints (Whisper transcription, Gemini vision, pypdf, OpenCV
frame sampling, JSON/text parsing), then runs both demo queries and prints
the synthesized cross-modal answer plus every raw retrieval hit.

> If an upload logs `failed: Gemini vision analysis failed for ...:
> GEMINI_API_KEY is not configured.` in the dashboard, it means `.env` is
> missing or empty — copy `.env.example` to `.env` and add real keys. The
> error message now includes the real underlying reason instead of a
> generic failure string.

**Option B — no API keys, no server (instant, deterministic):**

```bash
python test_data/seed_cross_modal_demo.py
python test_data/test_cross_modal_query.py
```

This writes equivalent knowledge nodes directly into an isolated Chroma
collection and asserts retrieval + answer synthesis actually connect
video, pdf, image, and json evidence across **both** scenarios — useful
for CI or a quick sanity check before a demo.
