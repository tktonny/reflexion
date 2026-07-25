# Clinic Demo

This workspace includes a tiered clinic demo API and UI that implement the 48-hour prototype.

## Overview

- Input: one uploaded video
- Output: a normalized clinic assessment with:
  - `visual_findings`
  - `body_findings`
  - `voice_findings`
  - `content_findings`
  - `risk_score`
  - `risk_label`
  - `quality_flags`
  - `session_usability`
  - `provider_meta`
  - `provider_trace`

## Provider Routing

`Qwen Omni -> Gemini -> fusion -> audio_only`

- Native full multimodal models first
- If both fail, fallback to transcript-plus-frames fusion
- If fusion fails, fallback to audio-only review
- Manual provider override supported
- Strict mode disables fallback

## Repository Layout (top level)

- `README.md`, `docs/` — main project description and planning documents.
- `mirror-app/` — Android/Expo **mirror device app** (smart-mirror client). Voice check-in uses the Qwen Omni Realtime relay under `mirror-app/server/`. See `mirror-app/docs/ECOSYSTEM.md`, `mirror-app/QWEN_RELAY.md`, `mirror-app/docs/ANDROID_BUILD.md`.
- `caregiver-app/` — caregiver / family Expo app (source repo: reflexion-native-app).
- `reflexion-server/` — caregiver Express + MongoDB backend (source repo: reflexion-caregiver-app-server).
- `platform/` — the original **clinic platform** (Python FastAPI "AI brain"). Run all Python commands from inside this folder.

### `platform/` sub-layout

- `platform/doctor/frontend_app/` — Doctor console served at `/doctor`
- `platform/care/frontend_app/` — Caregiver/patient dashboard served at `/care`
- `platform/backend/` — API, domain models, orchestration, media preparation, serving entrypoint
- `platform/clinic/frontend_app/` — Mirror/assessment terminal UI, result viewer, browser services
- `platform/clinic/database/` — persistence and storage access layer
- `platform/clinic/intelligence/` — prompt design, output schemas, provider adapters, multimodal inference
- `platform/clinic/configs/` — shared configuration and environment loading
- `platform/clinic/tests/` — product tests, local end-to-end validation, smoke entrypoints
- `platform/verification/`, `platform/schemas/`, `platform/templates/`, `platform/data/`, `platform/audio_server.py`

## Run

1. Create a Python 3.11+ environment. **All Python commands below run from the `platform/` folder:**

```bash
cd platform
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

   Also install these system binaries on any deployment host:
   - `ffmpeg`
   - `ffprobe`

   They are required for browser-recorded video uploads. Without them, the server cannot reliably standardize `webm` recordings to `mp4`, cannot compress oversized uploads for the Qwen batch path, and batch clinic analysis may fail even when the Python dependencies are installed.

3. Configure provider keys as needed:
   - `QWEN_API_KEY` or `DASHSCOPE_API_KEY`
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`

Store all local and server environment variables only in `platform/.secret/.env`.

4. Optional local smoke-test mode:

```bash
export REFLEXION_ALLOW_MOCK_PROVIDERS=true
```

5. Start the demo:

```bash
uvicorn backend.src.app.main:app --reload
```

## Surface Routes

- `http://127.0.0.1:8000/clinic`
  Mirror or assessment terminal for realtime interview plus batch upload
- `http://127.0.0.1:8000/doctor`
  Doctor management console for result review and patient statistics
- `http://127.0.0.1:8000/care`
  Caregiver or patient dashboard with simplified status updates

## Notes

- `Qwen 3.5 Omni Plus` is the preferred true full-modality path for MP4 video plus audio.
- `Gemini` is the secondary true full-modality path.
- The Qwen batch upload path has a stricter inline video limit than the generic app upload limit. Large session recordings should be standardized and compressed server-side with `ffmpeg`.
- `fusion` uses transcript plus extracted key frames and is intentionally preprocessing-based.
- `audio_only` is the final fallback when video-grounded review fails.
- `REFLEXION_ALLOW_MOCK_PROVIDERS=true` lets the UI and fallback chain run even if provider keys are absent, but the results are placeholders.
