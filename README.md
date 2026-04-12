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

## Product Layout

- `doctor/frontend_app/`
  Doctor management console assets served at `/doctor`
- `care/frontend_app/`
  Caregiver or patient dashboard assets served at `/care`
- `backend/`
  API, domain models, backend orchestration, media preparation, and serving entrypoint
- `clinic/frontend_app/`
  Mirror or assessment terminal UI, result viewer, browser-side services, styles, and docs
- `clinic/database/`
  Persistence and storage access layer
- `clinic/intelligence/`
  Prompt design, output schemas, provider adapters, and multimodal inference logic
- `clinic/configs/`
  Shared configuration and environment loading
- `clinic/tests/`
  Product tests, local end-to-end validation, smoke-test entrypoints, and manual validation scripts

## Run

1. Create a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Configure provider keys as needed:
   - `QWEN_API_KEY` or `DASHSCOPE_API_KEY`
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`

Store all local and server environment variables only in `.secret/.env`.

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
- `fusion` uses transcript plus extracted key frames and is intentionally preprocessing-based.
- `audio_only` is the final fallback when video-grounded review fails.
- `REFLEXION_ALLOW_MOCK_PROVIDERS=true` lets the UI and fallback chain run even if provider keys are absent, but the results are placeholders.
