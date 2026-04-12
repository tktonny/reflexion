# Clinic Product

`clinic/` is the standalone provider-facing clinic diagnostic-adjunct product.

The product now exposes two demo paths:

- `Realtime conversation demo`
  Live patient interview surface backed by `qwen-omni-realtime` when configured, using `server_vad` for continuous streaming turn detection, transcript capture, feature extraction, embedding-style similarity scoring, and a single-screen risk narrative.
- `Batch upload assessment`
  The original provider-mesh flow for one recorded patient video.

The current design direction for the next platform stage is documented in:

- `docs/architecture/realtime-batch-longitudinal-identity-design.md`
  The formal four-layer architecture for realtime capture, post-session batch assessment, longitudinal tracking, and identity linkage.

## Layout

- `frontend_app/`
  Frontend pages, browser services, styles, tests, and docs
- `database/`
  Persistence and storage access layer
- `intelligence/`
  Prompt design, output schemas, provider adapters, and multimodal inference routing dependencies
- `configs/`
  Shared configuration and environment loading
- `tests/`
  Product tests, smoke scripts, and end-to-end validation helpers
- `../backend/`
  Shared backend API, domain models, realtime routes, and batch orchestration used by clinic, doctor, and care

## Run

```bash
pip install -r requirements.txt
uvicorn backend.src.app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the realtime UI.

For local tests:

```bash
pip install -r requirements-dev.txt
```

Store local secrets in `.secret/.env`.

Realtime Qwen knobs:

- `REFLEXION_QWEN_OMNI_REALTIME_URL`
- `REFLEXION_QWEN_OMNI_REALTIME_MODEL`
- `REFLEXION_QWEN_OMNI_REALTIME_TRANSCRIPTION_MODEL`
- `REFLEXION_REALTIME_FLOW_PATH`
  Optional override for the staged realtime conversation flow JSON. Default: `clinic/configs/realtime_conversation_flow.json`

## Smoke Test

```bash
python3 clinic/tests/smoke_provider.py --video /absolute/path/to/sample.mp4 --provider qwen_omni --strict
```

Default Qwen model: `qwen3.5-omni-plus`.

Default Qwen realtime model: `qwen3-omni-flash-realtime`.
