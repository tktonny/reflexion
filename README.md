# Reflexion Startup Program

This workspace translates the April 5, 2026 Reflexion startup plan into execution-ready artifacts for product, data, ML, clinical validation, regulatory preparation, and operating cadence.

## What This Package Contains

- `docs/strategy/startup-blueprint.md`
  Company thesis, phase goals, team shape, and milestone gates.
- `docs/strategy/source-anchors.md`
  Internal and external sources that back the operating assumptions.
- `docs/product/clinic-diagnostic-prd.md`
  Product requirements for the provider-facing one-session assessment workflow.
- `docs/product/home-longitudinal-prd.md`
  Product requirements for the provider-managed home monitoring workflow.
- `docs/product/multi-surface-platform-prd.md`
  Product requirements for the provider console, caregiver or patient client app, and assessment terminal surfaces.
- `docs/architecture/platform-architecture.md`
  Target system architecture, environments, and system boundaries.
- `docs/architecture/realtime-batch-longitudinal-identity-design.md`
  Four-layer system design for live conversation, formal batch assessment, longitudinal tracking, and identity linkage.
- `docs/ml/model-development-and-evaluation.md`
  Modeling strategy, dataset rules, validation protocol, and release gates.
- `docs/clinical/clinical-validation-program.md`
  Pilot design, label strategy, endpoints, and evidence-generation plan.
- `docs/regulatory/singapore-regulatory-and-qms-plan.md`
  HSA- and PDPA-aligned operating assumptions and quality-system plan.
- `docs/operations/operating-roadmap.md`
  Quarter-by-quarter execution plan from April 2026 through March 2030.
- `schemas/`
  JSON schemas for the shared data contract, feature snapshots, identity linkage, and product outputs.
- `templates/`
  Templates for requirements traceability, risk management, and model release control.
- `clinic/`
  Clinic and mirror interaction surface assets, domain configs, storage, intelligence logic, and tests.
- `doctor/`
  Provider-facing management console frontend package.
- `care/`
  Caregiver or patient-facing dashboard frontend package.
- `backend/`
  Shared backend API, domain models, routing, and service orchestration for all three surfaces.

## Recommended Starting Order

1. Read the startup blueprint.
2. Align on the clinic and home PRDs.
3. Freeze the data contracts and architecture.
4. Start the clinical, ML, and regulatory workstreams in parallel.
5. Use the templates folder as the initial quality and design-control system.

## Source Inputs

- `Reflexion_Project_Plan_TechLead.docx`
- Previous prototype baseline from the public `tigerlaunch` repository
- Official and research references captured inside the strategy, ML, and regulatory docs

## Program Defaults

- Company start point: April 2026
- Geography: Singapore first
- Buyer: provider first
- Team shape: lean 4-6 person seed team
- Product order: clinic evidence first, home longitudinal moat second
- Claim posture: provider-supervised diagnostic-adjunct for early cognitive impairment, not Alzheimer subtype diagnosis

## Clinic Demo

This workspace now also includes a tiered clinic demo API and UI that implement the 48-hour prototype:

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
- Provider routing: `Qwen Omni -> Gemini -> fusion -> audio_only`
- Routing behavior:
  - Native full multimodal models first
  - If both fail, fallback to transcript-plus-frames fusion
  - If fusion fails, fallback to audio-only review
  - Manual provider override supported
  - Strict mode disables fallback

### Product Layout

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
  Product tests and local end-to-end validation
- `clinic/tests/`
  Smoke-test entrypoints and manual validation scripts

### Run

1. Create a Python 3.11+ environment.
2. Install dependencies:
   `pip install -r requirements.txt`
   For local tests:
   `pip install -r requirements-dev.txt`
3. Configure provider keys as needed:
   - `QWEN_API_KEY` or `DASHSCOPE_API_KEY`
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
   Store local secrets in `.secret/.env`.
4. Optional local smoke-test mode:
   `export REFLEXION_ALLOW_MOCK_PROVIDERS=true`
5. Start the demo:
   `uvicorn backend.src.app.main:app --reload`

### Surface Routes

- `http://127.0.0.1:8000/clinic`
  Mirror or assessment terminal for realtime interview plus batch upload
- `http://127.0.0.1:8000/doctor`
  Doctor management console for result review and patient statistics
- `http://127.0.0.1:8000/care`
  Caregiver or patient dashboard with simplified status updates

### Provider Smoke Test

You can validate a single provider directly from the terminal without using the web UI:

`python3 clinic/tests/smoke_provider.py --video /absolute/path/to/sample.mp4 --provider qwen_omni --strict`

This prints either:

- a normalized clinic assessment on success, or
- a structured error plus `provider_trace` on failure

### Notes

- `Qwen 3.5 Omni Plus` is the preferred true full-modality path for MP4 video plus audio.
- `Gemini` is the secondary true full-modality path.
- `fusion` uses transcript plus extracted key frames and is intentionally preprocessing-based.
- `audio_only` is the final fallback when video-grounded review fails.
- `REFLEXION_ALLOW_MOCK_PROVIDERS=true` lets the UI and fallback chain run even if provider keys are absent, but the results are placeholders.
