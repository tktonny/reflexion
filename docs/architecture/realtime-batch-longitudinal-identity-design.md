# Realtime, Batch, Longitudinal, And Identity Design

## 1. Purpose

This design defines the target four-layer product architecture for Reflexion's patient-conversation and dementia-analysis platform.

The architecture separates:

- realtime guided interaction
- post-session batch multimodal assessment
- longitudinal change tracking
- patient identity attribution and face linkage

The goal is to keep the live conversation loop responsive while reserving formal risk output for a controlled post-session assessment path and a separate longitudinal risk engine.

The same four layers support three delivery surfaces:

- provider management console
- caregiver and patient client app
- assessment terminal or mirror interaction surface

## 2. Product Layers

### Layer A: Realtime Conversation

Purpose:

- guide the patient through a structured interview
- capture synchronized audio, video, transcript, and lightweight quality signals
- keep the interaction fast enough for live use
- avoid making final diagnostic claims inside the live conversation

Primary responsibilities:

- session setup
- stage-by-stage prompt progression
- live transcript capture
- live webcam and microphone capture
- lightweight turn metrics
- browser-side recording of the full session
- websocket relay to a realtime provider when available
- guided fallback when live relay is unavailable

Primary output:

- a complete recorded session plus capture metadata

### Layer B: Batch Multimodal Assessment

Purpose:

- run the formal post-session cognitive-risk analysis on the recorded session
- normalize provider outputs into one clinic assessment contract
- preserve provider routing, fallback trace, and reproducible artifacts

Primary responsibilities:

- upload ingestion
- media standardization
- provider mesh routing
- normalized assessment generation
- assessment persistence

Primary output:

- clinic assessment JSON matching the provider-facing assessment contract

### Layer C: Longitudinal Tracking

Purpose:

- build a per-patient baseline from repeated sessions
- detect drift, slope, volatility, and persistent deviation
- route medium and high longitudinal changes back into provider workflow

Primary responsibilities:

- feature snapshot storage per session
- baseline completeness logic
- deviation and trend scoring
- adherence and quality coverage scoring
- escalation generation

Primary output:

- home longitudinal risk output

### Layer D: Identity And Face Linkage

Purpose:

- ensure the target patient is correctly attributed within a session
- prevent caregivers, family, or staff from polluting the patient's timeline
- link repeated sessions to the same person with confidence-aware rules

Primary responsibilities:

- primary face selection inside a session
- session-level face embedding extraction
- patient enrollment profile storage
- cross-session similarity matching
- identity uncertainty gating

Primary output:

- session-level identity attribution and linkage result

## 3. Design Principles

- Live conversation and formal assessment are separate concerns.
- The realtime layer may capture provisional signals, but the formal clinic output is batch-derived.
- Longitudinal risk is not the same objective as single-session clinic risk.
- Identity uncertainty must block or down-weight longitudinal aggregation.
- One platform should support provider, caregiver, and assessment-terminal surfaces without duplicating source-of-truth logic.
- Speech remains the gating modality for release scope; face remains a research modality until validated.
- Every output must be traceable to stored raw artifacts, feature versions, and model versions.

## 4. End-To-End Flow

1. Browser starts a realtime session for one patient.
2. Browser opens the realtime websocket and, when available, relays audio and sparse image frames to the live provider.
3. Browser records the full audio-video session locally using `MediaRecorder`.
4. Browser accumulates transcript turns and lightweight capture metrics.
5. When the session ends, the browser uploads the recorded session to the batch analysis endpoint.
6. Backend standardizes the uploaded media and routes it through the provider mesh.
7. Backend persists the normalized clinic assessment and any attached session record metadata.
8. Feature extraction jobs write a longitudinal-ready feature snapshot.
9. Identity linkage decides whether the session belongs to the enrolled patient with enough confidence for aggregation.
10. Longitudinal services update baseline and drift state only if identity and QC gates pass.
11. Provider console receives the full formal output, while the caregiver and patient client app receives only provider-approved simplified summaries.

## 5. Service Map

### Current Services

- [main.py](/Users/macbookair/Documents/Cloud/REFLEXION/backend/src/app/main.py)
  FastAPI entrypoint and static UI serving
- [routes.py](/Users/macbookair/Documents/Cloud/REFLEXION/backend/src/app/api/routes.py)
  HTTP and websocket endpoints
- [realtime_service.py](/Users/macbookair/Documents/Cloud/REFLEXION/backend/src/app/services/realtime_service.py)
  realtime websocket relay and guided demo orchestration
- [assessment_service.py](/Users/macbookair/Documents/Cloud/REFLEXION/backend/src/app/services/assessment_service.py)
  upload orchestration and provider mesh routing
- [media_preparer.py](/Users/macbookair/Documents/Cloud/REFLEXION/backend/src/app/services/media_preparer.py)
  ffmpeg-based media standardization and fallback artifact generation
- [storage.py](/Users/macbookair/Documents/Cloud/REFLEXION/clinic/database/storage.py)
  local persistence for uploads and assessments

### Proposed Additional Services

- `backend/src/app/services/session_service.py`
  Owns live session lifecycle, stage progression rules, and session-record assembly
- `backend/src/app/services/feature_snapshot_service.py`
  Converts completed sessions into reusable multimodal feature snapshots
- `backend/src/app/services/longitudinal_service.py`
  Computes baseline, deviation, slope, volatility, and escalation
- `backend/src/app/services/identity_service.py`
  Owns target-patient attribution and cross-session face linkage

## 6. Data Contracts

### Shared Session Record

Use [session-record.schema.json](/Users/macbookair/Documents/Cloud/REFLEXION/schemas/session-record.schema.json) as the common session envelope for both clinic and home products.

Recommended usage:

- realtime layer writes capture metadata and lightweight derived features
- batch layer attaches or references formal clinic assessment output
- longitudinal layer reads the same session envelope to build time-series state

### Clinic Assessment

Use [clinic-assessment-output.schema.json](/Users/macbookair/Documents/Cloud/REFLEXION/clinic/intelligence/schemas/clinic-assessment-output.schema.json) for post-session provider-facing output only.

### Home Longitudinal Output

Use [home-risk-output.schema.json](/Users/macbookair/Documents/Cloud/REFLEXION/schemas/home-risk-output.schema.json) for drift and escalation output after baseline logic.

### Recommended New Contracts

- `schemas/feature-snapshot.schema.json`
  one row per session with reusable speech, task, facial, and timing features
- `schemas/identity-link.schema.json`
  session-level target attribution, face-match confidence, and linkage verdict

## 7. Realtime Layer Design

### Inputs

- patient id
- language
- conversation plan
- webcam stream
- microphone stream

### Runtime Components

- browser recorder
- browser-side transcript capture
- websocket relay
- stage controller
- live UI transcript and prompt rail
- local capture metrics

### Runtime Rules

- no final dementia score is shown during live interaction
- the live model only guides the interview and captures metadata
- the browser records the entire session for later formal assessment
- guided fallback remains available when the live relay is unavailable

### Current Implementation Status

- websocket relay: implemented
- guided fallback: implemented
- browser transcript and capture metrics: implemented
- full-session browser recording and automatic upload: implemented in the current UI
- configurable stage engine with goals and exit rules: implemented

## 8. Batch Layer Design

### Inputs

- recorded session video
- patient id
- language
- optional provider preference
- optional sidecar session record

### Pipeline

1. save upload
2. standardize media
3. prepare fallback artifacts only when needed
4. route through provider mesh
5. normalize into one clinic assessment
6. persist assessment and sidecar metadata

### Current Implementation Status

- upload ingestion: implemented
- media standardization: implemented
- provider mesh fallback: implemented
- normalized clinic assessment persistence: implemented
- direct manual upload path: implemented

## 9. Longitudinal Layer Design

### Inputs

- repeated completed session records
- feature snapshots
- QC state
- identity linkage state
- provider follow-up labels when available

### Core Logic

- baseline completeness
- deviation from personal norm
- trend slope
- volatility
- adherence
- QC coverage
- escalation routing

### Gating Rules

- do not score longitudinal risk before baseline completeness
- do not aggregate sessions with unresolved identity
- do not overcall drift on low-quality or sparse windows

### Current Implementation Status

- product definition and schema: present in docs and schemas
- production service implementation: not yet implemented

## 10. Identity Layer Design

### Inputs

- session video
- detected faces over time
- patient enrollment record
- optional externally confirmed identity metadata

### Outputs

- target patient presence
- within-session primary face selection
- face embedding
- match-to-enrollment confidence
- longitudinal inclusion verdict

### Decision Rules

- if no stable primary face is found, mark identity uncertain
- if face-match confidence is below threshold, do not merge into the patient's longitudinal baseline
- if caregiver or interviewer dominates the visible track, mark the session as identity-risky

### Current Implementation Status

- coarse target-patient presence fields exist in the clinic assessment model
- browser face detection metrics exist
- patient-level face identity linkage is not yet implemented

## 11. Storage Model

### Raw Artifacts

- `data/uploads/<assessment_id>/`
  uploaded source media and sidecar metadata

### Prepared Artifacts

- `data/prepared/<assessment_id>/`
  standardized video and fallback artifacts

### Assessments

- `data/assessments/<assessment_id>.json`
  normalized formal clinic output

### Recommended Future Stores

- `data/session_records/`
  normalized session envelopes
- `data/feature_snapshots/`
  per-session reusable multimodal features
- `data/identity_links/`
  patient attribution and cross-session linkage outcomes
- `data/longitudinal_windows/`
  baseline and drift outputs by patient and window

## 12. API Surface

### Current

- `GET /api/clinic/realtime/status`
- `POST /api/clinic/realtime/analyze`
- `WS /api/clinic/realtime/ws`
- `POST /api/clinic/video/analyze`
- `GET /api/clinic/assessments/{assessment_id}`
- `GET /api/providers`

### Recommended Additions

- `POST /api/clinic/sessions`
  create a live session record and conversation plan
- `POST /api/clinic/sessions/{session_id}/complete`
  finalize session metadata before upload
- `GET /api/home/patients/{patient_id}/baseline`
  baseline completeness and coverage
- `GET /api/home/patients/{patient_id}/risk`
  current longitudinal risk window
- `POST /api/identity/enroll`
  capture a patient identity profile
- `POST /api/identity/link`
  evaluate whether a session belongs to a patient

## 13. Immediate Roadmap

1. Keep the current live-to-batch upload loop as the default clinic demo path.
2. Introduce a first-class session service so transcript, QC, and artifact metadata stop living only in browser state.
3. Write reusable feature snapshots after every completed batch assessment.
4. Add identity linkage and gating before enabling longitudinal aggregation.
5. Build the longitudinal engine only on top of stored session records and feature snapshots, not ad hoc browser metrics.

## 14. Decision Summary

The system should be treated as a four-layer platform:

- realtime captures and guides
- batch assesses formally
- identity decides whether the session belongs to the patient
- longitudinal tracking aggregates only trusted sessions over time

This keeps the live UX responsive, the clinical output auditable, and the long-term monitoring path clean enough to support future validation and regulated claims.

## 15. Product Surfaces

### Provider Management Console

- manages enrollment, consent, review, escalation, and clinician interpretation
- displays full formal assessment, longitudinal trend, and identity-review context

### Caregiver And Patient Client App

- delivers reminders, adherence support, exercises, recommendations, and provider-approved summaries
- does not default to showing raw investigational diagnostic outputs

### Assessment Terminal Or Mirror Surface

- runs the stage-based realtime conversation
- records the full session
- uploads the recording for formal batch multimodal analysis
