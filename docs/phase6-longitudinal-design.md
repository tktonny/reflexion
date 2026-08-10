# Phase 6 — Longitudinal Monitoring & Vectorization (DESIGN ONLY)

> Status: **design, not implemented** (per direction 2026-07-23). This specifies how to close the
> Phase-6 gaps without touching the MVP. Hard rule from `docs/reflexion-implementation-baseline.md` §4:
> **nothing here may drive the caregiver green/amber/red** — Phase 6 is a shadow research pipeline.
> A structural test (`reflexion-server/src/v1/monitoring/shadowIsolation.test.ts`) already enforces that
> `computeCaregiverStatus` reads no anomaly/longitudinal collections; every design below preserves it.

## 1. What exists today (baseline)
Real, unit-tested pipeline in `reflexion-server/src/v1/monitoring/`:
- `pipeline.ts::processCompletedSession` — consent → identity → quality gate → feature snapshot →
  (optional) embedding → operational baseline → research baseline + anomaly score → review case, all
  driven off the outbox (`session.completed`, `artifact.committed`).
- `features.ts` — structured speech/interaction features from transcript turns (missing values kept `undefined`).
- `embeddings.ts` — external OpenAI-compatible **semantic** text embeddings (env-gated, enrichment-only).
- `algorithms.ts` — median/MAD, robustZ (0.6745), cosine, centroid, EWMA, anomaly banding.
- Collections: `feature_snapshots`, `feature_embeddings`, `baseline_models(longitudinal_research)`,
  `anomaly_scores`, `monitoring_windows`, `review_cases`, `review_dispositions`, `identity_links`.

## 2. Gaps to close (design)

### 2.1 Real identity gate (replace the device-assignment stub)
Today `evaluateIdentity` returns `verdict:'linked', confidence:0.7, method:'device_assignment_only'` — it
proves only that the session came from the assigned mirror, not that the *enrolled patient* spoke.

**Design — speaker verification as a gate, additive and env-gated:**
- **Enrollment**: during onboarding, capture 3–5 short consented utterances → compute a speaker
  embedding centroid (ECAPA-TDNN / Resemblyzer / a hosted speaker-embedding service) → store an
  `enrollment_profiles` doc `{tenantId, patientId, family:'speaker_ecapa', modelId, centroid[], dim,
  sampleCount, createdAt, consentRef}`. Never store raw enrollment audio beyond the retention window.
- **Per-session verify**: from the session's patient VAD segments, compute the same speaker embedding,
  cosine-distance to the enrollment centroid → `speakerScore`. Verdict ladder:
  `verified` (score ≥ hi) → `linked` (device-assignment only, no enrollment) → `manual_review`
  (enrollment exists but score in the grey band) → `exclude` (score below lo AND a different-speaker
  centroid matches a household member, Phase 6.x).
- **Confidence**: replace the hardcoded 0.7 with a monotone function of `speakerScore`; when no
  enrollment exists, cap at the current `device_assignment_only` value and label it as such (already done).
- **Isolation**: identity feeds `identity_links` + the research anomaly confidence min() ONLY. It must
  never enter `computeCaregiverStatus`. (Add an assertion to the shadow test's forbidden list if an
  `enrollment_profiles` read ever appears in the caregiver path.)
- **Config**: `IDENTITY_SPEAKER_PROVIDER`, hi/lo thresholds in `ruleRegistry`-style config, not inline.

### 2.2 Acoustic embeddings (add alongside semantic)
Today only `family:'transcript_semantic'` exists. Design a parallel **acoustic** family:
- **Vector**: a fixed-dim utterance-level acoustic embedding (wav2vec2 / HuBERT / a hosted audio model)
  over the concatenated patient VAD segments; family `acoustic_wav2vec2`, stored in `feature_embeddings`
  exactly like the semantic one (same schema, different `family`/`modelId`/`dim`).
- **Baseline + distance**: reuse `buildEmbeddingBaseline` (centroid + distanceMedian/MAD) and
  `scoreEmbedding` verbatim — they are family-agnostic. The anomaly `components` gains
  `acousticDeviation` alongside `structuredDeviation`/`embeddingDeviation`; `raw` = mean of available
  components (already the pattern), so acoustic degrades gracefully to "not available" when absent.
- **Dependency**: requires raw session **audio artifacts** to actually be uploaded (current mirror gap —
  see §4). Until audio flows, acoustic silently stays "insufficient data"; this is acceptable and
  matches the existing "enrichment, not a gate" contract.
- **Scalars too**: extend `features.ts` with acoustic scalars (F0 stats, jitter/shimmer, speech rate,
  pause ratio) computed server-side from the audio — these join the structured baseline (`FEATURE_DIRECTIONS`).

### 2.3 Exact personal distance & persistence
Already implemented (`robustZ` vs personal centroid + `anomalyBand` with `persistenceCount`). Design
additions: (a) per-family z-scores surfaced in `anomaly_scores.components` for reviewer transparency;
(b) persistence across the *research* window (not calendar days) — currently 3 recent scores; make the
window + threshold config-driven; (c) a "change onset" timestamp for the reviewer timeline.

### 2.4 Reviewer workflow (exists; harden)
`review_cases` + `review_dispositions` + `/review-cases*` routes exist with an 8-outcome disposition
enum. Design additions: reviewer **shadow evaluation** — reviewers see the anomaly + all component
z-scores + linked session replay, and record a disposition; dispositions feed a precision/recall
dashboard (`review_metrics`) to calibrate thresholds. No disposition ever changes a caregiver status.

### 2.5 QC / identity ordering (exists; document)
Order is fixed in `combineGates`: consent → identity → quality; `exclude`/`manual_review` short-circuit
before any feature/embedding compute. Keep this; add QC of the **audio artifact** (SNR, clipping,
duration) as a quality signal once audio is uploaded.

## 3. Shadow-evaluation guarantee (must hold)
- Separate code path (`processCompletedSession`) + separate collections; caregiver status reads only
  operational signals.
- **Test-enforced**: extend `shadowIsolation.test.ts` forbidden list with any new research collection
  (`enrollment_profiles`, acoustic families) so a future leak into the caregiver path fails CI.
- A research result can request a review case; it can **never** flip green/amber/red. Promotion of any
  research signal into caregiver-facing status requires an explicit product policy + rule-registry
  version bump, not a code side-effect.

## 4. Upstream dependency (blocks acoustic + Whisper)
Phase 6 acoustic + higher-quality transcription need the mirror to upload raw session **audio (WAV/PCM)
artifacts**. Today the mirror uploads transcript events + optional camera frames only. The artifact
two-phase upload path + local/S3 object store already exist server-side; the mirror gap is capturing
the session WAV and running it through `artifact-upload-plans → PUT → commit`. **This is the first thing
to build when Phase 6 starts** — until then, semantic + structured features work; acoustic is inert.

## 5. Phasing
1. **6.0 (prereq)** — mirror uploads session audio artifact; server QC of audio.
2. **6.1** — acoustic embedding family + scalars (reuses existing baseline/scoring).
3. **6.2** — speaker-verification identity gate + enrollment (env-gated).
4. **6.3** — reviewer shadow-eval dashboard + threshold calibration from dispositions.
Each phase is enrichment-only and independently revertable; none changes MVP caregiver behaviour.

## 6. Config & data-protection notes
- All model ids/providers/thresholds via env + a research rule registry, never inline.
- Enrollment + acoustic vectors are biometric-adjacent → explicit consent (`consents.purpose`),
  Singapore-region storage, projection-exclude vectors from any caregiver-facing query, defined retention.
- Embeddings/vectors are enrichment: their provider failing must never block durable transcript
  ingestion (already the contract in `pipeline.ts`).
