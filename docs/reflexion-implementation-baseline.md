# Reflexion Implementation Baseline (Phase 0 Freeze)

> Status: **frozen 2026-07-23**. This is the single implementation baseline for the June–Aug 2026
> consumer product. It reconciles the code as it exists today against
> [`June-Aug 2026, Reflexion Tech Document.docx`](./June-Aug%202026,%20Reflexion%20Tech%20Document.docx)
> (the product source of truth) and [`docs/mirror-app/requirements-source-alignment.md`](./mirror-app/requirements-source-alignment.md).
> Where the requirement doc and the current code disagree, the decisions in §2 are authoritative for this cycle.

The product is a **daily reassurance companion**, not a diagnostic device. One insight governs every
screen: caregivers want to know *"is Mum okay today?"* — never a cognitive score.

Data flows **one direction**: `Mirror APK conversation signals → Backend status engine → Caregiver app display`.
The mirror never computes or shows green/amber/red. The caregiver app never computes status — it renders
what the backend sends.

---

## 1. Component responsibilities (unchanged, restated for the freeze)

| Component | Owns | Must never |
|---|---|---|
| **Mirror APK** (`mirror-app/`) | Aria conversation, raw session + per-turn signals, heartbeat, offline queue, device-level kiosk/boot | Compute or display status/score to the elderly user |
| **Backend status engine** (`reflexion-server/`) | Store raw signals, baselines, ratios, thresholds, status, scheduled jobs, notifications, agent tools | Run the AI conversation; render UI |
| **Caregiver app** (`caregiver-app/`) | Render backend status + reasons + notifications + trends + caregiver actions | Run the conversation; independently compute status |

---

## 2. Decisions that resolve the open conflicts

These were flagged by the phase audit (2026-07-23) as ambiguous. Resolved here per the requirement doc.

1. **v1 is canonical.** `reflexion-server/src/v1/**` (`/api/v1`) is the single source of truth for
   pairing, sessions, realtime tickets, status, tools, and monitoring. Every legacy endpoint
   (`/api/qwen-token`, `/nurse-patient-config/*`, `/nurse-patient-config/mirrors/connect`) is
   **deprecated**, gated behind `ENABLE_LEGACY_API`, and slated for removal at the 2026-12-31 Sunset.
   New work targets v1 only.
2. **Caregiver status source = `GET /api/v1/patients/:id/status`** (`computeCaregiverStatus`). The
   caregiver app must migrate off `/nurse-patient-config/latest`. This requires the app to obtain a
   v1 human JWT (`POST /auth/sessions`) — see §5.
3. **`establishing` is a first-class reassuring state (⚪), never red.** During the baseline period the
   card shows *"Getting to know [Name] — N of 7 sessions recorded"*. The app must have an explicit
   establishing branch; it must not bucket to `needs_attention`.
4. **Reassurance baseline uses median + MAD** (robust, per Signal-to-Status §8.3). The
   `algorithmVersion: 'reassurance-ewma-v1'` label is misleading and is corrected: either compute EWMA
   (α = 0.1) as the doc's Updated-Metrics §1.2 requests, or rename to `reassurance-median-mad-v1`.
   For this cycle we **keep median/MAD as the operating statistic** and additionally compute an EWMA
   trend value for the routine-window metrics so both labels are truthful.
5. **No dementia / cognitive-score wording anywhere the caregiver or elderly sees.** Remove the string
   *"Cognitive Stability Score"* from `caregiver-app/app/trend/[id].tsx`. V2-only per the doc.
6. **Agent write-tool policy.** The conversational agent may **confirm a medication occurrence**
   (`reminder.respond`, yes/no) — this is a top-3 WTP feature in the doc. It may **not** create or edit
   medication schedules or dosages (caregiver/provider only). `caregiver_task.create` is exposed as a
   controlled device tool (server-side schema + authz). All other writes stay caregiver/REST-only.
7. **Mirror session lifecycle is fail-closed** (no mid-session reconnect). Any transport blip ends the
   session; the captured turns are preserved via the offline queue and the session is completed or
   abandoned. This is an intentional contract decision — the "reconnect count" telemetry field is
   therefore expected to be `0` in MVP and exists only to detect the (unsupported) case.
8. **Heartbeat cadence = every 5 minutes, running in the background** (not foreground-only), so an
   ambient idle mirror is not mistaken for offline. Server unreachable threshold stays 15 min
   (`> 3 missed heartbeats`). `backend_reachable` must be a real probe, not `navigator.onLine`.
9. **Caregiver-app visual system = Option 1 (Sage / Bronze / Terracotta)** per the doc ("go w this
   first"): primary `#596C56`, bg `#F7F3EC`, statuses `#596C56 / #9A7A45 / #9B5F4E`, Georgia headings +
   Inter body, Lucide icons, 5-tab nav (Home / Loved Ones / Alerts / Guide / Settings), status **pills**
   not traffic lights, wording rules (Loved ones, Status, Conversation — never Patient/Risk/Assessment).
   This is binding for the trial but ranks **after** functional correctness in execution order.

---

## 3. FROZEN CONTRACT — Session upload (Mirror → Backend)

The mirror uploads **raw signals only**. All fields below map to Signal-to-Status §3/§15 and the
Updated-Metrics MVP set (M1–M7, M13). Server route: `POST /api/v1/sessions/:id/complete`.

### 3.1 `acquisitionSummary` (extended — both sides must match)

`validateAcquisitionSummary` in `reflexion-server/src/v1/routes/sessions.ts` is extended from the current
`{durationMs, patientSpeechMs, patientTurns}` to the full set. Unknown keys are ignored (forward-compatible).

```
acquisitionSummary: {
  durationMs:             int   // total session clock (endedAt - startedAt)
  patientSpeechMs:        int   // Σ user VAD speech (AI excluded) → M4 speech duration
  ariaSpeechMs:           int   // Σ Aria playback duration
  patientTurns:           int   // user_turn_count (responses ≥ 3 words) → M1/M6
  ariaTurns:              int   // aria_turn_count
  repromptCount:          int   // gentle re-prompts issued (empty/echo/timeout rejections)
  wordCount:              int|null   // CJK-aware user word count (null if ASR unavailable) → M7/M9
  transcriptAvailable:    bool  // did ASR produce usable text
  medianResponseLatencyMs:int|null   // median of valid-turn latencies (§10.5) → M7/M13
  sessionStatus:          'completed' | 'incomplete' | 'technical_error'
  technicalError:         bool
  technicalErrorType:     string|null
  timezone:               string   // IANA (also on session.acquisition.timezone at create)
  appVersion:             string
}
```

Completion (M1) is decided **by the backend**, not the mirror: `completed = patientSpeechMs ≥ 30000 AND
patientTurns ≥ 3 AND technicalError == false`. The mirror's `sessionStatus` is advisory context only.

### 3.2 Per-turn events (`kind: 'transcript_turn'`)

The server's `materializeTranscriptTurn` already reads `startedAt` / `endedAt` from the payload — the
mirror must now actually send them (today all `occurredAt` collapse to session end). Payload:

```
payload: {
  turnId, role: 'patient'|'assistant', text,
  startedAt:        ISO   // real utterance start (per-turn)
  endedAt:          ISO   // real utterance end
  protocolStage?:   'warm_up'|'yesterday_recall'|'present_planning'|'medication_reminder'|'reminiscence'
  cognitiveSignals?: string[]
  protocolVersion?: 'daily-conversation-v2'
}
```
`event.occurredAt` = the turn's real `endedAt` (not session end).

### 3.3 Per-turn timing metrics (`kind: 'capture_metric'`)

One `capture_metric` event per user turn carries the latency signal M7/M13 needs:

```
payload: {
  metric: 'turn_timing',
  turnId, questionId /* = protocolStage or scripted-question id */,
  ariaPromptEndAt: ISO, userSpeechStartAt: ISO|null,
  responseLatencyMs: int|null,     // null when no-response turn (routes to reprompt, not latency)
  userSpeechMs: int, repromptCount: int
}
```

---

## 4. FROZEN CONTRACT — Caregiver status read model

`GET /api/v1/patients/:id/status` returns (shape already in `monitoring.ts`, extended with M3/M5 outputs):

```
{
  patientId,
  baselineState: 'establishing' | 'complete',
  baselineProgress: { completedSessions, requiredSessions: 7, windowDays: 14 },
  status:      'establishing' | 'doing_well' | 'worth_checking' | 'needs_attention',
  primaryReason:   <REASON_CODE>,
  secondaryReasons: <REASON_CODE[]>,
  completedToday: bool,
  technicalState: 'ok' | 'possible_issue' | 'unreachable' | 'unknown',
  lastInteractionAt: ISO|null,
  updatedAt: ISO
}
```

Reason codes (stable, plain-English mapped in the app):
`LEARNING_PERSONAL_ROUTINE, DAILY_PATTERN_ON_TRACK, CHECKIN_COMPLETED_TODAY, CHECKIN_MISSED_TODAY,
CHECKIN_MISSED_REPEATEDLY, CHECKIN_MISSED_3_DAYS, CHECKIN_OUTSIDE_USUAL_WINDOW (M3),
WEEKLY_ENGAGEMENT_DOWN (M5), SPOKE_LESS_THAN_USUAL (M4), FEWER_RESPONSES (M6),
SLOWER_TO_RESPOND (M7), DEVICE_UNREACHABLE, AWAY_PERIOD_ACTIVE,
CAREGIVER_FLAG_WORTH_CHECKING, CAREGIVER_FLAG_NEEDS_ATTENTION`.

Priority: Red > Amber > Green, `establishing` overrides all when baseline incomplete (except
completion/missed/technical/manual-flag which are Day-1 active). The card shows exactly one
`primaryReason`; the rest are `secondaryReasons`.

**Shadow isolation (hard rule):** `computeCaregiverStatus` must read **only** operational signals
(`operationalBaselines`, `sessions`, `manualFlags`, `awayPeriods`, `devices`). It must never read
`anomalyScores`, `baselineModels(longitudinal_research)`, or `featureEmbeddings`. A regression test
enforces this (Phase 6).

---

## 5. Caregiver-app ↔ v1 auth bridge

The app currently stores `{nurseId, name, email}` with no token. v1 status/away/flag routes require a
human JWT. Freeze: on sign-in the app calls `POST /auth/sessions` to obtain a v1 access token +
patient list, stores the token in secure storage, and sends `Authorization: Bearer` on all v1 reads.
The legacy `nurseId` session is retained only until the caregiver dashboard fully runs on v1.

---

## 6. Deployment-gated items (code complete, need environment)

These cannot be verified in a dev-only environment and are explicitly **out of "done" for code review**,
tracked for the deploy runbook:

- **Object store**: `OBJECT_STORE_DRIVER=s3` (or the new local driver) must be configured or artifact
  upload returns 503. A filesystem/local driver is added for dev + single-box trial.
- **Workers**: the outbox worker + `session.completed` / `artifact.committed` consumers must be running
  or `processing-status` stays `queued` forever.
- **Scheduled jobs**: `evaluate_7pm` + `finalize_day` must be registered (per-patient, timezone-aware).
- **Native audio**: `getPlaybackBacklogMs()` accuracy and semantic-VAD-off TV/room robustness require a
  physical Android panel. The JS drain guard is hardened to fail **safe** (never fail-open) when the
  native backlog value is unavailable.

---

## 7. Per-phase acceptance criteria

**Phase 1 — realtime core.** Blocking: correct question order; user never interrupted on a normal
pause; Aria audio plays fully before advancing (drain-gated, not `response.done`-gated); one active
response at a time; each turn emits a structured production telemetry record. Latency is recorded but
**not** a blocking criterion this cycle. Device audio round-trip validated on real hardware before trial.

**Phase 2 — mirror UI.** All conversation states rendered from real conversation state (no timers);
ambient home localized to patient language; no status/score shown to the elderly; widget row wired to a
real source or removed.

**Phase 3 — unified backend.** All 9 primitives idempotent on v1; shared claim logic (no duplicated
legacy bridge); object store + workers documented in the deploy runbook.

**Phase 4 — caregiver status loop.** App renders v1 status incl. `establishing`; M1–M5 scored against
personal baseline; away days excluded from the missed-day proxy; midnight finalize persists
`daily_statuses`; thresholds in a `rule_registry`; notification dedup by `user+date+type`;
NO dementia score anywhere.

**Phase 5 — tools.** `weather.get`, `web.search` (allowlisted, redacted), `medication.list`,
`reminders.upcoming`, plus new `reminder.respond` (med confirm) and `caregiver_task.create`, all
device-scoped with server-side schema validation.

**Phase 6 — longitudinal (shadow).** Real features/embeddings/baselines/review-cases; identity gate
labeled "device-assignment only"; a test proves caregiver status never reads research signals.

**Phase 7 — device + trial.** Kiosk lock-task + launch-on-boot + immersive; no poison offline queue
(zero-transcript represented, retries capped/backed-off); background heartbeat + real reachability;
crash recovery + OTA + visible recording state; 24h + 1-week soak.

---

## 8. Execution order (critical path)

1. **Mirror per-turn/session capture + full upload contract** (§3) — the linchpin that feeds Phase 4 & 6.
2. **Backend engine**: extend `acquisitionSummary`, implement M3/M4/M5, midnight finalize + `daily_statuses`,
   `rule_registry`, away-day subtraction.
3. **Mirror production telemetry** (§ Phase 1 Req 4) — reuses the same captured records.
4. **Caregiver-app onto v1 status** (§4/§5) + establishing state + reasons/away/flag; remove score wording.
5. **Backend deploy switches** (§6): local object-store driver, worker/job registration, shared claim logic.
6. **Phase 5 tools**, **Phase 6 shadow test + identity relabel**, **Phase 7 device hardening**, **Phase 2 polish**.
7. **Retire legacy** before the 2026-12-31 Sunset.
