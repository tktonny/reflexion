# Reflexion implementation tracker

> Product source of truth: **Reflexion Caregiver App Architecture v4.0 (3 August 2026)**.
> The existing `/api/v1` platform contract remains the implementation boundary; this tracker records
> the work needed to make the caregiver app, Mirror APK and backend one coherent pilot system.

## Environment baseline

| Runtime | Supported / validated value | Source of truth |
| --- | --- | --- |
| Node.js | `24.x` (all JS packages, local/CI/EAS pin: `24.18.0`) | root `.nvmrc`, package `engines`, EAS build profiles |
| MongoDB for local/CI tests | `7.0.14` replica-set binary | `reflexion-server/package.json` `mongodbMemoryServer` config |
| MongoDB production | Atlas replica set, MongoDB `7.0.x` compatible | deployment runbook; do not promote MongoDB 8.x without a compatibility run |

## Vertical-slice tracker

| Slice | Backend contract | Caregiver surface | Mirror surface | State |
| --- | --- | --- | --- | --- |
| Account verification, password reset, setup progress | identity routes, setup persistence | auth/setup routes | n/a | Complete (local) |
| Mirror claiming, assignment, readiness | pairing, assignment, credentials, heartbeat | device setup/status | boot/pairing/readiness | Complete (local contract): credential exchange, assignment, heartbeat, device detail and troubleshooting are wired |
| Conversation to caregiver history | sessions, events, artifacts, outbox, processing | Home, Sessions, Detail | conversation and upload queue | Complete (local contract): Mirror persistence, processing status, bounded session feed and transcript detail are connected |
| Routine reminder and response | care plan, occurrences, response events | routine management/status | spoken reminder and response | Complete (local contract): CRUD, pause/end, materialization, spoken response mapping and caregiver policies are wired |
| Family message delivery | family-message state machine and delivery events | compose/schedule/status | notification/open/replay/dismiss | Complete (local contract): idempotent scheduling, device poll/open/dismiss and caregiver status polling are wired |
| Consent, permissions, retention, deletion | consent gates, audit, retention jobs | consent/privacy | consent and research controls | Complete (local contract): server consent gates, Mirror consent capture, Care Circle roles, scoped deletion, memory cleanup and resumable object cleanup are wired; account deletion is explicitly support-managed in the pilot |

## Cross-slice invariants

- `/api/v1` is the only new client contract; legacy routes are adapters until their sunset.
- Every patient-scoped read/write is authorized by `tenantId`, `patientId` and an active relationship or device assignment.
- MongoDB stores structured business facts and processing metadata; object storage holds consented media; queues/outbox carry durable work.
- Device health is technical state only. Caregiver status is factual, provenance-labelled and never diagnostic.
- All commands are idempotent, retry-safe and restart-safe; client state may be local, but backend state is authoritative.

## First-slice decisions

- Account verification is server-policy controlled by `AUTH_EMAIL_VERIFICATION_REQUIRED` (defaults to `true`).
  When `false` for the pilot, registration issues a session directly while leaving `emailVerifiedAt` null and
  creating no verification email. Existing pending accounts and the hashed, expiring, single-use six-digit
  verification flow remain intact; re-enabling the flag requires those accounts to verify without a rewrite.
- Password reset requests persist a hashed six-digit code with a five-attempt bound. Correct code verification
  mints a one-use reset token, and resetting a password revokes active human sessions.
- Setup has exactly eight v4 categories; Review is a completion stage, not a ninth category. Progress is stored
  per tenant/user with optimistic versioning and idempotency at `/api/v1/setup-progress`.

## Environment and pilot boundary decisions

- Node.js is pinned to `24.18.0` for the backend, caregiver app, Mirror tooling and EAS profiles. The exact
  `geoip-lite@2.0.3` dependency declares `node >=24`; it is not downgraded or replaced with an unmaintained
  compatibility fork. MongoDB Memory Server is pinned to `7.0.14`; production is a MongoDB 7.x Atlas replica set.
- CI runs `npm run test:full`, which expands to the complete server `src/**/*.test.ts` suite (currently 160 tests,
  including the platform/privacy/Care Circle coverage), then audits and builds the same artifact.
- Postmark, Twilio, S3-compatible object storage, Qwen credentials, staging access and physical Mirror hardware are
  external blockers. Local implementations fail closed or retain a visible retry/pending state when those providers
  are unavailable; they do not simulate delivery or deletion.
- The backend still accepts verified country-coded phone identifiers for migrated-client compatibility, but the
  caregiver pilot keeps Phone sign-in visible and routes it to the explicit “not available during the current pilot”
  dialog. Phone password reset and SMS delivery remain provider-gated until Twilio is configured. Google/Apple OAuth
  and paid subscriptions are intentionally deferred from the pilot; their architecture rows remain visible as “not
  configured/not available in this pilot” and no fake success path is exposed.
- Mirror device actions are implemented against the canonical `/api/v1` pairing/readiness/device contracts. A real
  hardware acceptance run still requires a provisioned device and staging credentials.
- Privacy & Data now exposes consent status/history, retention, vendor/cross-border disclosure and selected-data
  deletion. Account deletion is intentionally support-managed in this pilot because shared loved-one records and
  jurisdiction-specific retention obligations require an operator review; the caregiver sees that state and a working
  support contact instead of a fake immediate purge.

## Caregiver authentication and layout invariants

- New and changed passwords are at least 12 characters; legacy password length is not checked at sign-in.
- Phone, Google and Apple controls stay visible but open a pilot-unavailable dialog that returns to email sign-in.
- Raw API/provider validation text never reaches users. Field errors are local and actionable; request errors are
  mapped from stable codes and always explain the next step.
- Pending verification context is secure-storage backed, email-only and restart-safe. Verification copy never
  promises inbox delivery before the configured transactional provider accepts the request; account verification
  is shown only after the server validates the link.
- All caregiver routes use the shared safe-area, keyboard-aware, scrollable layout boundary and pass narrow-screen,
  dynamic-type and text-wrapping checks.
