# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

REFLEXION is a daily reassurance companion for Singaporean elderly: a smart-mirror device runs a short spoken daily check-in (Qwen omni realtime voice), a backend computes a deterministic reassurance status, and a caregiver app answers "is Mum okay today?". It is explicitly **not diagnostic** only advise — never surface a cognitive score or clinical wording only store in backend. Data flows one direction: Mirror raw signals → backend status engine → caregiver display. The mirror never computes status; the caregiver app never recomputes it.

**Canonical reference: `docs/ARCHITECTURE-AND-API.md`** (three apps, all end-to-end flows, every v1 endpoint, full collection map, deployment topology). The legacy Python clinic platform now archived in `_archived/` (dead code, reference only).

## Repo layout

Plain multi-package repo — **no root package.json, no workspaces, no Makefile**. Run every command from inside the specific package directory.

- `reflexion-server/` — Express + TypeScript ESM + raw MongoDB driver (no ODM). The single backend for all clients. One `tsc` build serves multiple processes: HTTP API, outbox worker, scheduled jobs.
- `mirror-app/` — Expo SDK 56 / RN Android smart-mirror app (`com.reflexion.mirror`), plus a Node relay under `server/` (web-dev diagnostic only) and a local Python wake-word training pipeline under `wakeword-training/`.
- `caregiver-app/` — Expo/RN caregiver app (iOS + Android).
- `admin-web/` — Vite + React operator SPA (patients onboarding, users, support threads), v1 API only.
- `docs/` — canonical doc plus `operations/phase3-server-deployment.md` (deploy runbook), `reflexion-implementation-baseline.md` (frozen decisions + mirror→backend upload contract), `releases/` (production facts), `architecture/`, `mirror-app/`.
- `hardware/SoundRecorder/` — the **system sound-recorder on the real Mirroh mirror hardware**, vendored here as AOSP platform sources (Soong `Android.bp`, `CleanSpec.mk`, `OWNERS`, `NOTICE`; 8 Java files under `src/com/android/soundrecorder/`). It is **stock, unmodified AOSP** `com.android.soundrecorder` (Google OWNER, zero Reflexion/Mirroh code) — a privileged system app that records to on-device files and registers the `MediaStore.RECORD_SOUND` intent. It is compiled into the **device system image** by the platform/AOSP build, not by any npm/Gradle flow in this repo, and there are no run/test commands for it here. **Not the check-in capture path**: the daily check-in streams realtime PCM through `mirror-app/modules/expo-pcm-audio` (16 kHz mono → Qwen), not this OS-level recorder — keep the two separate when reasoning about mic/audio behavior.

There is **no CI, no linter, no formatter** anywhere. The only static check is `npm run typecheck` (`tsc --noEmit`); caregiver-app doesn't even have that script (use `npx tsc --noEmit`). Do not invent `npm run lint`.

## Commands

### reflexion-server

```bash
npm run dev                # tsx watch, port 3001
npm run build              # tsc → dist/ (single artifact for API + worker + jobs)
npm run typecheck
npm test                   # node:test via tsx, mongodb-memory-server (no external Mongo needed; first run downloads a Mongo binary)
npm run test:integration   # src/v1/integration only (real ephemeral replica sets)
node --import tsx --test src/v1/path/to/file.test.ts   # single test file
npm run test:coverage      # 90% line/function gates (also :phase3, :api variants)
npm run db:indexes         # ensure Mongo indexes (part of release order)
```

Release-order gate (from the runbook): `npm ci → typecheck → test → coverage → build → db:indexes`.

Operational CLIs: `npm run bootstrap:admin -- --email=... --password=... --tenant=...` (first tenant admin), `npm run provision:device -- --serial=... --hardware=v1 --software=1.0.0` (mint per-device bootstrap token), `npm run migrate:legacy-v1`, `npm run smoke:deployment -- --base=<origin>`.

### mirror-app

```bash
npm run web                # Expo web dev at :8081 — open /realtime-test to exercise the voice pipeline in a browser
npm run android            # native dev build to a connected device (required for ws/webrtc modes, PCM audio, wake word)
npm run typecheck
npm run test:turn-taking   # esbuild-bundle + node --test (only server/turn-taking.test.ts — two more node:test suites exist in src/orchestration/*.test.ts with no npm script wired to them)
npm run relay              # build src/orchestration bundle + start Node relay on :8787 (reads .env.server.local)
node --env-file=.env.server.local server/smoke.mjs   # headless relay→Qwen check, no mic (more smoke-*.mjs alongside)
cd android && ./gradlew assembleRelease              # signed release APK (fails by design without REFLEXION_MIRROR_* signing env/Keychain)
```

Wake-word retraining lives in `wakeword-training/` (own Python venv, fully local Apple-Silicon pipeline) — see its README; the custom "Hello Aria" model is already swapped into `assets/wakeword/wakeword.onnx`.

### caregiver-app

```bash
npm install                # postinstall patches @expo/cli so expo run:android launches the right activity — required
npm run dev                # expo start
npm run android / ios
npx tsc --noEmit           # no typecheck script; no tests
```

### admin-web

```bash
VITE_DEV_API_TARGET=http://localhost:3001 npm run dev   # port 5174; /api proxied to the server (env var must be a shell var — vite.config.ts reads process.env, .env files won't reach it)
npm run typecheck
VITE_API_BASE_URL=https://reflexion.production.tktonny.top npm run build   # API origin is baked in at build time; tsc gates the build
```

## Architecture

### One backend, one MongoDB, three clients

All three clients hit reflexion-server. Two API surfaces coexist:

1. **`/api/v1` — the canonical contract.** Envelope `{data, meta}` / `{error: {code, message, retryable}, meta}`. Most resource-creating routes (sessions, pairings, event batches, flags, ...) require an `Idempotency-Key` header, enforced per route via `executeIdempotent` (replay with a different body → 409) — but not all mutations do (auth/identity routes and `device-credentials/exchange` don't); versioned mutations (patient PATCH, session complete, care-plan PUT) require `If-Match`. Routes live in `reflexion-server/src/v1/routes/`, cross-cutting machinery (tokens, auth, idempotency, outbox, object store, collections registry) in `src/v1/platform/`.
2. **Legacy routes, mounted bare at the server root** (`POST /auth/sign-in`, `/nurse-patient-config/*`, `/conversation-session*`, ... — no `/api` prefix on the wire except `/api/qwen-token`; the caregiver client strips the `/api` its own paths carry) — tokenless (identity = `nurseId` in query/body), mounts only when `ENABLE_LEGACY_API=true`, sunset 2026-12-31. The caregiver app still depends on legacy sign-in and `nurse-patient-config`; parts are already adapters over v1 collections (`src/lib/v1Conversations.ts`, `legacyV1Bridge.ts`) — see `reflexion-server/LEGACY_V1_ADAPTER.md`. The no-auth trust model is a deliberate transition state: never copy it into v1, and never disable legacy before the caregiver app migrates.

### Auth: three token kinds, DB-liveness on every request

Hand-rolled HS256 JWT (`src/v1/platform/tokens.ts`); `requireActor()` verifies the JWT **and** live DB state per request:

- **human** — email/password → 15-min access + rotating 30-day refresh (`auth_sessions`).
- **device** — 15-min access + 90-day rotating refresh credential; valid only while both `device_credentials` AND `device_assignments` are active. Revoking either instantly kills a still-valid JWT — that's how caregiver unlink works.
- **bootstrap** — per-device factory token (30-day TTL — a provisioned mirror must pair within 30 days) from `provision:device`, sent in the **`X-Device-Bootstrap` header** (not `Authorization`); can only create/poll pairings and exchange credentials.

**Pairing flow:** mirror (bootstrap) `POST /device-pairings` → 6-digit code + QR → caregiver claims it → mirror polls, receives a one-time 5-min exchange ticket → `POST /device-credentials/exchange` → rotating device credential stored in SecureStore. Provider keys never reach a device: for each conversation the mirror creates a backend session and gets a short-lived Qwen realtime ticket via `POST /sessions/:id/realtime-tickets`.

### Session lifecycle and status engine

Realtime audio streams **directly mirror ↔ Qwen** (DashScope omni realtime WS, `qwen3.5-omni-*` series — required for `semantic_vad`, which rejects speaker echo at turn-detection level). The backend only mints the ticket and ingests afterwards: session create → event batches (≤100, idempotent) → artifact upload-plans/commit → `complete` (If-Match) → 202 + outbox event → **the worker** runs the monitoring pipeline (gates → features → operational baseline median+MAD+EWMA) → pure evaluator `evaluateReassuranceStatus` → `establishing | doing_well | worth_checking | needs_attention`. Consequences:

- A finished check-in sits in `ingesting` until the outbox worker runs — never test only for `state === 'completed'`; run `node dist/jobs/outboxWorker.js --once` or the worker process.
- Failed completion uploads on the mirror go to a durable offline queue (`src/storage/conversationQueue.ts`) and retry later.
- Companion (non-check-in) sessions never move the baseline; the longitudinal research path runs in shadow isolation and never feeds caregiver status colour.

### Mirror conversation transports

`EXPO_PUBLIC_CONVERSATION_MODE` (build-time, default `relay` — see `mirror-app/src/config/conversationMode.ts`): `relay` = browser↔Node relay↔Qwen (web dev only), `http` = turn-based ASR→chat→TTS, `ws` = **production**: direct authenticated WS to DashScope with the turn-based hook as warm standby on failure (`useResilientConversation`), `webrtc` = direct MaaS endpoint (needs workspace-scoped URL). The relay under `mirror-app/server/` is not part of the Android release. The Qwen account is China-region (`dashscope.aliyuncs.com`); the relay retries the China host on 401/403 from intl.

### MongoDB shared contract

Same database for v1 (snake_case, names centralized in `src/v1/platform/collections.ts`) and legacy (PascalCase: `NursePatientConfig`, `MirrorPairingSessions`, ...). v1 ids are opaque prefixed strings (`pat_`, `dev_`, `ses_`, ...), but records migrated from legacy reuse 24-hex ObjectId strings — **treat all ids as opaque strings, never ObjectId**. Prod DB is `reflexion_production` on Atlas; local default is `ref` (`MONGODB_DB`).

### Deployment (production)

Aliyun ECS + BT-Panel + nginx. Two pm2 processes from one build: `reflexion-api` (`node dist/index.js`, port 3001, behind `https://reflexion.production.tktonny.top`) and `reflexion-worker` (`node dist/jobs/outboxWorker.js` — must be its own long-lived process with the same `.env`, never inside a request). Admin SPA is a static site at `admin.reflexion.production.tktonny.top` (cross-origin: backend `CORS_ALLOWED_ORIGINS` must include it; host rewrites unknown routes to index.html). Mobile releases are **local Gradle builds, not EAS**; APKs land in gitignored `dist-apks/`. Gradle pins differ per app (mirror 8.14.3 for onnxruntime, caregiver 9.3.1) — don't blindly upgrade.

## Critical constraints

- **Secrets/env**: every package's `.env` is gitignored (`.env.example` is the template); the backend `.env` exists only on the prod server — it has been lost once, treat it carefully. `JWT_SECRET`, `PAIRING_PEPPER`, `CREDENTIAL_ENCRYPTION_KEY` must each be ≥32 chars or `requireServerSecret()` throws at runtime.
- **`EXPO_PUBLIC_*` values are compiled into the APK.** Never put `QWEN_API_KEY`/`DASHSCOPE_API_KEY`/`MONGODB_URI` in a mobile `.env`; server-only keys go in `mirror-app/.env.server.local` (read only by the relay/smoke scripts via `node --env-file`). Each release APK embeds the API origin and a **per-device** bootstrap token — never share one bootstrap token across devices. A mirror build without `EXPO_PUBLIC_API_BASE` fails closed to unreachable `http://127.0.0.1:9`.
- **MongoDB must be a replica set** (single-node is fine for dev): pairing claim, credential exchange, and session completion use multi-document transactions.
- **Debugging the backend usually means querying Mongo directly**: 4xx business errors (e.g. `EXCHANGE_TICKET_INVALID`) are returned but never logged — inspect `audit_events`, `outbox_events`, `idempotency_records`.
- **caregiver-app invariants** (documented in-file, learned from crashes): the `AuthGate` in `app/_layout.tsx` must never re-gate whether `<Stack>` renders after first hydration (caused "Reflexion keeps stopping"). Legacy sign-in is primary and v1 login is best-effort by design — don't make v1 required. React Query is configured to never auto-refetch (`staleTime: Infinity`) — screens refetch explicitly via `useFocusEffect`/`invalidateQueries`. The two URL builders are incompatible: legacy `getApiUrl()` strips a leading `/api`, `getV1Url()` appends `/api/v1`.
- **Status wording is a product decision**: the caregiver app only renders the server-computed v1 status read model; reason codes map to warm non-clinical English, device problems are framed as connection issues, `establishing` is never shown as red.
- Wake word only works on a real device (emulator mic is silent) and degrades gracefully to tap-to-start; tuning is via `EXPO_PUBLIC_WAKE_WORD_*` env + APK rebuild.
- v1 sign-in matches `users.emailNormalized` (lowercased) — any user-creation path must write it or login 401s despite a correct password.

## Conventions

Conventional Commits with package/subsystem scope (`fix(mirror-app): ...`, `feat(admin-web): ...`, `docs:`, `chore:`); occasional Chinese in subjects is normal. Branches `type/short-kebab-desc`, merged to `main` via GitHub PRs.
