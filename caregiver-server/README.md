# Reflexion Platform API

The self-hostable Reflexion backend shared by the Caregiver app and Mirror devices. The `/api/v1` surface is the production contract. Old routes remain in source for migration only and are disabled unless `ENABLE_LEGACY_API=true` is set explicitly.

## Implemented platform boundary

- Human identity: password sessions, 15-minute access tokens, rotating refresh tokens, logout and password reset.
- Device identity: factory provisioning, six-digit pairing, one-time credential exchange, rotation, revocation, configuration and heartbeat telemetry.
- Patient and care data: tenant-scoped patients, care relationships, consent, care plans, medication plans, reminder occurrences and caregiver tasks.
- Conversation ingestion: session state machine, short-lived Qwen realtime tickets, ordered/idempotent event batches, direct-to-object-store upload plans, verification, completion and stable processing-status polling.
- Monitoring: transcript-derived structured features, explicit quality/identity/consent gates, 14-day operational baseline, 12-session/28-day research baseline, optional versioned embeddings, robust anomaly scoring, persistence rules and provider review cases.
- Assistant tools: Open-Meteo weather, optional Brave web search, medications and upcoming reminders behind a server allowlist.
- Notifications: relationship-authorized in-app feed with review-case delivery deduplication.
- Reliability: Mongo-backed idempotency records, transactional outbox, retry/dead-letter worker, visible processing failures and deterministic index bootstrap.

The live Mirror never receives account-level Qwen, search, database, object-store or embedding keys. It receives a short-lived Qwen ticket only after device authentication and backend session creation.

## Runtime topology

Run these as separate processes from the same build artifact:

1. `npm start` — HTTP API.
2. `npm run start:worker` — outbox, artifact verification and longitudinal processing worker.
3. `npm run reminders:materialize` from a scheduler — materializes upcoming reminder occurrences.

MongoDB must be a replica set (a single-node replica set is sufficient for development) because pairing, password reset, device credential exchange and session completion use transactions. Configure optional S3-compatible storage and an embedding provider independently; transcript-only ingestion and scalar monitoring still work when embeddings are disabled or temporarily unavailable.

## Local setup and verification

```bash
cp .env.example .env
npm ci
npm run typecheck
npm test
npm run test:coverage
npm run test:coverage:phase3
npm run build
```

`test:integration` runs the complete Phase 3 HTTP flow against an ephemeral real MongoDB replica set. It covers human and device authentication, Pairing v2, credential rotation, heartbeat, session ingestion, Qwen ticket minting, signed artifact upload plans, optimistic completion, outbox processing and retry recovery. `test:coverage:phase3` enforces at least 90% aggregate line and function coverage over the Phase 3 routes and platform boundary.

Initialize the database and first tenant administrator:

```bash
npm run db:indexes
npm run bootstrap:admin -- --email=admin@example.com --password='use-a-long-unique-password' --tenant='Reflexion'
```

Provision each physical Mirror with its unique hardware serial:

```bash
npm run provision:device -- --serial=RF-MIRROR-000001 --hardware=v1 --software=1.0.0
```

The command prints a device ID and a time-limited bootstrap token. Install that token into the matching device's secure provisioning channel; do not reuse one bootstrap token across mirrors and do not put any provider key in an APK.

Start the API and worker in separate shells:

```bash
npm start
npm run start:worker
```

For a one-shot worker diagnostic:

```bash
node dist/jobs/outboxWorker.js --once
```

After deploying a new HTTPS origin, verify that it is the compiled v1 service rather than a legacy health-only deployment:

```bash
npm run smoke:deployment -- --base=https://api.example.com
```

## Configuration

See [`.env.example`](./.env.example). Required for the core API:

- `MONGODB_URI`
- `JWT_SECRET`
- `PAIRING_PEPPER`
- `CREDENTIAL_ENCRYPTION_KEY`
- `QWEN_API_KEY` for live conversations

Optional integrations:

- `OBJECT_STORE_*` for audio/video/image artifacts.
- `EMBEDDING_*` for versioned semantic embeddings. If omitted, robust scalar monitoring remains active.
- `BRAVE_SEARCH_API_KEY` for `web.search`; weather uses Open-Meteo without a key.
- `EMAIL_PROVIDER=postmark` plus Postmark variables for password reset email.

In production, configure HTTPS at the reverse proxy/load balancer, allow only the caregiver web origin in `CORS_ALLOWED_ORIGINS`, keep secrets in the server's secret manager, and run `db:indexes` during controlled rollout.

## API contract and design

- OpenAPI: [`../docs/architecture/reflexion-api-v1.openapi.yaml`](../docs/architecture/reflexion-api-v1.openapi.yaml)
- API/domain design: [`../docs/architecture/platform-v2-api-and-domain-architecture.md`](../docs/architecture/platform-v2-api-and-domain-architecture.md)
- Database design: [`../docs/architecture/platform-v2-database-design.md`](../docs/architecture/platform-v2-database-design.md)
- Longitudinal monitoring design: [`../docs/architecture/longitudinal-vector-anomaly-design.md`](../docs/architecture/longitudinal-vector-anomaly-design.md)

All `/api/v1` success responses use `{ data, meta: { requestId } }`. Errors use `{ error: { code, message, retryable, details }, meta }`. Mutating creation/command routes accept `Idempotency-Key`; versioned resources use `If-Match` where required.
