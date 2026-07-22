# Phase 3 server deployment runbook

## Required topology

Deploy one build artifact as three independent workloads:

1. HTTP API: `npm start`
2. Long-running outbox worker: `npm run start:worker`
3. Scheduled reminder materialization: `npm run reminders:materialize`

The API and worker must use the same environment, MongoDB database and object-store configuration. Do not run the outbox worker inside a request-driven/serverless process.

## Infrastructure prerequisites

- Node.js 22 runtime behind HTTPS/TLS.
- MongoDB replica set with backups and restore testing.
- S3-compatible encrypted object storage for consented image/audio/video artifacts.
- Qwen API key and endpoint from the same enabled region.
- DNS for the final unified API origin.
- Independent 32+ character `JWT_SECRET`, `PAIRING_PEPPER` and `CREDENTIAL_ENCRYPTION_KEY` values in a secret manager.
- Process supervision and alerts for API failures, worker lag, retry/dead-letter events and provider failures.

Use `caregiver-server/.env.example` as the variable inventory. Never copy real secrets into source control, an APK or a browser bundle.

## Release order

```bash
npm ci
npm run typecheck
npm test
npm run test:coverage
npm run test:coverage:phase3
npm run build
npm run db:indexes
```

Start the API and worker from the same release. Run the reminder command from the platform scheduler. Only after both API and worker are healthy should DNS/client configuration move to the new origin.

Create the first tenant administrator and provision each physical mirror through the one-time CLI workflows:

```bash
npm run bootstrap:admin -- --email=admin@example.com --password='use-a-long-unique-password' --tenant='Reflexion'
npm run provision:device -- --serial=RF-MIRROR-000001 --hardware=v1 --software=1.0.0
```

## New URL smoke test

After DNS and TLS are active, run the compiled deployment probe:

```bash
npm run smoke:deployment -- --base=https://api.example.com
```

It verifies:

- `/health` returns 200 from the compiled service;
- security headers are active and the Express fingerprint is absent;
- `/api/v1/me` reaches v1 authentication and returns the standard 401 envelope;
- unknown v1 routes return the stable `ROUTE_NOT_FOUND` envelope.

Then use a staging patient/device to run the authenticated acceptance sequence: provision → Pairing v2 → credential exchange → heartbeat → session create → Qwen ticket → event batch → artifact upload/commit → complete → worker processing → processing-status completed.

## Client cutover

Set the same new API origin for Mirror and Caregiver builds. Rebuild the release APK; do not depend on an old origin compiled into an existing APK. Keep the previous release and database-compatible server artifact available until the authenticated sequence and worker processing pass on the new URL.

## Rollback

Rollback the API and worker together to the previous artifact. Do not roll back database collections destructively. Stop client cutover first, inspect pending/retry/dead-letter outbox events, and replay only after the compatible worker version is restored.
