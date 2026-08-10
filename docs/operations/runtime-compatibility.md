# Runtime compatibility contract

This repository is a multi-package JavaScript project, but it has one supported Node line:

- Node.js **24.18.0** (`24.x`) for local development, the Reflexion server, caregiver/Mirror tooling, CI and EAS
  builds. The root `.nvmrc`, package `engines`, CI `setup-node` step and both mobile `eas.json` files carry this pin.
- MongoDB **7.0.14** for the local/CI `mongodb-memory-server` replica set. Production must use a MongoDB 7.x Atlas
  replica set with backups and restore testing until a later version has passed the full suite.

`geoip-lite@2.0.3` is intentionally pinned exactly. Its package metadata requires Node `>=24.0.0`; the safe fix is
the supported runtime alignment above, not a forced dependency downgrade or an engine-ignore flag.

The server workflow runs `npm ci --include=dev`, `npm run typecheck`, `npm run test:full`, `npm audit` and `npm run build`.
`test:full` is the complete `src/**/*.test.ts` suite rather than a shortened smoke subset.

The checkout has no staging credentials or provider configuration. Postmark (email), Twilio (SMS), S3-compatible
object storage, Qwen credentials and physical Mirror hardware must be supplied by deployment. Local tests use
deterministic stubs and preserve explicit pending/retry/error states when those integrations are unavailable.

## Dependency audit boundary

The server lockfile is clean (`npm audit`: **0 vulnerabilities**). The mobile/admin lockfiles were repaired with
the non-breaking `npm audit fix --package-lock-only` path. The remaining advisories are transitive toolchain issues:

- caregiver: **10 moderate**, whose proposed fix replaces the Expo 56 toolchain with Expo 46;
- Mirror: **24** (11 moderate, 12 high, 1 critical), whose proposed fixes replace the Electron builder/runtime or
  Expo toolchain;
- admin: **4** (3 moderate, 1 high), whose proposed fix upgrades Vite to 8 and needs a separate compatibility pass.

Those forced upgrades are intentionally deferred rather than applied blindly. They must be handled as dedicated
toolchain migrations, with native builds and smoke tests, before release. No production server deploy should proceed
with a client build that has not completed that review.
