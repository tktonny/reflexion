# Fresh safety snapshot — 2026-08-10

Captured before the final consolidation/build phase at 2026-08-10 15:13 +0800. Both legacy roots were read only. No legacy file was modified, deleted, renamed, cleaned, or moved.

## Legacy Git state

### `/Users/chloetan/Documents/Reflexion app`

- Git root: `/Users/chloetan/Documents/Reflexion app`
- Remote: `origin https://github.com/tktonny/reflexion.git`
- Branch: `chloe/caregiver-mirror-update-2026-08-10`
- HEAD: `a6a4973a11e5d970be96a0cd683424dad179f68a`
- Staged changes: none.
- Unstaged tracked changes:
  - `reflexion-server/src/v1/care/reminderScheduler.ts`
  - `reflexion-server/src/v1/integration/phase3.api.integration.test.ts`
  - `reflexion-server/src/v1/integration/platform.api.integration.test.ts`
  - `reflexion-server/src/v1/routes/carePlan.ts`
  - `reflexion-server/src/v1/routes/caregiverHistory.test.ts`
  - `reflexion-server/src/v1/routes/caregiverHistory.ts`
  - `reflexion-server/src/v1/routes/consentGate.test.ts`
  - `reflexion-server/src/v1/routes/devices.ts`
  - `reflexion-server/src/v1/routes/identity.registration.test.ts`
  - `reflexion-server/src/v1/routes/identity.ts`
  - `reflexion-server/src/v1/routes/patients.ts`
  - `reflexion-server/src/v1/routes/privacyCareCircle.test.ts`
  - `reflexion-server/src/v1/routes/setupProgress.ts`
- Untracked source/config input: `caregiver-app/.env.production`.
- Total untracked paths: 63,817; the remainder is generated `.gradle-cache` content and was excluded from the source list.
- Environment-file names (contents not read):
  - `caregiver-app/.env`
  - `caregiver-app/.env.example`
  - `caregiver-app/.env.production`
  - `mirror-app/.env.example`
  - `mirror-app/.env.server.example`
  - `reflexion-server/.env`
  - `reflexion-server/.env.example`

### `/Users/chloetan/Documents/Reflexion Mirror`

- Git root: `/Users/chloetan/Documents/Reflexion Mirror`
- Remote: `origin https://github.com/tktonny/reflexion.git`
- Branch: `codex/mirror-eas-ota-baseline`
- HEAD: `8724437aa68e3ccde2303e6ceaee22960a3e2417`
- Staged changes: none.
- Unstaged tracked changes: 56 paths, including current caregiver, Mirror, server, EAS, pairing, diagnostics, orchestration, audio, and API changes.
- Untracked source/config input:
  - Caregiver: `app/chat/[id].tsx`, `app/demo.tsx`, `app/family-messages/[id].tsx`, `src/demo/demoConfig.ts`, `src/demo/demoRepository.ts`, `src/lib/buildInfo.ts`.
  - Mirror: `app/consent.tsx`, `app/demo.tsx`, `app/family-message.tsx`, `app/paired-success.tsx`, `app/research.tsx`, `app/routine.tsx`, `app/status.tsx`, `app/wifi-setup.tsx`, `src/api/familyMessages.ts`, `src/components/mirror/MirrorChrome.tsx`, `src/components/mirror/MirrorIcon.tsx`, `src/components/mirror/WifiSetupView.tsx`, `src/demo/demoCheckinController.ts`, `src/demo/demoConfig.ts`, `src/demo/demoRepository.ts`, `src/demo/demoWebAudio.ts`, `src/lib/buildInfo.ts`, `src/lib/pairingDiagnostics.ts`.
  - Server: `src/v1/routes/familyMessages.ts`, `src/v1/routes/familyMessages.test.ts`.
- Environment-file inventory: no `.env` files were present outside generated/dependency exclusions.

## Important ignored native inputs

- Both legacy caregiver Android trees exist and are ignored by `caregiver-app/.gitignore` at `/android`.
- The legacy Mirror Android tree exists and is ignored by the root `.gitignore` at `mirror-app/android/`.
- Legacy Mirror `android/local.properties` was present and ignored; its contents were not read.
- No legacy `google-services.json` was present at the checked path.

## Existing APK checksums

- `/Users/chloetan/Documents/Reflexion app/caregiver-app/android/app/build/outputs/apk/release/app-release.apk`
  - SHA256: `1368b7f2d78939bfa45595ffa389ad5732fe958dff75696fa9fb7ad73f0f2053`
  - Size: 129,611,326 bytes
  - Modified: 2026-08-10 14:03:11 +0800
- `/Users/chloetan/Documents/Reflexion Mirror/mirror-app/android/app/build/outputs/apk/release/app-release.apk`
  - SHA256: `7c672487a4857848d23502a8b533f3835e68541f81fe08cae958de668f9826ce`
  - Size: 101,399,276 bytes
  - Modified: 2026-08-10 13:58:44 +0800

These APKs are not canonical outputs and will not be copied into `dist-apks/`.

## Canonical repository state after snapshot

- Root: `/Users/chloetan/Documents/Reflexion`
- Branch: `chloe/reflexion-canonical-consolidation`
- HEAD: unborn; no commit created.
- Remote: `origin https://github.com/tktonny/reflexion.git`
- No fetch, commit, push, deployment, or release action performed.
