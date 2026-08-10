# Reflexion — canonical project root

This is the one canonical Reflexion project:

`/Users/chloetan/Documents/Reflexion`

It contains the caregiver app, Mirror app, backend, hardware sources, deployment scripts and the copied product source of truth. Do not edit or build from the legacy duplicate folders:

- `/Users/chloetan/Documents/Reflexion app`
- `/Users/chloetan/Documents/Reflexion Mirror`

Read [`AGENTS.md`](./AGENTS.md) and [`docs/product-source-of-truth/README.md`](./docs/product-source-of-truth/README.md) before substantial product work.

## Canonical subprojects

- `caregiver-app/` — caregiver Expo/React Native app, selected from the newer caregiver implementation in the former `Reflexion app` tree.
- `mirror-app/` — Mirror Expo/React Native app, selected from the newer Mirror implementation in the former `Reflexion Mirror` tree, with the required network/Electron pieces merged from the former caregiver tree.
- `reflexion-server/` — shared Express/MongoDB backend, merged selectively from both trees.
- `admin-web/`, `hardware/` — supporting web and device sources.

## Checks and demos

Run commands from the subproject directory under this root.

```bash
cd /Users/chloetan/Documents/Reflexion/caregiver-app
npm ci
npm run typecheck
npm test

cd /Users/chloetan/Documents/Reflexion/mirror-app
npm ci --legacy-peer-deps
npm run typecheck
npm run web:demo
# Static demo export: npm run export:web-demo, then serve mirror-app/dist/

cd /Users/chloetan/Documents/Reflexion/reflexion-server
npm ci
npm run typecheck
npm test
npm run build
```

The canonical Mirror Android release command is:

```bash
cd /Users/chloetan/Documents/Reflexion/mirror-app/android
./gradlew assembleRelease
```

The expected ignored output is `mirror-app/android/app/build/outputs/apk/release/app-release.apk`. Keep release copies, if needed for local testing, under `mirror-app/dist-apks/`; do not use an APK left in either legacy root as a canonical build.

Mirror web-demo output is generated under `mirror-app/dist/`. Backend compilation output is generated under `reflexion-server/dist/`. These are build artifacts, not product source.

## Environment and deployment

Use each subproject’s `.env.example` / `.env.server.example` as the template. Real `.env` files, signing keys and provider credentials must remain local and must never be copied into this canonical repository. Mirror EAS profiles are in `mirror-app/eas.json`; caregiver EAS profiles are in `caregiver-app/eas.json`; the backend deployment workflow is under `.github/workflows/`.

The final-source inventory and every caregiver/Mirror reference mapping are in [`docs/migration/`](./docs/migration/). The source specifications were copied read-only under [`docs/product-source-of-truth/`](./docs/product-source-of-truth/).
