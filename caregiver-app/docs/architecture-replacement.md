# Reflexion caregiver app — implementation map

> This is an implementation map, not the product authority. The canonical architecture and UI references are copied under `/Users/chloetan/Documents/Reflexion/docs/product-source-of-truth/caregiver/`; start with `/Users/chloetan/Documents/Reflexion/docs/product-source-of-truth/README.md` and the migration screen matrix.

The approved architecture and PNG references define the product and visual language: mostly white surfaces, teal and muted green, rounded cards, editorial headings, botanical details, Mirror and Bear illustrations.

## Canonical navigation

After setup the app has exactly four tabs: Home, Activity, Chat and Settings. The complete 73-screen map is maintained in `src/architecture/screenInventory.ts`; dynamic routes reuse the same screen from setup and Settings.

The seven functional setup categories are Household, Pair Device, Language & Accessibility, Routines and Reminders, Notifications, Older-Adult Consent & Control, and Research Participation. Review and Complete are stages, not categories. Generic category detail, filter, confirmation, status and media-upload states are not extra top-level routes.

## Current route map

| Area | Canonical implementation | Integration boundary |
| --- | --- | --- |
| Authentication | `/splash`, `/sign-in`, `/forgot-password`, `/reset-verification`, `/reset-password`, `/create-account`, `/account-verification` | Existing v1 identity and secure pending-verification flow |
| Setup | `/welcome`, `/setup`, `/setup/household`, `/setup/household-review`, `/device/select`, `/device/[id]/[stage]`, `/settings/language`, `/settings/voice-preview`, `/settings/routines`, `/settings/notifications`, `/settings/consent`, `/setup/review`, `/setup/complete` | Existing v1 patient, care-plan, device, routine, notification and consent contracts |
| Loved one | `/loved-one/[id]`, `/sessions`, `/sessions/[sessionId]`, `/sessions/[sessionId]/conversation`, `/weekly-summary`, `/trends`, `/history`, `/export` | Existing v1 status, sessions, transcripts, trends, routines, messages and device state |
| Activity | `/(tabs)/activity`, `/activity/[eventId]` | Timeline is assembled from existing sessions, routine occurrences, messages and device assignments; filters are an inline sheet |
| Chat | `/(tabs)/chat`, `/chat/[id]`, `/compose`, `/photo`, `/voice`, `/preview`, `/status/[messageId]` | Text delivery and opened status use the existing Mirror family-message contract; media screens remain truthful until media APIs exist |
| Settings | Account, App Language, Loved Ones, edit profile, Away Mode, Routines, Devices, device detail, Language & Accessibility, Notifications, Consent, Privacy, Research, Help, Contact Support, Feedback, Subscription, Payment Method, About | Existing v1 contracts where available; unsupported billing/research/media actions are explicit unavailable states |
| Research | `/research/overview`, `/research/study`, `/research/confirmation`, `/settings/research` | Existing separate research-consent purpose; no study-specific invitation endpoint exists yet |

## Shared information model

`src/architecture/models.ts` owns the approved objective interaction, device, setup, routine response, notification, consent, message and screen-state vocabulary. `src/lib/v1Status.ts` translates legacy backend status values without exposing wellness claims. Device status is rendered separately from loved-one interaction state.

The baseline rule is at least 3 valid sessions in a rolling 14-day window. No caregiver-facing screen says “doing well”, “healthy”, “safe” or “happy”.

## Preserved backend contracts

The v1 adapter keeps the production API base URL, authentication, field names, endpoint paths, device identifiers and message states. Existing account authentication, pairing/claiming, loved-one assignment, Wi-Fi and technical/audio checks, sessions and summaries, routines and reminder responses, text messages, delivery/opened status, product/research consent and technical online/offline state remain connected.

The production v1 family-message endpoint currently accepts `type: text`/`body` only. Photo and voice UI routes therefore never upload or fake delivery. The current v1 read model also exposes 7- and 30-day trends, not a 3-month trend, and has no PDF export or study-invitation data contract; the UI states those limitations rather than inventing data.

## Obsolete-route policy

The former generic setup category route, generic Settings section route, Care Circle route, loved-one `[view]` placeholder route and full-screen Activity filter are redirect bridges only. They do not render obsolete controls or become additional canonical screens.
