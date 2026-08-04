# Reflexion caregiver app — replacement architecture

This file translates the Final Reflexion Caregiver-App Architecture into the implementation boundary for this app. The architecture document is the product source of truth; the visual references define only the visual language.

## Replacement route map

| Area | Routes | Notes |
| --- | --- | --- |
| Authentication | `/`, `/sign-in`, `/forgot-password`, `/reset-verification`, `/reset-password`, `/create-account`, `/account-verification` | The public flow; no setup shortcut before account creation. |
| First-time setup | `/welcome`, `/setup`, `/setup/[category]`, `/setup/review`, `/setup/complete` | Eight independent categories: household, device, accessibility, routines, notifications, consent, care-circle, review. |
| Main navigation | `/(tabs)/home`, `/(tabs)/activity`, `/(tabs)/chat`, `/(tabs)/settings` | Exactly Home, Activity, Chat and Settings. |
| Loved-one dashboard | `/loved-one/[id]`, `/loved-one/[id]/sessions`, `/loved-one/[id]/session/[sessionId]`, `/loved-one/[id]/weekly-summary`, `/loved-one/[id]/trends`, `/loved-one/[id]/history`, `/loved-one/[id]/export` | Today, sessions, detail, weekly summary, trends, history and export. |
| Device flow | `/device/[id]`, `/device/[id]/pairing`, `/device/[id]/wifi`, `/device/[id]/test`, `/device/[id]/placement`, `/device/[id]/success`, `/device/[id]/troubleshooting` | Mirror, Bear, App and Other supported device only. |
| Activity | `/activity/filter`, `/activity/[eventId]` | Filters: person, date range, conversations, routines, chat, technical and caregiver actions. |
| Chat | `/chat/[lovedOneId]`, `/chat/[lovedOneId]/compose`, `/chat/[lovedOneId]/preview`, `/chat/[lovedOneId]/status/[messageId]` | Text, photo or voice; now or specific date/time; no replies or requests. |
| Settings | `/settings/account`, `/settings/notifications`, `/settings/household`, `/settings/accessibility`, `/settings/routines`, `/settings/devices`, `/settings/away`, `/settings/consent`, `/settings/care-circle`, `/settings/privacy`, `/settings/help`, `/settings/feedback`, `/settings/subscription`, `/settings/about` | Settings mirrors onboarding options where the feature overlaps. |

## Shared data models

`src/architecture/models.ts` owns the product vocabulary for interaction, device, setup, routine response, notifications, consent, Care Circle, messages, screen states and baseline rules. Screens must not define parallel enums or new status strings.

## Shared component inventory

- `AppHeader`, `BrandLockup`, `BotanicalCorner`
- `PrimaryButton`, `SecondaryButton`, `TertiaryButton`, `InputField`
- `ChoiceCard`, `RadioRow`, `StatusPill`, `SetupProgressCard`
- `LovedOneCard`, `ConfigurationBanner`, `ProvenanceSection`
- `ActivityRow`, `MessageBubble`, `MessageStatusTimeline`, `SettingsRow`
- `LoadingState`, `EmptyState`, `ErrorState`, `OfflineState`

## Design tokens

`src/theme.ts` is the single token system. It uses the architecture’s warm ivory, card white, navy, teal, state colours, controlled radii, 4 px grid-derived spacing and mobile type scale. All accessible text and touch-target constraints are checked through the app’s existing token checks.

## Obsolete implementation to replace

- The four-step onboarding route and its account/patient/mirror/notification-only state.
- Old Mirror-management naming and its server-specific connected-card screen.
- The generic Settings `care preferences` prototype route.
- Any legacy session/trend/history naming at top level; these become person-specific dashboard destinations.
- Alert-era navigation and any state labels not declared in the Final architecture.
- Notification options that differ from the architecture or introduce quiet hours.
- Any routine wording that treats a reported response as independently verified.
- Any reply, request or "Mark followed up" flow.

Existing API clients are intentionally not treated as product architecture. They can be integrated only when they support these models and states without changing copy, flow or available options.

## V4 authentication and responsive-layout rules

- New registrations, password resets and password changes require at least 12 characters. Sign-in never
  applies that rule to an existing account, so migrated accounts with a shorter legacy password remain usable.
- Phone, Google and Apple sign-in remain visible and tappable during the pilot. Each opens a truthful
  pilot-unavailable dialog with one action, **Continue with email**; no partial provider flow is started.
- Raw backend, schema, provider or database messages never reach a caregiver. Client validation and stable
  error codes map to field-level, actionable copy; request-wide failures use a clear retry instruction.
- The pending verification context persists only the email and timestamp in secure storage. It survives
  navigation and restart, never stores a plaintext password, and is cleared after the server verifies the link.
- Verification and resend screens do not claim inbox delivery. A delivery-request success is shown only for a
  provider-acceptance signal, while account verification itself is shown only after the server validates the link;
  an unavailable or rejected transactional-email provider is a visible retryable error.
- Every screen uses the shared `ScreenLayout` and `layout` tokens for safe areas, horizontal boundaries,
  keyboard avoidance and scrolling. Text may wrap at increased system font sizes; fixed heights and truncation
  are not used for headings, legal copy, fields, cards or primary actions.
