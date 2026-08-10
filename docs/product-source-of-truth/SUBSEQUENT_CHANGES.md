# Subsequent product amendments

This file records explicit or evidence-backed changes that must be applied on top of the approved architecture/UI baseline. It is intentionally limited to requirements established by the current migration evidence, current code/diffs, repository documentation, and the attached product-source instructions.

## Priority rule

These amendments take precedence over conflicting details in the copied baseline documents and PNGs. They do not authorize speculative redesign or restoration of removed legacy screens.

## Mirror voice reply and conversation ending

The later Mirror architecture DOCX, `Reflexion_Mirror_Architecture_UPDATED_voice_reply_end_conversation.docx`, was saved after the Mirror Markdown baseline and explicitly changes the current release:

- Voice-message replies are supported on the Mirror.
- Voice reply is a state/overlay of `MIR-11`, not a new full-screen route.
- The loved one records, reviews, and explicitly taps **Send reply**.
- A reply is linked to the original family message and appears in the caregiver family thread.
- Failed sending offers Retry/Cancel without duplicate replies.
- Text reply and structured request workflows remain out of scope.
- Saying **Goodbye** ends the active conversation without confirmation. An explicit tapped End action may still use the appropriate confirmation behavior.

The current Mirror checkout contains corresponding family-message voice-reply implementation work in `mirror-app/app/family-message.tsx`, `mirror-app/src/api/familyMessages.ts`, and related audio code. This work must be preserved during consolidation and tested against the service contract.

## Caregiver implementation refinements to preserve

The current caregiver implementation and migration evidence show later work that must remain layered on top of the final caregiver architecture:

- Shared typography/design-token work, including consistent heading/body treatment.
- Shared card, box, button, and control spacing that remains usable with Android/system font scaling.
- Text wrapping that avoids awkward split words and border collisions.
- Monday-first calendar behavior with working previous/next-month navigation in `src/lib/monthCalendar.ts` and its tests.
- Redesigned Settings and Setup surfaces from the current A caregiver implementation rather than older legacy routes.
- Interactive/motion behavior in the shared component layer.
- Working buttons, dropdowns, pairing actions, consent states, and device setup states.
- The newer caregiver architecture/state layer in `src/architecture/`, demo mode, loved-one insights/history/trends, research flows, PWA/web export, and privacy/consent surfaces.

These items are implementation evidence and still require the architecture, function, and PNG visual gates before being called complete.

## Mirror implementation refinements to preserve

Current code and migration evidence require preservation of:

- Beige/light visual direction and readable/darker pairing code treatment.
- Back navigation where requested.
- Separate Wi-Fi, internet, Reflexion service, authentication/bootstrap, pairing, assignment, microphone, speaker, and device-identity diagnostic categories.
- Microphone pipeline, speaker/TTS, replay, playback completion, and turn-taking fixes.
- Automatic Aria speaking-to-listening transition after speech completes.
- Actual microphone capture and natural response handling.
- Answering a user's question before returning to the deterministic check-in controller.
- No immediate jump to the next scripted question without responding.
- No repeated covered topics, limited natural follow-up, deterministic completion, and partial completion on early ending.
- Current consent, pairing/device identity, backend/API, family-message, and web-demo behavior.
- The B checkout's `web:demo`/`export:web-demo` work and demo fixtures.
- The A checkout's network setup, Electron proxy/portal, on-screen keyboard, phone setup instructions, and device-consent integration.

## Known source conflicts to resolve explicitly

- Caregiver `AUTH-07` has a PNG and an A route, but the approved caregiver architecture says email verification is not required in the current release and does not include AUTH-07 in its 70-route inventory. Do not make it canonical without an explicit product decision.
- Caregiver `CHAT-05` is named “Video message recorder” in the PNG filename, while the approved architecture defines a voice-message recorder and the later instructions preserve voice-message behavior. Resolve the label/asset conflict before a UI PASS.
- Mirror Markdown says no family-message reply in the current release, while the later updated DOCX adds voice reply. The later DOCX wins.
- Mirror routine, research, status, Wi-Fi, and family-message PNGs represent canonical states; do not add unnecessary navigation routes where the architecture defines a state or overlay.

## Non-goals

Do not restore removed/deferred legacy features, including caregiver-side Mirror Wi-Fi setup, Mirror activity hubs, mood-score screens, text replies, structured loved-one requests, continuous ambient recording, or standalone processing/clarification/check-in-progress routes.
