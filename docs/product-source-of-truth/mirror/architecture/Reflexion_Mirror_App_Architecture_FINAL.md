# Reflexion Mirror App Architecture

**REFLEXION | MIRROR APP ARCHITECTURE | 9 August 2026**

**Purpose:** This document defines the current product scope, older-adult experience, mirror UI, conversation behaviour, caregiver-app contracts, pairing lifecycle, canonical state models and full-screen inventory for the Reflexion Mirror. A designer, developer or coding model should be able to build the current Mirror product without relying on the older mirror mockups or architecture diagrams.

## 0. Document control and source-of-truth rules

| Item | Decision |
|---|---|
| Status | Current build scope and normative Mirror product architecture. |
| Supersedes | All prior Mirror UI mockups, UX diagrams and Mirror architecture drafts where they conflict with this document. |
| Caregiver-app dependency | The final Caregiver App Architecture dated 9 August 2026 remains authoritative for caregiver-facing states, consent/control, message status, routine state, device status and pairing assignment. |
| Visual references | Previous mockups may guide visual direction only. They do not override the flows, features or states in this document. |
| Implementation rule | When current code, an old mockup and this document conflict, this document wins for the Mirror app. |
| Change discipline | Any new Mirror feature must first be reflected in the relevant state model, caregiver-app contract and canonical screen inventory. |

## Contents

1. Product definition and goals
2. Actors and product boundaries
3. Mirror UX and visual principles
4. System lifecycle and navigation model
5. Cross-device contracts
6. Canonical state models
7. Boot, connectivity and pairing
8. Idle Home
9. Aria conversation system
10. Daily check-in
11. Free-talk mode
12. Routines and reminders
13. Family messages
14. Consent, control and research
15. Offline, degraded mode, diagnostics and updates
16. Reusable states, overlays and accessibility
17. Canonical full-screen inventory
18. Explicitly removed, merged or deferred scope
19. Build rules and acceptance criteria

## 1. Product definition and goals

Reflexion Mirror is the primary older-adult-facing Reflexion device. It provides a low-friction conversational interface for daily check-ins, reminders, family messages and everyday companionship while synchronising permitted information with the caregiver app.

The Mirror should:

- Be understandable without prior technical knowledge.
- Allow the loved one to start a conversation simply by saying **“Hello Aria”** or using a clearly visible start control.
- Support a short structured daily check-in while allowing natural conversation around the required topics.
- Resume a partially completed check-in from the remaining topics instead of restarting from the beginning.
- Present caregiver-configured routines and reminders and record the loved one’s response as **user-reported**, never independently verified.
- Deliver caregiver text, photo and voice messages with clear playback/view status.
- Keep ordinary product consent/control separate from Research Participation.
- Expose technical problems clearly without showing raw developer errors.
- Preserve the loved one’s immediate ability to stop a conversation.

The Mirror is **not** a diagnostic, emergency-response, continuous-surveillance or medication-verification device.

### 1.1 Current product boundaries

| Reflexion Mirror does | Reflexion Mirror does not |
|---|---|
| Conduct conversational check-ins and free talk through Aria. | Diagnose dementia, cognitive decline, depression, delirium or another condition. |
| Record session facts and permitted conversation content for downstream processing. | Claim the loved one is safe, healthy, stable, happy or “doing well.” |
| Present reminders and record what the loved one reports. | Verify that medication, meals, hydration, exercise or another routine actually occurred. |
| Display and play permitted family messages. | Provide a loved-one reply or structured request workflow in the current release. |
| Show its own technical status and pairing readiness. | Treat device failure as evidence about the loved one. |
| Respect product consent/control and study-specific research consent. | Collect or upload unrelated ambient conversations while inactive. |

## 2. Actors and product boundaries

| Actor / surface | Role and capability |
|---|---|
| Loved one | Uses the Mirror, starts/stops conversations, hears reminders, reports routine responses, and opens family messages. |
| Reflexion Mirror | Hosts the older-adult UI, audio interaction, local device identity, pairing state, cached configuration and technical status. |
| Aria | Conversational assistant that follows the deterministic session controller, product boundaries and approved language rules. |
| Caregiver app | Assigns the Mirror to a loved one, configures routines/notifications/language/accessibility, sends family messages and reviews permitted results. |
| Reflexion service | Validates device credentials, stores authoritative assignment/configuration, orchestrates sessions, processes permitted data and synchronises state. |
| Research service/team | Receives only protocol-approved research data after valid study-specific consent. |

## 3. Mirror UX and visual principles

### 3.1 Older-adult-first interaction

- One clear purpose per screen.
- Large text, large tap targets and generous internal spacing.
- Normal UI text uses the Reflexion sans-serif UI typeface.
- Serif/display typography is reserved for large warm hero prompts, greetings and selected conversational moments.
- Avoid dense menus, tiny status text and nested navigation.
- Do not rely on colour alone to communicate status.
- Every important state should be understandable from several steps away.

### 3.2 Visual system

- Use a **warm ivory / soft beige** interface with muted Reflexion teal/green accents.
- Avoid a full-screen pure-black interface as the default visual language.
- Use high-contrast dark text and clearly separated cards/controls.
- The physical mirror/reflection should remain visually calm; UI surfaces should feel lightweight rather than like a phone app enlarged onto the mirror.
- Aria may use a small, consistent avatar/brand marker, but the avatar must not dominate the interface or be required to understand state.
- Family-message sender identity may use the caregiver’s photo/initial when available.

### 3.3 Interaction and language rules

- Use the loved one’s preferred name.
- Speak at the configured pace; current target is approximately **0.85× normal conversational speed** unless the caregiver changes it.
- Support the configured language; current product supports English and Chinese where the speech stack is available.
- Captions follow the Language & Accessibility configuration.
- Give one clear spoken instruction at a time.
- Reassure without infantilising or making health claims.
- Do not show caregiver-facing labels such as “No interaction today” on the Mirror.
- Do not expose stack traces, API hosts, tokens or raw network exceptions.

## 4. System lifecycle and navigation model

The Mirror is not a conventional multi-tab application. It moves between a small number of device states and content surfaces.

**Primary lifecycle:**

`Boot → Device readiness → Pairing if required → Configuration sync → Idle Home → Activation → Session route → Conversation / reminder / family message → Session close → Idle Home`

### 4.1 Logical states that are not separate screens

The following are controller states and should not create unnecessary full-screen routes:

- Session router
- Check-in progress
- Processing / transcription wait
- One-time clarification decision
- Post-session memory/summary processing
- Message event acknowledgement
- Configuration sync
- Heartbeat reporting

### 4.2 Current activation model

Current product activation is:

- Loved one says **“Hello Aria”**; or
- Loved one taps the prominent **Start / Talk to Aria** control.

Presence-based proactive invitation is **not required for the current release** and is listed as deferred in Section 18.

## 5. Cross-device contracts

| Feature | Caregiver / service behaviour | Mirror behaviour | State returned to caregiver app / service |
|---|---|---|---|
| Device registration | Service provides/validates device bootstrap capability and authoritative device identity. | Mirror creates/stores its local device identity and securely persists the resulting device credential. | Device registered/authenticated or explicit technical failure category. |
| Wi-Fi / internet | No caregiver-side Wi-Fi setup is required in the current pairing flow. | Mirror configures Wi-Fi locally, checks internet reachability and then checks Reflexion service reachability. | Wi-Fi, internet and service readiness are reported separately. |
| Pairing | Caregiver selects loved one + device type, then scans QR or enters six-digit code. Service atomically assigns device to loved one. | Mirror displays QR + six-digit code only when service/auth readiness is valid, then waits for assignment. | Paired device, assigned loved one and credential/config bundle. |
| Profile/configuration | Caregiver saves loved-one name, language, accessibility, routines, consent/control and related settings. | Mirror applies the latest valid configuration version and acknowledges successful application. | Applied configuration version and any validation failure. |
| Conversation | Service creates session and provides next-turn orchestration subject to consent and session controller. | Mirror captures the loved one’s speech only during an active interaction, plays Aria’s responses and shows conversation state. | Session facts, permitted transcript/audio objects, completion reason and processing state. |
| Routine | Caregiver creates routine schedule, spoken reminder and caregiver notification rule. | Mirror presents the due reminder and records the loved one’s response. | Presented, Reported complete, Deferred, Declined, No response or Device unavailable. |
| Family message | Caregiver sends/schedules text, photo or voice. | Mirror shows a family-message notification; loved one may open/play, replay and dismiss the surface. | Delivery plus recipient interaction: text/photo Viewed or Replayed; voice Played or Replayed. |
| Product consent/control | Caregiver may review/update ordinary product consent/control. | Mirror synchronises current state; loved one may stop an active conversation immediately. | Current consent/control state and change metadata. |
| Away mode | Caregiver sets Away dates and optional routine behaviour. | Mirror follows the configured routine behaviour and changes Home copy appropriately. | Away state and return date. |
| Device health | Service records heartbeats and technical events. | Mirror reports online state, Wi-Fi/service status, app version and diagnostics. | Online/offline, last seen, version/update, Wi-Fi, microphone and speaker status. |
| Research | Study workflow is enabled only when an active approved study applies. | Mirror shows protocol-gated study content only when the older adult is an invited participant and device consent is permitted by the protocol. | Study-specific consent/status/version/data scope. |

## 6. Canonical state models

### 6.1 Device readiness and pairing states

| State | Meaning / behaviour |
|---|---|
| Booting | App is loading local identity, credentials and cached configuration. |
| Wi-Fi required | No usable Wi-Fi connection exists. Open local Wi-Fi setup. |
| Internet unavailable | Wi-Fi exists but the internet cannot be reached. |
| Reflexion service unavailable | Internet works but the Reflexion service cannot be reached. |
| Device authentication unavailable | Service is reachable but device authentication/bootstrap is invalid or unavailable. |
| Ready to pair | Device is authenticated and can create a pairing QR/code. |
| Awaiting assignment | Pairing code is valid and the Mirror is waiting for caregiver assignment. |
| Paired / syncing | Service assignment succeeded; latest loved-one/configuration bundle is being applied. |
| Ready | Pairing and required configuration are valid; Mirror may enter Idle Home. |
| Assignment revoked | Device is no longer assigned; return to pairing-required state without showing old loved-one data. |

### 6.2 Daily check-in states

| State | Meaning |
|---|---|
| Ready | Today’s check-in has not yet been completed. Idle Home invites the loved one to start. |
| In progress | A check-in session is active. |
| Partially complete | Some core topics were covered but the check-in ended before completion. The next check-in activation resumes from remaining topics. |
| Complete | Deterministic controller confirms all required core topics for the current check-in were sufficiently covered or appropriately skipped. |
| Paused | Product control or Away mode means the daily check-in should not be actively invited. |

The LLM does **not** decide whether a check-in is complete. A deterministic controller tracks which core topics were naturally covered.

### 6.3 Conversation-turn states

`Aria speaking → Listening → Processing → next Aria response`

Possible branches:

- One simplified clarification after low transcription confidence.
- Reminder branch.
- Family-message branch.
- User requests stop / says goodbye.
- Technical failure.
- Safety-boundary response.

### 6.4 Routine response states

The Mirror records exactly the caregiver architecture states:

- Presented
- Reported complete
- Deferred
- Declined
- No response
- Device unavailable

**Reported complete** always means the loved one reported completion; it is not independently verified.

### 6.5 Family-message interaction states

**Delivery state:** Draft → Scheduled → Queued → Delivered to device → Expired / Failed

**Recipient interaction is stored separately:**

- Text/photo: Viewed, Replayed
- Voice: Played, Replayed

Mirror rules:

- Opening a text/photo message records **Viewed**.
- Starting a voice message records **Played**.
- Using Replay records **Replayed** while preserving the initial event timestamp.
- Dismiss closes the message surface but does not erase Viewed/Played/Replayed.
- In the current release, Dismiss is not a separate caregiver-facing final status after a message has been opened/played.

### 6.6 Product consent and control

| Model | States | Mirror rule |
|---|---|---|
| Product consent | Pending, Accepted, Declined, Withdrawn | Mirror may only run ordinary product conversations and permitted sharing when the current consent state allows it. |
| Product control | Active, Paused | Pausing is temporary and does not equal withdrawal. |
| Research participation | Not invited, Invitation pending, Consented, Declined, Withdrawn, Study closed | Completely separate from ordinary product consent/control. |

A loved one may always stop the **current active conversation** immediately from the Mirror even when Product control is Active.

### 6.7 Connectivity states

Technical states must be separate from loved-one interaction state:

- Wi-Fi connected / unavailable
- Internet available / unavailable
- Reflexion service reachable / unavailable
- Device authentication ready / failed
- Pairing ready / unavailable
- Assignment pending / assigned

Do not collapse these into a generic **“Something went wrong”** when the actual category is known.

## 7. Boot, connectivity and pairing

### 7.1 Boot

On launch, the Mirror should:

1. Load local device identity and secure credential.
2. Load cached configuration.
3. Check Wi-Fi.
4. Check internet.
5. Check Reflexion service.
6. Validate device authentication.
7. Determine assignment/pairing state.
8. Sync the latest valid configuration if paired.
9. Route to Pairing or Idle Home.

### 7.2 Wi-Fi setup

If Wi-Fi is not configured or fails:

- Show nearby networks in a simple list.
- Allow password entry with show/hide password.
- Show Connecting / Connected / Failed states.
- Provide **Try again** and **Choose another network**.
- After a successful connection, automatically continue through Internet and Reflexion-service checks.

### 7.3 Pairing readiness screen

Before showing a code, display large, readable readiness rows:

- Wi-Fi connected
- Internet available
- Reflexion service reached
- Device authentication ready
- Pairing code ready
- Awaiting caregiver assignment / Assigned

Each failed row should provide the most relevant action, for example:

- **Wi-Fi settings**
- **Retry internet check**
- **Retry Reflexion service**
- **Retry device authentication**
- **Refresh pairing code**

The ordinary consumer flow must **not** expose bootstrap tokens, JWTs, API hosts or raw technical secrets. Prototype/support-only credential entry, if retained in code, belongs behind an internal support mode and is not part of the canonical consumer UI.

### 7.4 Pairing code

When readiness is valid:

- Display a high-contrast QR code.
- Display the six-digit pairing code in very large text.
- Include concise instructions: **“On the caregiver app, choose Pair device and scan this code or enter the six digits.”**
- Show the code expiry/refresh state without making the code visually faint.
- Refresh an expired code automatically where possible.
- Keep Wi-Fi/service status visible in a compact form.
- Allow navigation back to device status/Wi-Fi setup.

### 7.5 Assignment and completion

After caregiver assignment:

- Confirm the selected loved one.
- Apply the latest loved-one profile/configuration bundle.
- Clear any pairing code from display.
- Show **“Device paired successfully — Ready for [Name] to use.”**
- Continue to Idle Home.

## 8. Idle Home

Idle Home is the resting screen and the main return point after every interaction.

### 8.1 Core content

Show:

- Time
- Day and date
- Weather when available
- Loved-one greeting
- Current check-in status
- One primary instruction/action
- Relevant reminder or family-message cards, if any
- Small device/help control

Examples:

**Check-in ready**
> Good morning, Nana.  
> Your daily check-in is ready.  
> **Say “Hello Aria” when you’re ready.**

**Check-in complete**
> Hi Nana.  
> Your check-in is complete for today.  
> **Say “Hello Aria” anytime to chat.**

**Partially complete**
> Welcome back, Nana.  
> We can continue where we left off.  
> **Say “Hello Aria” when you’re ready.**

**Paused / Away**
> Daily check-ins are paused until Friday.  
> **Say “Hello Aria” to chat anytime.**

**Service unavailable**
> Aria is temporarily offline.  
> Some features may be unavailable.

### 8.2 Home cards

Home may surface only relevant, actionable cards such as:

- **You have 2 reminders today**
- **Medication reminder at 9:00 AM**
- **A message from Mei**

Do not turn Home into a dashboard of caregiver-facing metrics.

### 8.3 Weather

- Weather is optional and must fail gracefully.
- Use the configured location/unit system.
- If weather cannot load, omit it or show **Weather unavailable**; do not present it as a device-critical error.

## 9. Aria conversation system

### 9.1 Activation acknowledgement

After **“Hello Aria”** or a Start tap:

- Give immediate visual/audio acknowledgement.
- Route based on today’s check-in state:
  - Ready → start daily check-in.
  - Partially complete → resume remaining check-in topics.
  - Complete → start free talk.
  - Paused → start free talk only if product consent/control allows it.

The router is a logical state, not a full-screen menu.

### 9.2 Aria speaking screen

Show:

- Time/date and compact technical indicators only if useful.
- Small Aria identity marker.
- **Aria is speaking…**
- Large text of what Aria is saying.
- Subtle waveform/activity indicator.
- Controls:
  - **Repeat**
  - **Stop**
  - **Continue** only when a user confirmation is logically needed

Avoid showing three large controls on every turn when only Stop/Repeat are relevant.

### 9.3 Listening screen

Show:

- The question/prompt Aria just asked.
- Clear microphone/listening state.
- **Listening…**
- Audio waveform/activity indicator.
- **Repeat question**
- **Stop**
- Reassuring copy such as **“Take your time. I’ll wait until you finish.”**

### 9.4 Processing state

Processing should be a lightweight transition state rather than a separate destination.

- Show a calm **“One moment…”** or equivalent.
- Do not display raw transcription/provider details.
- If processing exceeds the expected period, move to the correct service/error state.

### 9.5 Speech uncertainty and clarification

Current conversation rule:

- After approximately 7–10 seconds without a clear response, give one gentle re-prompt.
- If transcription confidence is low, Aria may ask **one simplified clarification** for that turn.
- Example: **“Sorry, I didn’t quite catch that. Could you say it once more?”**
- Do not loop repeated clarifications.
- If still unclear, record the turn as unclear/no response and continue appropriately or end gracefully.

### 9.6 Stopping and ending

The loved one may stop by:

- Saying **“Goodbye”**, **“Stop”** or an equivalent supported phrase; or
- Tapping **Stop**.

When Stop is tapped during an active session, show a simple confirmation sheet:

- **Continue conversation**
- **End conversation**

Do not require confirmation for an explicit repeated voice request to stop.

### 9.7 Session close

If daily check-in completed:

> Thank you, Nana.  
> Your check-in is complete for today.

If incomplete:

> That’s alright. We can continue another time.

The completion screen must not claim the loved one is healthy, safe or “doing well.”

Post-session processing runs in the background after returning to Idle Home.

## 10. Daily check-in

The check-in is a conversational flow, **not a menu of activities**.

### 10.1 Core topic controller

The deterministic controller tracks whether the following core areas were naturally covered:

1. Warm greeting / general check-in
2. Recall of recent/yesterday events
3. Planning for today / upcoming activities
4. Due routine/reminder branch where relevant
5. Reminiscence prompt approximately twice weekly where appropriate
6. Pending family-message branch where appropriate
7. Warm close

Topics may occur in a natural order.

Rules:

- Aria may answer the loved one’s question before returning to the check-in.
- Already-covered topics are not asked again.
- Topics may be skipped where inappropriate or unavailable.
- Aria should use prior permitted session context for continuity without exposing caregiver-only notes.
- The LLM does not determine completion.
- If the user ends early, the controller records partial completion and resumes from remaining topics later.

### 10.2 Natural follow-up

Aria may ask a limited natural follow-up when it helps the conversation, but should not allow the check-in to become an unbounded interview.

### 10.3 No mood-score screen

A general wellbeing question may be asked conversationally, but the current product does **not** require a mood-icon selection screen or generate a mood score from the response.

## 11. Free-talk mode

When the daily check-in is already complete, saying **“Hello Aria”** starts free talk.

Free talk should:

- Respond naturally within the product/safety boundaries.
- Use permitted memory/context where available.
- Avoid turning ordinary conversation into a clinical assessment.
- Allow the user to stop at any time.
- Not retroactively alter daily check-in completion unless the deterministic controller explicitly records a missing check-in topic as covered under an allowed resume flow.

## 12. Routines and reminders

### 12.1 When reminders appear

A reminder may appear:

- At its scheduled time while the Mirror is idle; or
- At an appropriate turn boundary during an active conversation if configured and not disruptive.

Do not interrupt Aria or the loved one mid-sentence.

### 12.2 Reminder screen

Show:

- Routine type/name
- Spoken reminder text
- Large icon
- Clear response choices

Recommended user-facing choices:

- **Done** / routine-specific equivalent
- **Not yet**
- **Remind me later**

Spoken refusal such as **“No, I’m not doing that”** may map to **Declined**.

Backend mapping:

- Reminder displayed/spoken → Presented
- User says they completed it → Reported complete
- Remind me later → Deferred
- Explicit refusal → Declined
- No clear response → No response
- Reminder could not be delivered → Device unavailable

For medication, never convert a button press into an independently verified “Medication taken” fact. It remains the loved one’s report.

### 12.3 Reminder copy

Avoid health claims such as **“Your health is our priority”** or congratulatory clinical-sounding reassurance. Use neutral, warm copy.

## 13. Family messages

### 13.1 Notification on Home

Pending family messages may appear as a Home card with:

- Sender name/photo or initial
- Message type
- Short preview for text/photo captions
- **Open**

### 13.2 Text message

When opened:

- Show sender identity.
- Show large text.
- Read the message aloud when configured.
- Controls:
  - **Replay**
  - **Dismiss**

Opening records **Viewed**. Replay records **Replayed**.

### 13.3 Photo message

When opened:

- Show the photo at a readable size.
- Show sender and caption.
- Read the caption aloud when configured.
- Controls:
  - **Replay caption**
  - **Dismiss**

Opening records **Viewed**. Replay records **Replayed**.

### 13.4 Voice message

When opened:

- Do **not** autoplay.
- Show sender identity.
- Provide a large **Play** control.
- After first playback, provide **Replay**.
- Provide **Dismiss**.

Starting playback records **Played**. Replay records **Replayed**.

### 13.5 No reply flow in current release

The current product does **not** include:

- Voice reply recording
- Text reply
- Structured loved-one requests
- “Reply” button on family-message screens

Any prior mockup showing a Reply button is obsolete for the current release.

## 14. Consent, control and research

### 14.1 Ordinary product consent/control

The Mirror must consume the same canonical states as the caregiver app:

- Product consent: Pending, Accepted, Declined, Withdrawn
- Product control: Active, Paused

The Mirror must:

- Prevent ordinary product conversation/sharing when the current consent state does not permit it.
- Reflect Paused state clearly on Idle Home.
- Allow the loved one to stop the current conversation immediately.
- Apply caregiver-recorded consent/control updates after successful configuration sync.

### 14.2 Product consent explanation

Where the Mirror is used to help the loved one review consent:

- Use simple language.
- Explain what ordinary Reflexion conversations do and what may be shared.
- Allow Accept, Decline or Request help when that workflow is enabled.
- **Request help** leaves consent Pending; it is not a separate consent state.

### 14.3 Research Participation

Research is separate from ordinary product use.

Mirror research surfaces should appear only when:

- There is an active approved study;
- The older adult is an invited participant or authorised representative route applies; and
- The approved protocol permits consent/interaction on the Mirror.

Declining research must not disable ordinary Reflexion use.

## 15. Offline, degraded mode, diagnostics and updates

### 15.1 Offline / degraded behaviour

When paired but temporarily offline:

- Continue showing local time/date.
- Hide or mark weather unavailable if it cannot refresh.
- Continue locally cached reminders where safe and supported.
- Do not start cloud-dependent Aria conversation turns when the service is unavailable.
- Queue eligible local technical/routine events for later reconciliation where supported.
- Show the exact known technical category.

Example:

> **Wi-Fi is disconnected**  
> Aria needs Wi-Fi for conversations.  
> **Open Wi-Fi settings**

or

> **Reflexion service is unavailable**  
> Your Wi-Fi is connected. Aria cannot reach Reflexion right now.  
> **Try again**

### 15.2 Failure during an active conversation

If connectivity fails mid-session:

- Stop retry loops.
- Preserve already-completed check-in topics where possible.
- Say a short neutral apology.
- Mark the session as technically interrupted.
- Return to Idle Home / degraded state.
- Allow later resume from remaining check-in topics.

### 15.3 Device Status & Help

Provide a simple technical screen reachable from setup and the Home help control:

- Wi-Fi
- Internet
- Reflexion service
- Device authentication
- Pairing / assignment
- Microphone
- Speaker
- App version
- Update state
- Last successful service contact

Actions should be contextual: Wi-Fi settings, Retry, Test speaker, Test microphone, Restart app/device, Contact support instructions.

### 15.4 Updates and restart

- Do not install/restart during an active conversation, reminder response or message playback.
- Prefer idle periods.
- Show a simple **Updating Reflexion…** / **Restarting…** state.
- If an update fails, retain the last working app version where technically possible and report the failure through device health.

## 16. Reusable states, overlays and accessibility

These are variants/overlays and should not inflate the canonical screen count.

### 16.1 Reusable states

- Loading
- Empty / nothing pending
- Processing
- Network retry
- Service retry
- Pairing-code expired/refreshing
- Configuration syncing
- Summary/session processing
- Check-in partially complete
- Check-in complete
- Paused / Away
- Update in progress

### 16.2 Confirmation overlays

Use a confirmation overlay for:

- Stop button during a conversation
- Forget/reconfigure Wi-Fi where destructive
- Restart when an active operation would be interrupted

### 16.3 Safety-boundary response

If the loved one expresses language suggesting immediate danger or urgent help needs, Aria must follow the constrained safety policy rather than improvise a clinical assessment.

The Mirror should:

- Use calm, direct language.
- Encourage contacting a trusted person or local emergency services when appropriate.
- Never claim that help has been contacted unless a confirmed communication feature actually did so.
- Never imply Reflexion is monitoring continuously or guaranteeing safety.

The safety response is a constrained conversation state, not a diagnosis screen.

### 16.4 Accessibility

Apply caregiver-configured settings consistently:

- Text size
- Captions
- Speaking pace
- Voice
- Volume
- Hearing-support mode
- High contrast
- Simplified interface
- Primary/secondary language

Critical controls such as **Stop**, **Replay**, **Play**, **Done** and **Remind me later** must remain usable at larger text sizes.

## 17. Canonical full-screen inventory

Only distinct full-screen surfaces receive Screen IDs. Logical states, transient processing, confirmations and technical overlays are not separate routes.

| # | Screen ID | Screen name |
|---|---|---|
| 1 | MIR-01 | Boot / loading |
| 2 | MIR-02 | Wi-Fi setup |
| 3 | MIR-03 | Device readiness & pairing status |
| 4 | MIR-04 | Pairing QR & six-digit code |
| 5 | MIR-05 | Device paired successfully |
| 6 | MIR-06 | Idle Home |
| 7 | MIR-07 | Aria speaking |
| 8 | MIR-08 | Aria listening |
| 9 | MIR-09 | Session close |
| 10 | MIR-10 | Routine reminder |
| 11 | MIR-11 | Family message viewer/player |
| 12 | MIR-12 | Product Consent & Control |
| 13 | MIR-13 | Device Status & Help |
| 14 | MIR-14 | Research Participation (protocol-gated) |

### 17.1 Screen reuse rules

- Check-in and free talk reuse **MIR-07 Aria speaking** and **MIR-08 Aria listening**.
- Processing is a transient state of the conversation surface, not a route.
- One-time clarification reuses the speaking/listening surfaces.
- Stop confirmation is a sheet/overlay, not a route.
- Check-in progress is logical controller state, not a route.
- Text, photo and voice family messages reuse **MIR-11** with content-type variants.
- Pairing-code expired/refreshed is a variant of **MIR-04**.
- Offline/degraded states reuse **MIR-03**, **MIR-06** or **MIR-13** depending on context.
- Update/restart is a system state/overlay and does not need another route.
- Safety-boundary behaviour reuses the active conversation surface.

**Total: 14 canonical full-screen Mirror routes.**

## 18. Explicitly removed, merged or deferred scope

| Item from older mockups/architecture | Current decision |
|---|---|
| **“Let’s do something meaningful together” activity selection screen** | Removed from the current main flow. Planning and reminiscence are embedded naturally inside the daily conversation. |
| Remember Together as a standalone module | Removed from current main flow; reminiscence remains an embedded conversational topic. |
| Plan My Day as a standalone module | Removed from current main flow; planning remains a core daily check-in topic. |
| Light Challenge | Deferred. Not part of the current Mirror product architecture. |
| Engage Home / Engage Activity hub | Deferred. Do not create these routes for the current build. |
| Mood-icon selection / mood score | Removed from the current flow. General wellbeing may be discussed conversationally without turning it into a mood assessment. |
| Family-message Reply button / voice reply recorder | Removed from current release to match caregiver architecture; Replay and Dismiss remain. |
| Loved-one request workflow | Deferred. No structured request route in the current release. |
| Large user avatar on every Mirror screen | Not required. Use the loved one’s name; Aria/sender identity may appear where useful. |
| Caregiver-facing “No interaction” / trend labels on Mirror | Removed. Mirror uses loved-one-facing check-in wording only. |
| Separate Processing full screen | Merged into transient conversation state. |
| Separate Clarification full screen | Merged into Aria speaking/listening variants. |
| Separate Check-in Progress screen | Logical state only. |
| Caregiver-side Wi-Fi setup | Removed. Wi-Fi is configured locally on the Mirror; caregiver pairing remains QR/code + assignment. |
| Manual bootstrap-token/JWT entry in ordinary Mirror UI | Removed from canonical consumer flow. Internal prototype/support tooling may exist behind support mode only. |
| Presence-based automatic check-in initiation | Deferred until hardware sensing, privacy behaviour and false-activation handling are validated. |
| Continuous ambient recording while idle | Not permitted in current scope. |

## 19. Build rules and acceptance criteria

### 19.1 Build rules for designers and coding models

- Build shared state models before screens: device readiness, pairing, check-in progress, conversation turn, routine response, family-message interaction, consent/control and research.
- Mirror and caregiver app must use the same canonical routine, consent, assignment and message states.
- Keep technical/device state separate from loved-one interaction content.
- Do not expose raw technical exceptions or credentials.
- Do not add a route merely because it appears in an old Figma/mockup.
- Do not add reply/request functionality unless the caregiver architecture is updated first.
- The deterministic controller, not the LLM, determines daily check-in completion.
- Preserve partial check-in progress across an interrupted session where technically possible.
- Use one simplified clarification maximum per unclear turn.
- Never infer verified medication adherence from a reminder response.
- Every safety output must follow the constrained safety policy.
- Every research feature must be gated by an active study and protocol-valid consent.

### 19.2 Core acceptance tests

**Pairing**

- Wi-Fi, internet, Reflexion service, authentication, pairing and assignment failures are distinguishable on-screen.
- QR and six-digit code are large and readable.
- Caregiver assignment immediately resolves to the correct loved one.
- Assignment survives app/device restart.
- Revoked assignment removes loved-one-specific data and returns to setup safely.

**Idle Home**

- Correct copy for Ready, Partially complete, Complete, Paused/Away and Offline states.
- No caregiver-facing health/status interpretation appears on Mirror Home.
- Weather failure does not break Home.

**Conversation**

- “Hello Aria” starts/resumes the correct mode.
- Speaking, Listening and Processing states are visually unambiguous.
- Only one simplified clarification is attempted per unclear turn.
- Stop works immediately and does not loop.
- Partial check-in resumes from remaining topics.
- Completed check-in does not restart unnecessarily that day.

**Routines**

- Due reminder appears at the correct time/turn boundary.
- Done maps to Reported complete, not verified completion.
- Deferred, Declined and No response are distinguishable.
- Device-unavailable events do not become loved-one behaviour claims.

**Family messages**

- Text/photo open → Viewed.
- Voice first playback → Played.
- Replay → Replayed.
- Dismiss closes the message without erasing prior interaction status.
- No Reply control appears in current release.

**Consent/control**

- Pending/Declined/Withdrawn states gate ordinary product behaviour correctly.
- Paused is temporary and distinct from Withdrawn.
- Loved one can stop the active conversation immediately.
- Caregiver changes synchronise to the Mirror.

**Offline/degraded**

- The Mirror names the known failure category.
- No raw Java/network/provider error appears.
- Conversation interruption ends gracefully and preserves valid progress where possible.

**Final source-of-truth statement:** This Mirror architecture replaces the previous Mirror UI mockups, UX architecture and UI architecture wherever they conflict. Future Mirror mockups and code should reference the Screen IDs in Section 17 and the cross-device contracts in Section 5.
