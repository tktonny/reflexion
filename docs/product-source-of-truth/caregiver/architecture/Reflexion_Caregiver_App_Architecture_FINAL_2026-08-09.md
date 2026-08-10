# Reflexion Caregiver App Architecture

REFLEXION \| CAREGIVER APP ARCHITECTURE 9 August 2026

**Purpose:** This document defines the exact current product scope, caregiver-app UX, mirror behaviour, cross-device state models and full-screen inventory. A new designer, developer or coding model should be able to build the current Reflexion consumer product without referring to an older architecture.


## 0. Document control and source-of-truth rules
| **Item**            | **Decision**                                                                                                                                      |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Status              | Current build scope and normative product architecture.                                                                                           |
| Supersedes          | All prior caregiver-app screen lists, mockups and architecture drafts.                                                                            |
| Visual references   | May guide colours, typography, spacing and component style only. They do not override this document’s features, options, labels, flows or states. |
| Implementation rule | When current code, a mockup and this document conflict, this document wins.                                                                       |
| Change discipline   | Any new feature must first be added to the relevant product model, cross-device contract and canonical screen inventory before implementation.    |


## Contents

1. Product definition and goals
2. Actors, devices and user permissions
3. Product and language principles
4. Navigation and information architecture
5. Cross-device app–mirror contracts
6. Canonical state models
7. First-time setup
8. Home and loved-one dashboard
9. Activity
10. Chat and family messaging
11. Settings
12. Research Participation
13. Reusable state variants and native overlays
14. Canonical screen inventory
15. Explicitly removed or deferred scope
16. Build rules for designers and coding models
## 1. Product definition and goals
**Reflexion** is a consumer caregiving system that connects a caregiver mobile app to an older adult’s Reflexion device, initially the Reflexion Mirror and later the Bear or older-adult app. It supports conversations, routines, family messages, device management and factual caregiver updates. It is not a diagnostic, emergency-response or continuous-surveillance product.
•  Know whether a recent interaction occurred.
•  Know whether a specific update may need caregiver attention.
•  Call or leave a family message with minimal friction.
•  Review factual conversation summaries, routine responses and changes over time.
•  Manage devices, routines, permissions, research choices.
•  Preserve the older adult’s control over conversations, sharing and participation. The caregiver may record or update the loved one's ordinary product-consent choice from the caregiver app; a direct loved-one action to stop the current conversation on their Reflexion device takes immediate effect and cannot be bypassed for that active conversation.
### 1.1 Current product boundaries
| **Reflexion does**                                                              | **Reflexion does not**                                                          |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Record session facts such as date, time and duration.                           | Claim that a loved one is healthy, safe, stable, happy or “doing well.”         |
| Summarise what the older adult said, with provenance labels.                    | Independently verify a statement such as medication completion.                 |
| Compare eligible sessions with a recent baseline after minimum evidence exists. | Diagnose decline, disease, mood or clinical risk.                               |
| Deliver reminders and record the older adult’s response.                        | Guarantee that a reminder was followed.                                         |
| Show technical device status separately.                                        | Use device outage as evidence about the loved one.                              |
| Support family messages that the older adult opens or plays.                    | Provide loved-one reply or structured request workflows in the current release. |
## 2. Actors, devices and permissions
| **Actor / surface**     | **Role and capabilities**                                                                                                                                         |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Caregiver app users     | Pairs devices, configures routines and notifications, reviews permitted summaries, sends family messages, and reviews/updates ordinary product Consent & Control. |
| Older adult / loved one | Uses the Mirror, Bear or older-adult app. Can start or stop conversations, hear reminders, and open family messages.                                              |
| Reflexion Mirror        | Primary older-adult device. Conducts conversations, presents reminders, displays/plays family messages.                                                           |
| Reflexion service       | Synchronises configuration and events, creates permitted summaries, applies rule-based states and queues scheduled messages.                                      |
| Research team           | Only receives protocol-approved research data after valid, study- specific consent.                                                                               |
**Consumer terminology:** Use “loved one” in caregiver-facing copy and the person’s preferred name wherever possible. “Older adult” is acceptable in consent and policy explanations. Do not use “elderly”, “endpoint” or “patient” in ordinary consumer UI.
## 3. Product and language principles
- **Objective, not diagnostic.** Report what was observed by the system, what the loved one shared, and what could not be confirmed.
- **Reassurance without overclaiming.** Reassurance comes from confirmed interaction and a clear next step, not from a wellness claim.
- **One information model.** The same labels, options and states must be sourced from shared models across onboarding, Home, Activity and Settings.
- **Older-adult control.** The caregiver app may review and update the loved one’s ordinary Reflexion product consent and control state. Changes made from the caregiver app are persisted, timestamped and synchronised across the caregiver app and Reflexion device. The loved one may also stop a current conversation directly from their Reflexion device. Product consent is separate from Research Participation.
- **Technical separation.** Device online/offline state is displayed separately and never changes a loved-one status by itself. **Progressive disclosure.** Show the minimum information needed for the current decision; allow details through drill-down screens.
- **No hidden feature inheritance.** An existing feature is not valid merely because it already exists in code or an old mockup.
### Typography
- Normal UI text uses the Reflexion sans-serif UI typeface: body text, buttons, tabs, status text, inputs, settings rows, labels and standard headings.
- The serif/display typeface is reserved for intentional editorial/brand moments such as major hero headings and selected loved-one names.
- Typography is implemented through shared design tokens rather than page-specific font declarations.
### Responsive components
- Shared buttons, cards, selectable rows and controls use consistent internal horizontal/vertical padding and minimum touch dimensions. Components must tolerate system font scaling and text wrapping without text touching borders or actions becoming unusable.
### Consumer UI must never display
fetch failed  
java.net.UnknownHostException  
stack traces  
API hostnames  
raw HTTP/network exceptions.
For caregiver-app errors, identify at least:
Phone internet  
Reflexion service  
Authentication/session  
Required permission  
Loved-one device  
Unknown
And for **Mirror setup/pairing**, define separately:
Wi-Fi  
Internet  
Reflexion service  
Authentication/bootstrap  
Pairing code  
Assignment
### 3.1 Canonical provenance labels
| **Label**           | **Meaning**                                                      | **Example**                                                 |
|---------------------|------------------------------------------------------------------|-------------------------------------------------------------|
| Observed            | Direct session or device facts.                                  | Conversation recorded at 9:18 AM and lasted 18 minutes.     |
| What Mum shared     | Information stated by the loved one; not independently verified. | Mum said she had breakfast and planned a walk.              |
| Compared with usual | System comparison using the current eligible baseline.           | The session was shorter than Mum’s recent pattern.          |
| Recommended action  | A non-clinical caregiver action.                                 | Check in with mum when you can.                             |
| Limitations         | What Reflexion could not confirm or what data was unavailable.   | Reflexion cannot confirm whether the planned walk occurred. |
## 4. Navigation and information architecture
**Bottom navigation after setup:** Home · Activity · Chat · Settings
| **Tab**  | **Purpose**                                         | **Key destinations**                                                                                                    |
|----------|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Home     | Immediate household overview and person drill-down. | Loved-one card, Today, Sessions, Weekly Summary, Trends, History, Export, Call, Leave a message.                        |
| Activity | Cross-household chronological events.               | Timeline, filters, event detail.                                                                                        |
| Chat     | Family messages sent to each loved one.             | Thread, text/photo/voice composer, schedule, preview, delivery status.                                                  |
| Settings | Configuration and account management.               | You, Household, Devices, Notifications, Older-Adult Consent & Control, Privacy, Research, FAQ & Help, Feedback, Subscription, About. |
**History naming:** “Activity” is always the top-level tab and page name. “History” is only the person-specific calendar or chronological view inside a loved-one dashboard.
## 5. Cross-device caregiver app–mirror contracts
| Feature | Caregiver app behaviour | Mirror / older-adult experience | Result returned to app |
|---|---|---|---|
| Pairing | Select loved one + device type → Scan QR OR enter 6-digit code → Finish pairing | Create/store device identity → Check Wi-Fi → Internet → Reflexion service → Authentication → Generate pairing token/code → Await assignment → Paired | Validate pairing token → atomically assign device to loved one → return device credential/config bundle → idempotent retries |
| Conversation | Caregiver does not initiate or listen live. The app receives permitted results after a session. | Loved one starts or responds to a Reflexion conversation and can stop at any time. | Observed time/duration, summaries, topics, user-reported information, limitations and processing state. |
| Routine | Caregiver creates schedule, spoken text and notification rule. | Mirror presents the reminder and records the loved one’s response. | Scheduled, Presented, Reported complete, Deferred, Declined, No response or Device unavailable. |
| Text message | Caregiver writes or schedules a text message. | Mirror first shows a family-message notification. On selection, text is displayed and read aloud; Replay reads the message aloud again. Dismiss closes/dismisses the message. | Delivery state plus recipient interaction: Viewed or Replayed. |
| Photo message | Caregiver selects photo and adds a caption. | Mirror first shows a family-message notification. When opened, the photo and caption are displayed. The caption may be read aloud. Replay reads the caption aloud again. Dismiss closes/dismisses the message. | Delivery state plus recipient interaction: Viewed or Replayed. |
| Voice message | Caregiver records up to two minutes and sends now or schedules. | Mirror first shows a family-message notification. The message does not autoplay. The loved one selects Play to hear it. Replay plays the recording again. Dismiss closes/dismisses the message. | Delivery state plus recipient interaction: Played or Replayed. |
| Research participation | Caregiver may receive an optional study invitation and view study status. Consent controls are shown only to the invited participant(s) — caregiver, older adult or both — as permitted by the approved study protocol. | — | Study-specific status, version, date, participant and permitted research data scope. |
| Away mode | Caregiver sets dates and optionally pauses routines. | Mirror follows selected routine behaviour; ordinary comparison of baseline interactions is suspended. | Away status, return date and suppressed-comparison state. |
| Device health | Caregiver views technical status and troubleshoots. | Mirror performs connection, microphone and speaker tests. Mirror displays technical status. | Online/offline, last seen, Wi-Fi, version, update, microphone and speaker status. |
## 6. Canonical state models
### 6.1 Setup states
**Seven functional setup categories:** Household, Pair Device, Language & Accessibility, Routines, Notifications, Older-Adult Consent & Control, and Research Participation. Review and Complete are completion stages and are not counted in “x of 7”.
**Setup state is account state, not device-local onboarding state.**
Setup Overview derives each category's status from persisted Reflexion account data. On sign-in, restart, app reinstall or use on another caregiver phone, the app reconstructs setup progress from existing household profiles, paired devices and saved preferences. Local onboarding flags must not be the authoritative source of setup completion.
| **State**      | **Meaning**                                                                                                      |
|----------------|------------------------------------------------------------------------------------------------------------------|
| Not started    | No saved data for the category.                                                                                  |
| In progress    | Some data saved; category not complete.                                                                          |
| Complete       | Required information and decisions saved.                                                                        |
| Skipped        | Caregiver selected Set up later. The category remains resumable.                                                 |
| Not applicable | This category does not currently apply; it counts as satisfied for setup completion and requires no user action. |
### 6.2 Home card states
| **State**                             | **Colour** | **Required evidence / behaviour**                                                                                                      |
|---------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Interaction recorded today            | Green      | At least one valid interaction was recorded today. Do not convert this into a health claim.                                            |
| No interaction yet today              | Amber      | No valid interaction recorded by the configured point in the day; show factual timing and a proportionate next step.                   |
| Recent interaction shorter than usual | Amber      | Only after at least three valid sessions exist in the current 14-day window.                                                           |
| Device may be offline                 | Grey       | Technical uncertainty. Show last seen and device action; do not infer anything about the loved one.                                    |
| Needs your attention                  | Red        | A specific rule or explicit statement creates a concrete caregiver action. Never use red for a technical outage or inferred diagnosis. |
**Baseline rule:** Use the most recent 14 calendar days of valid sessions. At least three valid sessions are required. Before that, show “Building Mum’s usual pattern”. Weekdays and weekends are not separated. Travel, holidays and Away mode suspend ordinary comparison and missed-interaction logic.
### 6.3 Routine response states
| **State**          | **Consumer meaning**                                                         |
|--------------------|------------------------------------------------------------------------------|
| Scheduled          | Configured but not yet presented.                                            |
| Presented          | Mirror displayed or spoke the reminder.                                      |
| Reported complete  | Loved one said they completed it; not independently verified.                |
| Deferred           | Loved one chose to be reminded later.                                        |
| Declined           | Loved one said they would not complete it.                                   |
| No response        | No clear response recorded.                                                  |
| Device unavailable | Reminder could not be reliably delivered because the device was unavailable. |
### 6.4 Message states
- Delivery state: Draft → Scheduled → Queued → Delivered to device → Expired / Failed
- Do not collapse these states into “Sent”. If the device is offline, keep the message queued until the configured expiry rule.
- Recipient interaction is recorded separately from delivery state so later actions do not erase earlier ones.
Text and photo messages:
- Viewed
- Replayed
Voice messages:
- Played
- Replayed
Rules:
- Viewed means the message content was opened/displayed on the Reflexion device.
- Played means voice playback was initiated.
- Replayed means the loved one used Replay at least once after the initial view/play.
- If the loved one views/plays a message and subsequently dismisses it, leave as “Viewed” or “Played”.
- Store interaction events/timestamps so Viewed/Played and Replayed can all be retained where applicable.
- If the device is offline, keep the message queued until the configured expiry rule. Caregiver-facing status should therefore show: Text/photo: Delivered, Viewed, Replayed, Expired or Failed. Voice: Delivered, Played, Replayed, Expired or Failed.
### 6.5 Product consent, product control and research states
| **Model**              | **States**                                                                    | **Rule**                                                                                                                                                                                                                                                                     |
|------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Product consent        | Pending, Accepted, Declined, Withdrawn                                        | Controls whether ordinary Reflexion conversations and permitted caregiver updates may operate. The caregiver app may review and update this state. Changes are persisted, timestamped and synchronised with the loved one’s Reflexion device.                                |
| Product control        | Active, Paused                                                                | Controls whether ordinary Reflexion conversations are currently active. Pausing is temporary and does not equal withdrawal of product consent. The loved one may stop an active conversation immediately from their device.                                                  |
| Research participation | Not invited, Invitation pending, Consented, Declined, Withdrawn, Study closed | Completely separate from product consent and product control. Declining or withdrawing from research does not restrict ordinary Reflexion use. Research consent may only be changed by the permitted participant or representative according to the approved study protocol. |
**Request help is not a consent state.** If the loved one requests help understanding product consent, Product consent remains **Pending** until an Accepted or Declined choice is recorded. A separate “needs consent help” indicator may be recorded and shown to the caregiver.
## 7. First-time setup
Setup begins immediately after successful account creation or sign-in. Email verification is not required in the current release. Every setup screen includes Back, the current category, a primary action and Set up later. Categories are never locked and completed information is preserved.
### 7.1 Authentication
**Current release authentication rule:** Email verification is not required after account creation. A caregiver may create an account, sign in and proceed directly to setup without verifying their email address. The backend must not block sign-in or setup because emailVerified is false or absent. Email verification may be introduced in a later release.
- Splash: official logo, Care. Connected. and subtle loading indicator.
- Sign in: Email/Phone selector; separate country code and phone field; password/show; Forgot password;
- Google/Apple; Create account; Terms acknowledgement.
- Forgot/reset: email or phone, six-digit verification code, resend, new password and repeat password. The new password must meet the shared password requirements below.
- Create account: preferred name, email, country code, phone number, create password, repeat password, Terms and Privacy Policy.
- Password requirements for all new passwords created, all reset and change flows:
  - At least 12 characters;
  - At least one uppercase letter;
  - At least one lowercase letter;
  - At least one number;
  - At least one special character.
- Show password requirement validation while entering the password and clearly indicate whether the two password fields match.
- Account creation, password reset and later password changes must use the **same shared password-validation rules**.
- Do not show **Set up later** during account creation or password recovery.
### 7.2 Welcome and Setup overview
- Welcome explains the system as profile + device + preferences + permissions, then Start setup or Set up later.
- Setup overview lists all seven functional categories with Not started, In progress, Complete, Skipped or Not applicable.
### 7.3 Household
- Loved-One Profile: blank photo placeholder, preferred name, date of birth or age, gender, relationship, country code and phone, emergency contact, living arrangement.
- Add another loved one using the same blank form.
- Review household shows photo/placeholder, preferred name, relationship, Edit, Remove and Add another loved one.
No stock photo; no Primary/Lives nearby/Lives with caregiver labels.
### 7.4 Pair Device
- Select loved one, and device type: Mirror, Bear, App or Other supported device.
- Choose Scan QR code or Use pairing code.
- Pairing code is six digits.
- The backend validates the pairing token and assigns the device to the selected loved one.
- Wi-Fi, internet, Reflexion-service and device-authentication readiness are handled and reported by the Mirror/device, not configured through the caregiver pairing flow.
- Success: “Device paired successfully — Ready for Mum to use.”
### 7.5 Language & Accessibility
- Country, time zone, preferred spoken language, secondary language, text size, captions, speaking pace, assistant voice, volume, hearing support, high contrast and simplified interface.
- Voice preview demonstrates the selected language, voice, pace and volume (of the speaker on the mirror app) before saving.
### 7.6 Routines
- Categories: Medication, Meals, Hydration, Medical appointments, Exercise, Family events and Custom/Other.
- Example screens may prefill examples but nothing is activated until the caregiver saves a routine.
- Add/Edit: routine name, one or more times, repeat days, start date, optional end date, spoken reminder text and notification rule.
<!-- -->
- Notification rule per routine: Do not notify me; Notify me after one missed or unclear response; Include it in my daily summary.
- Review routines shows active schedules and opens edit; use “Mum reported that she took her medication”, never “Medication taken”.
### 7.7 Notifications
- Enable notifications, then invoke the operating system permission request.
- Choose one daily update time, with the currently supported preset options supplied by the shared Notification Preferences model; optionally notify every time a conversation session completes.
### 7.8 Older-Adult Consent & Control
- Explain simply what Reflexion does and how caregiver updates are generated.
- Product consent may be Pending, Accepted, Declined or Withdrawn.
- The caregiver may review and update ordinary product consent from the caregiver app.
- The loved one may Accept, Decline or Request help with the explanation.
- Request help leaves Product consent as Pending until a final choice is recorded.
- Product control may be Active or Paused.
- Pausing Reflexion is temporary and does not withdraw product consent.
- A loved one may stop the current conversation immediately from their Reflexion device.
- Product consent/control and Research Participation remain separate.
### 7.9 Review and Complete
- Review shows all seven categories with their current **Not started, In progress, Complete, Skipped or Not applicable** state and an Edit action where relevant.
- Finish setup or Complete later.
- Setup Complete summarises household, device readiness, routines, notifications, product consent, and research status.
## 8. Home and loved-one dashboard
### 8.1 Home structure
- Header: Reflexion brand, greeting to caregiver and date.
- Show one card per loved one.
- Each loved-one card contains:
  - photo or placeholder;
  - preferred name and chevron;
  - current interaction state;
  - last interaction timing;
  - conversation duration, where available;
  - **Leave a message**;
  - **Call**.
- The entire loved-one card opens that loved one’s dashboard.
- **Call** opens the phone dialler.
- **Leave a message** opens the Chat composer pre-addressed to that loved one.
- Device status must remain visually and semantically separate from the loved one’s interaction state.
- When relevant, show a compact device/configuration card above the loved-one card, for example **Device not paired** with **Pair device**.
- Below the loved-one card, Home may show a contextual **Suggested next step** and relevant setup, consent or configuration card.
- Do not show configuration cards when no caregiver action is required.
- **Explore** provides shortcuts to:
  - Sessions;
  - Weekly Summary;
  - Trends;
  - History;
  - Export Summaries.
- Explore shortcuts open the existing loved-one dashboard routes and do not create additional canonical routes.
### 8.2 Configuration banners
| **Situation**                   | **Title**                         | **Supporting text**                                                 | **Action**           |
|---------------------------------|-----------------------------------|---------------------------------------------------------------------|----------------------|
| Setup started                   | Complete your Reflexion setup     | {completed} of 7 sections complete                                  | Continue setup       |
| Loved one added, no device      | Connect Mum’s Reflexion device    | Pair a Mirror, Bear or App so Reflexion can begin.                  | Pair device          |
| Device paired, setup incomplete | Finish setting up Mum’s Reflexion | Choose routines, notifications and consent settings.                | Continue setup       |
| Notifications disabled          | Turn on notifications             | Allow Reflexion to tell you when an update may need your attention. | Enable notifications |
| Consent pending                 | Mum’s consent is still pending    | Some information remains unavailable until consent is completed.    | Review status        |
| Device configuration incomplete | Mum’s device needs attention      | One part of the device connection is not ready yet.                 | View device status   |
### 8.3 Loved-one dashboard
- Today: interaction state, last interaction, duration, short factual summary, topics, routine activity, Recommended action and limitations.
- Sessions: chronological session list by date/time and duration.
- Session Detail: summary, topics, user-reported information, routines discussed, provenance sections, caregiver note, Full Conversation transcript and inline “Was this update useful?” feedback.
- Weekly Summary: sessions, conversation time, days with interaction, routines, messages, notable events, Recommended action and limitations.
- Trends: 7 days, 30 days and 3 months.
- History: calendar or chronological view for the selected loved one.
- Export summaries in pdf: For a selected date range: Conversations and interaction time, Routines, Recommended actions, Messages
- Show preview of export summaries before Share Summaries.
## 9. Activity
- Activity is a cross-household timeline, not a loved-one profile tab.
- Events: conversation recorded; reminder presented; reminder response recorded; family message; device offline/reconnected; routine updated; permission changed; caregiver action.
- Filters: Person, Date range, Conversations, Routines, Chat, Technical and Caregiver actions.
- Event Detail links to the underlying session, routine, message, permission change or device event.
## 10. Chat and family messaging
- Chat list shows one thread per loved one and recent delivery state.
- Chat thread is a caregiver-side chronological record of family messages and delivery states.
- Message to Mum entry offers quick emojis, editable suggested text and Text, Photo or Voice type.
- Text composer: up to the product limit, send now or schedule.
- Photo composer: photo plus required or strongly prompted caption; caption can be read aloud on the Mirror.
- Voice recording: maximum two minutes; no autoplay on the Mirror.
- Schedule: Now or Specific date and time; show both caregiver local time and device time zone when different.
- Preview: final content, recipient, type and delivery time.
- Message status: show both delivery and recipient-interaction status. Delivery: Draft, Scheduled, Queued, Delivered to device, Expired or Failed. Interaction: Viewed/Played or Replayed
- No loved-one reply or request flows in the current product.
## 11. Settings
| Group | Routes and exact functions |
|---|---|
| You | Caregiver name, email, country code/phone, password and sign-in methods, and sign out. **Change Password uses the same shared password requirements defined in Section 7.1.** |
| Household | Loved Ones; Edit Loved-One Profile; Mark as Away; Language & Accessibility; Routine Management; Connected Devices. |
| Notifications | Exactly the same options and shared data model as onboarding. |
| Language and Accessibility | Exactly the same options and shared data model as onboarding. |
| Privacy & Data | View Product Consent, Product Consent History, data retention, delete selected data, delete account, vendor disclosure and cross-border disclosure. |
| Older-Adult Consent & Control | Current product-consent status; review/update consent; pause/withdrawal status; explanation/help. Reuses the relevant SETUP-18/19/20 consent screens rather than creating duplicate routes. |
| Research Participation | Study status, information sheet, consent copy, data scope, study contact and withdrawal where permitted by the protocol. |
| FAQ & Help | Searchable frequently asked questions organised by category, including account, setup, pairing, conversations, routines, messages, consent, privacy, devices and troubleshooting. Include Contact Support when the caregiver cannot find an answer. |
| Feedback | Allow caregivers to send feedback about Reflexion. Categories: Feature suggestion, Improvement, Confusing content, App issue, Device issue or Other. Feedback may include optional free-text details. |
| Subscription | Display the current configured Reflexion plan and price, billing date, payment method, manage and cancel subscription. |
| About Reflexion | Official logo, Care. Connected., description, story, safety/privacy, Terms, Privacy Policy, version and contact. |
## 12. Research Participation
### 12.1 Research Participation Overview
- Research Participation must not block ordinary product setup. When no active eligible study exists, the Research Participation setup category is treated as **Not applicable / satisfied for setup completion**.
- Identify the study title, sponsor/institution, principal investigator or responsible team, and who is being invited: caregiver, older adult or both.
- Explain why they are invited, what participation involves.
- Summarise exact data categories requested, including whether audio, transcript, routine, device or caregiver-app data is included.
- State that participation is voluntary, declining does not affect ordinary Reflexion use, and the user can decide later.
- Actions: View study information.
- Use the approved Reflexion and participating-institution logos/branding assets.
- For the current NUS/NUHS study, use the approved Reflexion, NUS Yong Loo Lin School of Medicine and NUHS assets.
### 12.2 Study Information and Consent
- Purpose of the research and study procedures.
- Expected duration, number/frequency of research interactions and any additional tasks.
- Foreseeable risks or discomforts; potential benefits and a clear statement if there is no guaranteed direct benefit.
- Compensation, reimbursement or costs, where applicable.
- Exact research data, recording, storage, retention, data transfer and confidentiality arrangements.
- Who may access identifiable or de-identified data.
- Voluntary participation, right to decline, withdrawal process and protocol-defined handling of data already collected.
- Research-team contact and independent participant-rights or ethics contact.
- Information-sheet version and date.
- Comprehension confirmations, identity and capacity/authorised-representative basis where the approved protocol permits proxy consent.
- Electronic signature, checkbox acknowledgement or recorded consent only as permitted by the approved protocol.
- Optional permission for raw audio or transcript when the study requires it.
### 12.3 Research Confirmation and Settings
- Confirmation shows participant, study, status, date/time, information-sheet version, consent method and permitted data scope.
- Provide a downloadable or viewable consent copy where permitted.
- Settings shows current study status, study information, contacts, consent copy and Withdraw/Leave study action where applicable.
- Ordinary product consent and research status are always displayed separately.
## 13. Reusable state variants and native overlays
These are not separate screens and must not be counted in setup progress or navigation.
### 13.1 Essential screen states
Relevant screens should support:
- **Loading** — Show that content is being retrieved.
- **Empty** — Explain that no information is available yet.
- **Error** — Explain what failed and provide **Try again**.
- **Phone offline** — State that the caregiver’s phone has no internet connection.
- **Device offline** — State that the loved one’s Reflexion device is offline.
- **Setup incomplete** — Explain what remains and link to the relevant setup step.
- **Summary processing** — State that the conversation summary is still being prepared.
- **Summary failed** — Keep the session date, time and duration visible and provide **Try again**.
Error messages should clearly identify whether the problem is with:
- Caregiver phone internet
- Reflexion service
- Authentication/session
- Required permission
- Loved-one device
- Unknown issue
Do not show a generic error when the cause is known.
### 13.2 Phone permissions
Request phone permissions only when the caregiver uses the relevant feature:
- Notifications.
- Camera for QR-code pairing.
- Photo library when choosing a photo.
- Microphone — request when the caregiver starts recording a voice message.
### 13.3 Confirmations
Show a confirmation before:
- Deleting an account or data.
- Withdrawing consent.
- Discarding unsaved changes.
### 13.4 Inline feedback
Relevant summaries may show:
**Was this update useful?**
- Yes
- Not really
Feedback should remain inline and must not open a full-screen popup.
## 14. Canonical full-screen inventory
This inventory contains only distinct full-screen routes. Reusable states, configuration banners, operating-system permission prompts, confirmation sheets, filters, pickers and inline feedback are not separate screens and are not counted here. The same screen may be opened from onboarding and Settings without being duplicated in this inventory.
### 14.1 Authentication
| **\#** | **Screen ID** | **Screen name**      |
|--------|---------------|----------------------|
| 1 | AUTH-01       | Splash               |
| 2 | AUTH-02       | Sign in              |
| 3 | AUTH-03       | Forgot password      |
| 4 | AUTH-04       | Reset verification   |
| 5 | AUTH-05       | Reset password       |
| 6 | AUTH-06       | Create account       |
These routes cover account creation, sign-in and password recovery as defined in Section 7.1. Email verification after account creation is not required in the current release.
### 14.2 First-time setup
| **\#** | **Screen ID** | **Screen name**                        |
|--------|---------------|----------------------------------------|
| 7 | SETUP-01      | Welcome to Reflexion                   |
| 8 | SETUP-02      | Setup overview                         |
| 9 | SETUP-03      | Loved-One Profile                      |
| 10 | SETUP-04      | Review household                       |
| 11 | SETUP-05      | Select loved one and device type       |
| 12 | SETUP-06      | Choose pairing method                  |
| 13 | SETUP-07      | Scan QR code                           |
| 14 | SETUP-08      | Enter six-digit pairing code           |
| 15 | SETUP-09      | Placement guide                        |
| 16 | SETUP-10      | Device paired successfully             |
| 17 | SETUP-11      | Language & Accessibility               |
| 18 | SETUP-12      | Voice preview                          |
| 19 | SETUP-13      | Routine category selection             |
| 20 | SETUP-14      | Routine examples and templates         |
| 21 | SETUP-15      | Add or edit routine                    |
| 22 | SETUP-16      | Review routines                        |
| 23 | SETUP-17      | Notification Preferences               |
| 24 | SETUP-18      | Older-Adult Consent & Control overview |
| 25 | SETUP-19      | Consent explanation                    |
| 26 | SETUP-20      | Consent status                         |
| 27 | SETUP-21      | Review setup                           |
| 28 | SETUP-22      | Setup complete                         |
Setup reuse rules
- SETUP-03 Loved-One Profile is reused when adding another loved one.
- SETUP-11 Language & Accessibility and SETUP-12 Voice preview are also opened from Settings.
- SETUP-14 Routine examples and templates changes its examples according to the selected category.
- SETUP-15 Add or edit routine is reused from Routine Management.
- **SETUP-17 Notification Preferences** is reused from Settings.
- SETUP-18/19/20 form the shared Older-Adult Consent & Control flow used from both Setup and Settings.
- Research Participation opens the research screens in Section 14.7.
### 14.3 Home and loved-one dashboard
| **\#** | **Screen ID** | **Screen name**                |
|--------|---------------|--------------------------------|
| 29 | HOME-01       | Home                           |
| 30 | LOVED-01      | Loved-one dashboard — Today    |
| 31 | LOVED-02      | Loved-one dashboard — Sessions |
| 32 | LOVED-03      | Session Detail                 |
| 33 | LOVED-04      | Full Conversation              |
| 34 | LOVED-05      | Weekly Summary                 |
| 35 | LOVED-06      | Trends                         |
| 36 | LOVED-07      | History                        |
| 37 | LOVED-08      | Export Summaries               |
Home and dashboard rules
- The Home configuration banners are variants of **HOME-01**, not separate screens.
- The loved-one card states are variants of the same Home card.
- Trends uses one screen with **7 days, 30 days and 3 months** controls.
- History uses one screen with calendar and chronological views.
- Selecting a date updates the History screen rather than opening a separate “Selected day” route.
- Export Summaries includes date-range selection, included sections, preview and **Share Summaries** on the same screen.
- Adding a caregiver note happens inside Session Detail.
- “Was this update useful?” remains inline.
The required Home and loved-one information is defined in Section 8.
### 14.4 Activity
| **\#** | **Screen ID** | **Screen name**       |
|--------|---------------|-----------------------|
| 38 | ACT-01        | Activity timeline     |
| 39 | ACT-02        | Activity event detail |
Activity filters open as a filter sheet or panel on **ACT-01** and are not a separate full-screen route. Activity remains the cross-household timeline, while History remains person-specific.
### 14.5 Chat and family messaging
| **\#** | **Screen ID** | **Screen name**        |
|--------|---------------|------------------------|
| 40 | CHAT-01       | Chat list              |
| 41 | CHAT-02       | Chat thread            |
| 42 | CHAT-03       | Text message composer  |
| 43 | CHAT-04       | Photo message composer |
| 44 | CHAT-05       | Voice message recorder |
| 45 | CHAT-06       | Message preview        |
Chat rules
- The recipient is selected before or when the composer opens.
- Quick emojis, suggested text and the Text, Photo or Voice selector appear in the Chat thread or a small type-selection sheet.
- **Send now** or **Schedule** is selected within the composer.
- Date and time selection uses a picker or sheet, not another route.
- Delivery and recipient-interaction status are shown in the Chat thread and message preview. They are not separate screens.
- Failed messages use the reusable error and retry treatment.
- There is no loved-one reply workflow.
These routes implement the message types, scheduling and preview requirements in Section 10 without creating unnecessary screens.
### 14.6 Settings
| **\#** | **Screen ID** | **Screen name**                    |
|--------|---------------|------------------------------------|
| 46 | SET-01        | Settings overview                  |
| 47 | SET-02        | Account                            |
| 48 | SET-03        | App Language                       |
| 49 | SET-04        | Loved Ones                         |
| 50 | SET-05        | Edit Loved-One Profile             |
| 51 | SET-06        | Away Mode                          |
| 52 | SET-07        | Routine Management                 |
| 53 | SET-08        | Connected Devices                  |
| 54 | SET-09        | Device Detail and Status           |
| 55 | SET-10        | Device Troubleshooting             |
| 56 | SET-11        | Privacy & Data                     |
| 57 | SET-12 | View Product Consent |
| 58 | SET-13 | Product Consent History |
| 59 | SET-14        | Data Retention                     |
| 60 | SET-15        | Vendor and Cross-Border Disclosure |
| 61 | SET-16        | FAQ & Help                         |
| 62 | SET-17        | Contact Support                    |
| 63 | SET-18        | Feedback                     |
| 64 | SET-19        | Subscription                       |
| 65 | SET-20        | Payment Method                     |
| 66 | SET-21        | About Reflexion                    |
Settings reuse rules
- **App Notifications** opens **SETUP-17 Notification Preferences**.
- **Language & Accessibility** opens **SETUP-11**.
- Voice preview opens **SETUP-12**.
- Add or edit routine opens **SETUP-15**.
- Older-Adult Consent & Control opens SETUP-18 and reuses SETUP-19/20 within that flow.
- Research Participation opens **RES-04 Research Participation Status**.
- Delete selected data and Delete account are actions inside **SET-11 Privacy & Data**, followed by confirmation sheets. They are not separate full-screen routes.
- Cancel subscription is handled inside **SET-19 Subscription**, followed by a confirmation sheet.
- Report App Issue and Report Device Issue are categories within **SET-18 Feedback**, not separate screens.
- FAQ answers are displayed within SET-16 FAQ & Help using expandable sections or in-page detail states; each FAQ answer does not receive a separate Screen ID.
This follows the Settings groups and functions defined in Section 11 while avoiding duplicate routes for shared configuration screens.
### 14.7 Research Participation
| **\#** | **Screen ID** | **Screen name**                 |
|--------|---------------|---------------------------------|
| 67 | RES-01        | Research Participation overview |
| 68 | RES-02        | Study information and consent   |
| 69 | RES-03        | Research consent confirmation   |
| 70 | RES-04        | Research Participation status   |
Research rules
- **RES-01** is opened from Setup when Research Participation is selected.
- **RES-04** is opened from Settings.
- The consent copy may be viewed or downloaded from **RES-04**; it does not need another Reflexion route.
- Withdrawal uses the action on **RES-04** followed by a confirmation sheet.
- Ordinary product consent and research participation remain separate.
These four screens cover the research overview, approved information and consent, confirmation and ongoing status requirements in Section 12.
### 14.8 Items that are not separate full-screen routes
The following must be implemented but must not receive additional Screen IDs:
- Home configuration banners.
- Home loved-one card states.
- Loading, empty, error and offline states.
- Summary processing and summary failed states.
- Native notification, camera, photo-library and microphone prompts.
- Delete, withdrawal, removal and discard confirmations.
- Activity filters.
- Message-type selector.
- Schedule date-and-time picker.
- Message delivery-state changes.
- Trend time-range controls.
- History calendar or chronological controls.
- Selected-day content inside History.
- Caregiver-note editing.
- Inline usefulness feedback.
This matches the rule that reusable states, native permissions and confirmations are not separate screens.
**Total: 70 canonical full-screen routes.**
## 15. Explicitly removed, merged or deferred scope
| **Item**                                           | **Decision**                                                                                                                     |
|----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Add loved-one photo                                | Merged into Loved-One Profile; not a route.                                                                                      |
| Pair Device introduction                           | Removed as a distinct route; setup overview leads directly to Select person and device type.                                     |
| Separate microphone/speaker pages                  | Removed from caregiver first-time pairing; technical checks are device-owned and surfaced through Device Status/Troubleshooting. |
| Name device                                        | Removed from current pairing flow. Loved-one assignment occurs through Select loved one and device type.                         |
| Routine Setup overview                             | Removed as a separate route; Routine category selection is the category landing page.                                            |
| Routine confirmation settings route                | Merged into Add/Edit Routine as the three notification options.                                                                  |
| Notification quiet hours                           | Removed from current main scope.                                                                                                 |
| System notification permission screen              | Native operating-system overlay, not a Reflexion route.                                                                          |
| What data is not collected                         | Not a standalone route; concise reassurance may appear inside consent explanation.                                               |
| Detailed audio/transcript preferences              | Deferred from product consent and ordinary settings.                                                                             |
| Loved Ones Overview / Loved-One Hub tab            | No top-level Loved Ones tab. Person-specific routes open from Home.                                                              |
| Weekly View 7/30/90                                | Replaced by Weekly Summary plus Trends 7/30/3 months.                                                                            |
| Generic Export Data                                | Not a primary Settings route; Export summaries remains in the loved-one dashboard.                                               |
| Resume Setup from Home / Setup Progress Overview   | Behaviours of Home — setup started and Setup overview, not separate routes.                                                      |
| Detailed threshold tuning / advanced status engine | Deferred pending pilot evidence.                                                                                                 |
| Advanced billing states                            | Deferred; current scope covers plan, billing date, payment method, manage and cancel.                                            |
## 16. Build rules for designers and coding models
- Build shared data models before screens: setup states, loved-one states, device states, routine response states, message states, consent, and research.
- Use one shared source for Notification Preferences in onboarding and Settings, and one shared source for routines in onboarding and Routine Management.
- Do not copy feature options from visual references. Visual references control aesthetics only.
- Do not create a route for every popup, state or permission. Use Section 13 to classify overlays and variants.
- Do not preserve an old route or option simply because it exists in code.
- Use exact consumer terms and approved state labels from this document.
- Keep device status separate from loved-one information throughout the data model and UI.
- No comparison claim before three eligible sessions exist in the current 14-day window.
- Every summary must preserve provenance and limitations.
- Every research feature must be gated by an active study and protocol-valid consent.
- Before implementation, produce a route map, shared model schema, component inventory, obsolete-route deletion list and test matrix for all canonical states.
- Do not require email verification for account creation, sign-in or setup in the current release. Password-reset verification remains required.
**Final source-of-truth statement:** This version replaces the previous caregiver-app architecture and all earlier screen inventories. Future mockups and code should reference the Screen IDs in Section 14 and the cross-device contracts in Section 5.
