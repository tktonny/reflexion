# Product Requirements Document: Multi-Surface Platform

## 1. Product Summary

Reflexion should be treated as one clinical platform delivered through three coordinated product surfaces:

- provider management console
- caregiver and patient client app
- assessment terminal or smart-mirror style interaction surface

These surfaces share the same data contracts, identity model, capture pipeline, and longitudinal engine, but they do not show the same level of clinical detail to each user type.

The provider console remains the clinical control plane. The caregiver and patient client app is a supervised support surface for adherence, coaching, and simplified status communication. The assessment terminal is the guided capture surface for structured realtime interaction and full-session recording.

## 2. Surface Overview

### Provider Management Console

Primary users:

- geriatricians
- neurologists
- memory clinic physicians
- care coordinators
- research operations staff

Core jobs:

- enroll and manage patients
- confirm identity and consent
- configure and launch clinic or home protocols
- review formal batch assessment outputs
- review longitudinal trends and escalations
- review identity-link confidence and exclusion flags
- approve what can be communicated to caregivers or patients

### Caregiver And Patient Client App

Primary users:

- older adults in provider-managed programs
- family caregivers
- hired caregivers acting under provider supervision

Core jobs:

- receive reminders and nudges for scheduled sessions
- complete approved home check-ins or exercises
- view simplified progress, adherence, and wellness status
- receive activity recommendations and structured exercises
- view caregiver alerts when provider-approved thresholds are crossed
- submit context such as sleep, mood, appetite, falls, or behavior changes

Important constraint:

- this app must not present raw clinical-risk outputs as a standalone diagnosis
- clinician-facing risk scores stay in the provider console unless explicitly converted into a provider-approved patient-safe summary

### Assessment Terminal Or Mirror Surface

Primary users:

- patients during supervised or semi-supervised sessions
- caregivers helping with home capture
- clinic staff launching one-session assessments

Core jobs:

- run a stage-based realtime conversation
- display one prompt at a time with a calm simple UI
- record the full video and audio session
- perform local quality checks
- upload the full recording for formal multimodal analysis
- support repeatable capture from the same physical setup

## 3. Core User Stories

### Clinic Assessment Story

As a clinician, I need to launch a structured assessment on a dedicated capture surface, review the formal post-session multimodal result in the provider console, and record the final clinical disposition.

### Home Monitoring Story

As a provider, I need the caregiver and patient client app to collect repeated short interactions, while the provider console shows baseline completeness, trend changes, and escalation signals.

### Caregiver Support Story

As a caregiver, I need a simple app that tells me what to do next, what activities or assessments are recommended, whether the patient is adhering, and when provider follow-up has been recommended.

## 4. Surface Responsibilities

### Provider Management Console

Must show:

- patient roster and enrollment state
- identity and consent state
- clinic assessment history
- provider trace and QC context
- longitudinal trend charts
- escalation queue
- patient-safe summary editor or approval state

Must not depend on:

- local browser-only state
- hidden heuristic outputs not persisted in shared records

### Caregiver And Patient Client App

Must show:

- next recommended task or exercise
- schedule and adherence progress
- simple status badges such as on-track, needs repeat capture, or provider review pending
- caregiver view with alerts, adherence, and recommended actions
- personalized activity recommendations approved by the care program

Must avoid:

- raw investigational risk scores by default
- unsupported disease-subtype claims
- unreviewed identity-sensitive media review details

### Assessment Terminal Or Mirror Surface

Must show:

- one active stage at a time
- stage goal and completion cues
- simple capture-state feedback
- reassurance and pacing for older adults

Must avoid:

- raw provider routing details during interaction
- real-time diagnostic language
- cluttered multi-pane clinician UI

## 5. Shared Capability Mapping

### Realtime Conversation Layer

Used primarily by:

- assessment terminal

Secondary consumers:

- provider console for protocol configuration

### Formal Batch Multimodal Assessment Layer

Used primarily by:

- provider console

Secondary consumers:

- caregiver or patient client only through provider-approved summary objects

### Longitudinal Tracking Layer

Used by:

- provider console for trend review
- caregiver and patient client for adherence and simplified status

### Identity Attribution Layer

Used by:

- assessment terminal for within-session target checks
- provider console for manual review and override
- longitudinal engine for inclusion and exclusion decisions

## 6. Information Disclosure Rules

- Provider console can display full investigational assessment details, QC context, and provider routing trace.
- Caregiver and patient client app should display simplified, provider-approved messages.
- Assessment terminal should display guidance, progress, and capture status only.
- Identity uncertainty should be visible to providers and hidden from patients unless it changes the requested next action.
- Medium and high longitudinal changes route to provider review first before client-facing interpretation.

## 7. Example Client-App Feature Areas

The caregiver and patient client app should support experience areas such as:

- personalized activity recommendations
- cognitive assessments and exercises
- adherence and daily check-in history
- caregiver view with alerting and support cues
- provider-approved follow-up guidance

These are support and engagement features, not independent diagnostic claims.

## 8. Cross-Surface Workflow

1. Provider configures the protocol in the management console.
2. Patient completes a clinic or home interaction on the assessment terminal or home client surface.
3. Realtime capture records transcript, audio, video, and quality metadata.
4. Full recording is uploaded for formal batch multimodal analysis.
5. Provider console receives the formal assessment and any longitudinal updates.
6. Identity linkage decides whether the session is safe to include in the patient timeline.
7. If provider rules allow, a simplified result or next-step summary is sent to the caregiver and patient client app.

## 9. Non-Goals For Version 1

- direct-to-consumer unsupervised self-diagnosis
- autonomous emergency or referral decisions without provider review
- showing raw investigational model confidence directly to patients by default
- treating caregiver input as equivalent to clinician adjudication

## 10. Release Sequence

### Phase 1

- assessment terminal for clinic capture
- provider management console for review and interpretation

### Phase 2

- caregiver and patient client app for home adherence, exercises, and simplified summaries
- baseline and longitudinal monitoring workflow

### Phase 3

- stronger multi-device continuity
- patient-safe communication templates
- remote caregiver escalation workflows
