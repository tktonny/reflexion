# Product Requirements Document: Home Longitudinal Monitoring

## 1. Product Summary

Reflexion Home is a provider-managed longitudinal monitoring workflow that captures repeated short interactions in the home, establishes an individual baseline, and escalates meaningful drift from personal norm back into provider workflow.

The product is not designed as standalone consumer self-diagnosis. It is a remote monitoring extension linked to a clinic or provider program.

## 2. Primary Users

- Patients enrolled by partner providers
- Caregivers supporting setup or adherence
- Provider care teams reviewing remote risk trends

## 3. Primary Surfaces

- Caregiver and patient client app
  Used for reminders, simplified status, exercises, check-ins, and adherence support.
- Provider management console
  Used for enrollment, trend review, escalation handling, and clinician follow-up.
- Home interaction terminal when needed
  Used when the program requires a more repeatable camera and microphone setup than a handheld phone alone.

## 4. Core User Story

As a provider, I need a remote monitoring workflow that turns repeated short interactions into a baseline, trend view, and escalation signal so that I can identify changes earlier than periodic clinic visits alone.

## 5. Operational Defaults

- Session target length: 2-5 minutes
- Minimum usable cadence: 3 high-quality sessions per week
- Target cadence for better signal: 5 or more short sessions per week
- Baseline completion rule:
  - at least 12 high-quality sessions
  - collected across at least 28 days
  - with activity represented in at least 3 distinct weeks
- Drift assessment rule:
  - no stable drift verdict before 90 days unless there is a sharp persistent deterioration
- Stronger progression evidence window:
  - 6-12 months of repeated data

## 6. Inputs

- Repeated speech interactions
- Interaction timing features
- Optional structured micro-tasks
- Optional facial video in research mode
- Patient adherence and device health events
- Clinician labels from follow-up visits when available

## 7. Required Outputs

- Baseline state:
  - not started
  - building
  - complete
  - stale
- Deviation-from-baseline score
- Trend slope
- Volatility score
- Adherence score
- QC coverage score
- Escalation band:
  - low
  - medium
  - high
- Explainability summary:
  - speech slowing
  - increased hesitation
  - declining structured task results
  - reduced expressivity
  - reduced interaction consistency

## 8. Surface-Specific Delivery Rules

- The provider management console can display full longitudinal drift, baseline, and escalation context.
- The caregiver and patient client app should default to simplified messages such as on-track, repeat check-in needed, or provider review pending.
- The client app may show recommendations, exercises, and adherence coaching without exposing raw investigational risk scores by default.

## 9. Initial Risk Logic For Investigational Use

- **Low**
  Baseline complete, good adherence, deviation is modest, and the trend is flat or recovering.
- **Medium**
  Baseline complete, deviation is moderate or the slope is worsening, but the pattern is not yet persistent enough for high-risk escalation.
- **High**
  Deviation is large and persistent, or a moderate deviation repeats across consecutive windows with worsening slope and good QC coverage.

These thresholds must be tuned on proprietary longitudinal data before commercialization.

## 10. Product Rules

- No longitudinal risk output until baseline completeness criteria are met.
- If adherence or QC coverage is weak, the product shows a monitoring-insufficient status instead of overconfident risk output.
- All medium and high escalations route to provider review, not directly to patient diagnosis.
- The product stores each windowed score together with the underlying coverage and QC context.

## 11. Success Metrics

- Weekly active adherence above 70 percent in enrolled users
- Baseline completion above 75 percent for newly enrolled users
- False alert burden kept below an agreed patient-month threshold
- Provider review workflow supports action within clinically acceptable response times

## 12. Required Interfaces

Shared session input:

- `schemas/session-record.schema.json`

Home output contract:

- `schemas/home-risk-output.schema.json`
