# Product Requirements Document: Clinic Diagnostic-Adjunct Assessment

## 1. Product Summary

Reflexion Clinic is a provider-supervised software workflow that captures a short multimodal cognitive assessment session and produces an investigational early cognitive impairment risk summary for clinician review.

The first release is not an Alzheimer subtype classifier and not a standalone diagnosis. It is a diagnostic-adjunct workflow intended to support triage, referral, and repeat testing decisions.

## 2. Primary Users

- Geriatricians
- Neurologists
- Memory clinic physicians
- Clinic coordinators
- Research staff running investigational pilots

## 3. Primary Surfaces

- Assessment terminal or smart-mirror style capture surface
  Used by the patient during the structured realtime conversation and full-session recording.
- Provider management console
  Used by clinic staff to launch sessions, review formal outputs, and record final interpretation.

## 4. Core User Story

As a clinician or clinic coordinator, I need to run a structured 8-12 minute assessment, confirm that the captured session is usable, view a calibrated risk estimate with explanation, and decide whether the patient should return to routine follow-up, repeat testing, or a fuller cognitive workup.

## 5. Workflow

1. Staff selects patient and confirms identity.
2. Staff confirms consent state and assessment language.
3. Assessment terminal runs a brief free-speech segment plus structured cognitive prompts.
4. The terminal records the full session and uploads it for formal multimodal analysis.
5. Product computes session QC before inference.
6. Provider console returns:
   - impairment likelihood
   - confidence or calibration band
   - QC flags
   - top contributing feature groups
   - recommended follow-up band
7. Clinician records final interpretation and any reference test results.

## 6. Inputs

- Patient identity and visit metadata
- Consent state and consent version
- Device and site identifiers
- Audio capture for all sessions
- Structured task responses
- Optional facial video in research mode only
- Clinician-provided label and follow-up outcome when available

## 7. Required Outputs

- Session usability verdict:
  - usable
  - usable with caveats
  - unusable
- Early cognitive impairment probability
- Risk band:
  - low
  - elevated
  - high
- QC flags:
  - excessive noise
  - low speech duration
  - clipping or distortion
  - task non-completion
  - identity mismatch
- Top contributing signal groups:
  - speech timing
  - acoustic stability
  - task performance
  - facial expressivity
- Recommended action:
  - routine monitoring
  - repeat assessment
  - clinician review for fuller workup

## 8. Additional Surface Rules

- The assessment terminal must not show raw investigational risk scores to the patient during capture.
- The provider management console is the only surface that can display the full formal clinic assessment by default.
- If a patient-safe summary is ever shown outside the provider console, it must be explicitly provider-approved.

## 9. Non-Goals For Version 1

- Unsupervised consumer use
- Alzheimer versus non-Alzheimer subtype classification
- Fully passive home monitoring
- Autonomous referral without clinician review
- Real-time edge-only inference as a launch blocker

## 10. Product Rules

- No output is shown if consent is missing or withdrawn.
- No risk score is shown if session QC is below minimum threshold.
- The dashboard always displays that the output is for clinician interpretation.
- The dashboard stores both model output and clinician final disposition.
- Patient-level audit history is retained for every viewed output and edited interpretation.

## 11. Performance And Acceptance Targets

- Session completion rate: at least 85 percent in supervised clinics
- QC pass rate: at least 90 percent after staff training
- Identity attribution success: at least 99 percent in supervised use
- Primary model metrics on locked external validation:
  - sensitivity >= 0.80
  - specificity >= 0.80
  - ROC-AUC >= 0.85
- Calibration:
  - expected calibration error within an agreed release threshold

## 12. Required Interfaces

Shared session input:

- `schemas/session-record.schema.json`

Clinic output contract:

- `clinic/intelligence/schemas/clinic-assessment-output.schema.json`

## 13. Open Research Extension

Facial analysis may be collected in research mode, but it does not enter the production claim until it proves incremental value on external validation and subgroup robustness.
