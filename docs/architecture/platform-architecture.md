# Platform Architecture

## 1. Architecture Goal

Build a regulated evidence and product platform that can support:

- clinic-supervised diagnostic-adjunct assessments
- provider-managed home monitoring
- reproducible ML development
- clinical evidence generation
- auditability and controlled model release

## 2. System Domains

### Capture Domain

- Clinic capture rig
  Stable supervised hardware for pilot and provider deployment
- Home pilot rig
  Simpler remote device optimized for repeatability and adherence
- Capture application
  Handles consent check, patient selection, session orchestration, local QC, and secure upload
- Assessment terminal or mirror application
  Patient-facing guided conversation and capture surface used in clinic or repeatable home setups

### Platform Domain

- Identity and enrollment service
- Consent registry
- Session ingestion API
- Raw artifact store for audio, video, and transcripts
- Derived feature store
- Annotation and adjudication workspace
- Audit log service
- Dashboard backend
- Provider management console backend
- Caregiver and patient client backend
- Notification and recommendation service

### ML Domain

- Feature extraction pipeline
- Cross-sectional clinic scoring service
- Longitudinal home risk engine
- Model registry
- Dataset registry
- Evaluation pipeline with locked test management

### Compliance Domain

- Requirements traceability
- Risk management
- Change control
- Model release approval
- Post-market monitoring and incident workflow

## 3. Environment Separation

### Research Environment

- Fast iteration
- Public benchmark data allowed under usage rules
- No production clinical claims

### Validation Environment

- Controlled pilot use
- Frozen data contracts
- Full audit logging
- Restricted model changes

### Locked Release Environment

- Approved release candidates only
- Signed model artifacts
- Controlled configuration
- Formal change review

## 4. Data Flow

1. Patient and visit context are established.
2. Consent state is checked against the active consent registry.
3. Capture app records raw artifacts and immediate QC signals.
4. Raw artifacts and metadata are uploaded to secure storage.
5. Feature extraction jobs create speech, task, interaction, and optional facial features.
6. Session record is assembled under the shared contract.
7. Clinic scoring or home risk scoring runs depending on product mode.
8. Dashboard displays output together with QC and audit metadata.
9. Clinician interpretation and follow-up outcome are written back to the record.
10. Dataset curation pipelines use only approved, versioned snapshots.

## 5. Shared Data Contract

Every session must store:

- consent state and version
- patient, site, and device identifiers
- raw artifact references
- derived features by modality
- quality-control metrics
- model outputs
- clinician labels and follow-up outcomes
- longitudinal state when applicable

Reference schema:

- `schemas/session-record.schema.json`

## 6. System Design Principles

- Speech-first version 1
- Modality additions only when they improve external validation
- One shared platform delivered through provider console, caregiver or patient client app, and assessment terminal surfaces
- No hidden mutable data paths outside the versioned registries
- No inference on sessions that fail minimum QC
- Patient-level train/test isolation enforced in data pipelines
- Product outputs always tied to the exact model and data version used

## 7. Immediate Build Priorities

- Identity and consent workflow
- Session ingestion and raw artifact retention
- Feature extraction service with reproducible versioning
- Provider dashboard shell
- Annotation and clinician label workflow
- Audit trail for every output view and label edit
