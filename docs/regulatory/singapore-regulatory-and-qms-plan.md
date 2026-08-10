# Singapore Regulatory And QMS Plan

## 1. Regulatory Posture

Reflexion should operate from the assumption that its provider-facing software is within the scope of medical device regulation in Singapore if it is intended to analyze patient data for clinical assessment. HSA states that standalone software may fall within the definition of a medical device and is deemed an active medical device when classified in its own right.

Because the product is intended to support clinical assessment, Reflexion should build the company as though regulated product registration, dealer licensing, clinical evaluation, change control, and post-market obligations will apply.

## 2. Intended Use Statement For Planning

Reflexion is software intended to assist trained healthcare professionals in assessing the likelihood of early cognitive impairment based on structured speech-first assessment sessions and related session quality data. It is not intended to provide a standalone diagnosis or determine Alzheimer subtype.

This intended use should remain frozen unless changed through formal design control.

## 3. Planning Assumptions

- Singapore first
- provider-supervised use first
- clinical decision support or diagnostic-adjunct posture
- investigational pilots before commercial supply
- no standalone consumer diagnostic claim in version 1

## 4. HSA-Aligned Workstreams

### Design And Development Controls

- product requirements
- software architecture
- verification and validation planning
- risk management
- configuration and change control

### Quality Management System Foundation

- document control
- training records
- CAPA-ready issue management
- complaint and incident workflow
- supplier control where relevant

### Clinical Evaluation

- intended use and indications
- evidence plan
- performance metrics
- subgroup analysis
- clinical benefit-risk narrative

### SaMD Change Management

- model release approval
- data drift monitoring
- trigger criteria for retraining or relabeling
- change impact assessment

## 5. PDPA Operating Requirements

PDPC states that organizations must meet notification, consent, purpose limitation, accuracy, protection, and accountability obligations for personal data. Reflexion should therefore implement:

- named data protection officer responsibility
- clear consent and purpose wording
- withdrawal handling
- access and correction workflow
- minimum necessary data collection
- security controls for personal and health-related data

## 6. Near-Term Deliverables

**By September 30, 2026**

- intended use frozen
- requirements traceability live
- risk register live
- consent version control live
- audit logging live
- model release checklist live
- regulatory gap assessment completed

**By September 30, 2027**

- clinical evaluation draft assembled
- locked model specification prepared
- post-market monitoring plan draft completed
- supplier and deployment controls defined

## 7. What Not To Do

- do not make public claims beyond the frozen intended use
- do not ship silent model updates
- do not rely on ad hoc notebooks as evidence of validation
- do not treat public benchmark performance as registration-grade evidence

## 8. Official Reference Anchors

- HSA medical devices regulatory overview
- HSA guidance documents catalog, including:
  - GL-04-R4 Software Medical Devices - A Life Cycle Approach
  - GL-07-R2 Risk Classification SaMD-CDSS
  - GN-20-R2 Clinical Evaluation
  - GN-37-R1 Change Management Program for ML-enabled SaMD
- PDPC PDPA overview and Data Protection Obligations pages
