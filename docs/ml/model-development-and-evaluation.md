# Model Development And Evaluation Plan

## 1. Modeling Strategy

Reflexion uses two model families with shared upstream data infrastructure:

- **Clinic model**
  Cross-sectional classifier for provider-supervised single-session assessment
- **Home model**
  Longitudinal drift and escalation engine built on repeated measures

These systems may share feature extraction and encoders, but they must not share the same final training objective.

## 2. Dataset Policy

### Public Datasets

Allowed use:

- benchmarking
- feature exploration
- pretraining support
- robustness checks

Not allowed use:

- sole basis for commercial claims
- unqualified inclusion in commercial models where data-use terms prohibit that use

Priority public benchmark sources:

- DementiaBank corpora
- ADReSSo family tasks
- Other approved speech corpora with clear license and access controls

### Proprietary Data

Reflexion-owned or partner-authorized Singapore clinical data is the commercial foundation. It must become the main source of model tuning, calibration, and clinical claims.

Target proprietary dataset by September 2027:

- 150-200 participants
- healthy controls, MCI, and mild dementia
- at least one clinic session per participant
- repeated home data for a monitored subset

### Aspirational Private Research Sources

- ADNI4 remote speech resources such as Storyteller, if access is granted and terms are workable
- Provider-partner datasets gathered under Reflexion-approved protocols

## 3. Label Policy

Ground truth priority:

1. Clinician adjudication
2. Reference neuropsychological assessment
3. Longitudinal clinical follow-up outcome

Disallowed as primary ground truth:

- model predictions
- weak heuristic labels
- public dataset labels with no protocol alignment to the deployed product

## 4. Feature Strategy

### Version 1 Gating Modalities

- speech acoustics
- fluency and pause structure
- interaction timing
- structured task performance

### Research Modality

- facial features and action-unit style expressivity measures

Facial features only move into the product claim after proving incremental value on external validation and subgroup robustness.

## 5. Candidate Model Order

Clinic model sequence:

1. Penalized logistic regression
2. Random forest baseline for comparability
3. XGBoost or LightGBM
4. Calibrated ensemble if justified by validation

Home model sequence:

1. Rules-plus-statistics baseline using deviation, slope, and volatility
2. Bayesian or probabilistic updating across windows
3. Learned temporal models only after sufficient proprietary longitudinal data exists

## 6. Evaluation Rules

- Patient-level isolation across train, validation, and test
- Site holdout for external-like validation where feasible
- Device-version reporting
- Language-stratified reporting
- Age-band and sex subgroup reporting
- No threshold tuning on the locked test set
- Calibration measured on held-out data
- Every release compared to the prior locked baseline

## 7. Primary Metrics

### Clinic Product

- sensitivity
- specificity
- ROC-AUC
- positive predictive value
- negative predictive value
- calibration error
- false-positive rate by subgroup

### Home Product

- alert precision
- false alerts per patient-month
- detection lead time before clinically observed decline
- baseline completion rate
- adherence-adjusted performance

## 8. Release Gates

No model is releasable unless:

- data provenance is documented
- feature generation version is frozen
- train/validation/test split manifest is archived
- subgroup report is produced
- calibration report is produced
- failure analysis is completed
- product owner, ML owner, and clinical owner approve the release

## 9. Required Operational Artifacts

- `templates/model_release_checklist.md`
- `templates/requirements_traceability.csv`
- `templates/risk_register.csv`

## 10. External Constraints

- TalkBank clinical datasets require compliance with access and use rules.
- TalkBank rules explicitly limit commercial use of the data themselves and place restrictions around model inclusion.
- HSA expects lifecycle discipline for software medical devices and has published dedicated SaMD guidance documents.
