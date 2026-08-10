# Reflexion Startup Blueprint

## 1. Company Thesis

Reflexion should be built as a provider-first cognitive assessment company that earns trust through clinical evidence and proprietary longitudinal data. The clinic product funds and validates the business; the home product becomes the longer-term moat.

Two products must be developed in parallel but treated as distinct systems:

- **Clinic diagnostic-adjunct workflow**
  A one-session, provider-supervised assessment that estimates early cognitive impairment risk and recommends next clinical follow-up.
- **Home longitudinal monitoring workflow**
  A repeated-measure system that builds a personal baseline, measures drift from that baseline, and escalates concerning trajectories back into provider workflow.

## 2. Strategic Decisions Frozen As Of April 5, 2026

- Market entry is Singapore-first.
- The first paying customer is a provider, not a direct-to-consumer family purchaser.
- The first wedge is an investigational clinic deployment that matures into a regulated provider product.
- The first regulated claim targets early cognitive impairment detection under clinician supervision.
- Speech is the gating modality for version 1.
- Facial analysis remains a research stream until Reflexion owns sufficient proprietary multimodal data.
- Public datasets such as DementiaBank and ADReSSo are used for benchmarking, feature exploration, and pretraining support only.

## 3. Product Portfolio

### Clinic Product

Purpose:

- Create a provider-usable intake or triage workflow.
- Generate high-quality labeled data under clinic supervision.
- Establish the evidence package for regulatory submission.

Core output:

- Calibrated impairment likelihood score
- Quality-control flags
- Top contributing signals
- Recommended follow-up band:
  - routine follow-up
  - repeat assessment
  - refer for fuller cognitive workup

### Home Product

Purpose:

- Build an individual baseline over repeated natural interactions.
- Detect drift, decline slope, instability, and persistent deviation.
- Route escalations back into provider workflow.

Core output:

- Baseline completeness
- Deviation-from-personal-norm score
- Trend slope
- Volatility score
- Low / medium / high escalation status

## 4. Phase Plan With Calendar Dates

### Phase 0: Evidence Platform Foundation

**Dates:** April 2026 to September 2026

Goals:

- Freeze intended use and labeling.
- Stand up a minimum viable quality system.
- Convert the MVP into a reproducible evidence-collection platform.
- Lock the session data contract and annotation workflow.

Exit criteria:

- Consent, identity, audit logging, QC, and raw artifact retention exist in product scope.
- Training and evaluation are reproducible with versioned datasets and model artifacts.
- Provider-facing clinic workflow is specified and demoable.

### Phase 1: Prospective Pilot Buildout

**Dates:** October 2026 to September 2027

Goals:

- Run the first Singapore pilot with provider-grade labels.
- Collect the first proprietary speech-first dataset.
- Train cross-sectional baselines with calibration and subgroup analysis.
- Deploy investigational dashboards to pilot sites.

Exit criteria:

- 150-200 participant dataset targeted and materially underway.
- All clinic participants have clinician labels and reference test linkage.
- At least one pilot partner uses the dashboard in live workflow.

### Phase 2: External Validation And Home Monitoring Productization

**Dates:** October 2027 to March 2029

Goals:

- Expand to multi-site validation.
- Freeze a locked model specification for the first submission path.
- Formalize home longitudinal monitoring as a provider-managed program.
- Add English and Mandarin production support.

Exit criteria:

- External site holdout results meet locked evidence thresholds.
- Home program has defined baseline, escalation, and adherence logic.
- Clinical, technical, and post-market plans are draft-complete.

### Phase 3: Regulated Launch Preparation

**Dates:** April 2029 to March 2030

Goals:

- Submit the first regulated provider product if evidence supports it.
- Launch the clinic product with controlled rollout.
- Commercialize home monitoring as a provider extension.

Exit criteria:

- Blinded external validation supports the claim.
- Paid deployment economics are proven.
- Post-market monitoring and model change processes are live.

## 5. Team Design For A Lean Seed Company

Minimum team:

- CEO / clinical-commercial founder
- Product and operations lead
- Full-stack platform engineer
- ML and data engineer
- Clinical research and data operations manager
- Fractional regulatory / quality lead

Add next:

- Speech scientist or applied ML researcher
- Site implementation or customer success lead
- Firmware / edge engineer only when clinic workflow is stable

## 6. Company-Level Milestone Gates

### Technical Gate

- Session completion rate above 85 percent in supervised use
- Audio QC pass rate above 90 percent
- Reliable patient, site, and device attribution
- Feature drift monitored by site, device, language, and software version

### ML Gate

- Patient-level train/test isolation
- External site holdout
- Calibration review on the locked test set
- Subgroup reporting by age band, sex, language, and device revision
- Comparative ablation for speech-only, speech-plus-task, and speech-plus-face

### Clinical Gate

- Blinded comparison against clinician adjudication and reference cognitive measures
- Sensitivity, specificity, and ROC-AUC as primary metrics
- Prospective workflow data showing usability and acceptable false-positive burden

### Business Gate

- At least one pilot converts to paid provider use
- A repeatable procurement and deployment model exists
- Unit economics are supportable on a per-site or per-patient basis

## 7. Non-Negotiable Rules

- Do not market version 1 as a standalone diagnosis.
- Do not anchor the commercial product to public research datasets alone.
- Do not merge the clinic classifier and home longitudinal engine into one objective.
- Do not scale hardware complexity before data quality and evidence quality are solved.
- Do not introduce facial analysis into the clinical claim until Reflexion has demonstrated external validation benefit on proprietary data.

## 8. Core References

- Reflexion internal tech-lead document
- Tigerlaunch prototype baseline
- HSA medical device guidance pages and SaMD guidance catalog
- PDPC PDPA obligations pages
- TalkBank usage rules
- ADNI4 remote and Storyteller references
