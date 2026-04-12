# Video Assessment Prompt Design

This note explains the prompt strategy used in the clinic video demo.

## Goal

The prompt is designed to help a multimodal foundation model review a single patient video and produce a **screening-oriented** assessment, not a diagnosis.

The output is intentionally split into two layers:

### 1. Information extraction layer

- `visual_findings`
- `body_findings`
- `voice_findings`
- `content_findings`
- `speaker_structure`
- `target_patient_presence`
- `target_patient_basis`
- `detected_languages`
- `language_confidence`

### 2. Risk assessment layer

- `screening_classification`
- `risk_score`
- `risk_label`
- `risk_tier`
- `screening_summary`
- `evidence_for_risk`
- `evidence_against_risk`
- `alternative_explanations`
- `reviewer_confidence`
- `context_notes`
- `quality_flags`
- `session_usability`

## Why The Prompt Looks At These Signals

### Speech and language

Recent reviews consistently report that dementia-related and MCI-related speech changes often include:

- slower speech rate
- more frequent and longer pauses or hesitations
- flatter prosody
- word-finding difficulty
- reduced lexical diversity
- reduced coherence and informativeness
- shorter or less complex sentences

These patterns are summarized in the speech-language literature and review articles used for this project.

### Short conversational samples

Short conversational samples can still be useful. A 2025 clinical study reported strong discrimination between cognitively declined and cognitively normal groups using one-minute conversational voice samples, with extracted voice features such as silent intervals, F0, MFCCs, and HuBERT-based representations.

### Face and visible behavior

Face-related video features are less mature than speech-based markers, but there is evidence that facial features can contribute useful signal. A 2023 study using the PROMPT interview dataset reported that automated face-related features could distinguish dementia from healthy controls, with the strongest results coming from HOG-based features and weaker but still useful results from action units and face mesh features.

## Prompt Guardrails

The prompt deliberately separates **quality problems** from **impairment-like signals**.

It also separates:

- extraction of multimodal evidence
- conservative screening judgment based on that evidence

The risk layer now explicitly asks the model to expose:

- a direct demo classification
- strongest evidence supporting risk
- strongest evidence against overcalling risk
- plausible non-cognitive alternative explanations
- reviewer confidence

This is intended to make the output safer and easier to discuss in demo settings, while still remaining non-diagnostic.

The direct demo classification is:

- `healthy`
- `needs_observation`
- `dementia`

This field is intentionally stricter than low-risk scoring alone. In particular, the prompt now tells the model not to output `healthy` for silent or language-unassessable clips based on visual cues alone.

The prompt also explicitly handles:

- multi-speaker clips and likely target-patient selection
- language detection as a quality/context aid rather than a penalty
- exclusion of inferred race, SES, and guessed age from evidence

The model is instructed to:

- avoid using demographics, stereotypes, or metadata as evidence
- avoid overcalling risk from a single weak cue
- degrade to `usable_with_caveats` or `unusable` when the clip is low quality
- treat `cognitive_risk` as a screening label that suggests further review, not a diagnosis

This is important because the speech-language literature also warns that dataset imbalance, demographic confounding, and task design can inflate apparent performance.

## Source Anchors

- [A systematic literature review of automatic Alzheimer’s disease detection from speech and language](https://pubmed.ncbi.nlm.nih.gov/32929494/)
- [Noninvasive automatic detection of Alzheimer’s disease from spontaneous speech: a review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10484224/)
- [Detecting Dementia from Face-Related Features with Automated Computational Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10376259/)
- [Utility of artificial intelligence-based conversation voice analysis for detecting cognitive decline](https://pmc.ncbi.nlm.nih.gov/articles/PMC12129157/)
- [MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer’s Detection](https://arxiv.org/html/2502.19208v1)
