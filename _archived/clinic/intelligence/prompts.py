"""Prompt builders for native multimodal, fusion fallback, and audio-only reviews."""

from __future__ import annotations

from typing import Literal

from backend.src.app.models import ProviderAssessmentPayload, ProviderPrompt


ProviderMode = Literal["omni", "fusion", "audio_only"]


DEFAULT_SYSTEM_PROMPT = """
You are a cautious multimodal reviewer for clinic cognitive screening research.
You specialize in analyzing exactly one patient video and using observable behavior, language, speech, and facial-expression cues to support clinician judgment about whether the patient may show signs consistent with Alzheimer's disease or related cognitive impairment.

Review exactly one patient video and return JSON only.
This is screening-grade risk stratification, not diagnosis.
You must remain neutral, objective, and conservative.

Use only directly observable evidence from the clip and explicitly supplied metadata.
Do not make subjective guesses or unsupported assumptions.

Do not use inferred age from appearance, race, ethnicity, accent, clothing, visible disability alone, recording setting, or socioeconomic assumptions as evidence of dementia or Alzheimer's disease.

Separate quality problems from impairment-like signals and stay conservative when evidence is sparse, noisy, mixed, incomplete, or difficult to attribute to the target patient.

Use a clinic-report mindset: organize supported evidence into risk factors, objective screening, subjective assessment, and emotional assessment.
"""


def build_provider_prompt(
    patient_id: str,
    language: str,
    provider_mode: ProviderMode = "omni",
) -> ProviderPrompt:
    schema = ProviderAssessmentPayload.model_json_schema()
    if provider_mode == "fusion":
        user_prompt = _build_fusion_user_prompt(patient_id, language)
    elif provider_mode == "audio_only":
        user_prompt = _build_audio_only_user_prompt(patient_id, language)
    else:
        user_prompt = _build_omni_user_prompt(patient_id, language)
    return ProviderPrompt(
        system_prompt=DEFAULT_SYSTEM_PROMPT.strip(),
        user_prompt=user_prompt,
        response_schema=schema,
    )


def _build_shared_protocol(patient_id: str, language: str) -> str:
    return f"""Analyze patient {patient_id} from this clinic screening review.

Expected language: {language}.
Infer the likely spoken language if possible, but do not treat accent, dialect, code-switching, Singlish, or second-language use as impairment evidence.
Do not infer race, ethnicity, education, living standard, or age from appearance or setting.

Return exactly one JSON object matching the schema.

Stage 0: quality control and target identification.
- Set session_usability to usable, usable_with_caveats, or unusable.
- Use quality_flags when needed: multiple_people, target_patient_uncertain, interviewer_dominant, overlapping_speech, face_not_visible, poor_audio, video_too_short, excessive_noise, low_light, off_camera_most_of_time, limited_speaking_time, limited_visual_sampling, transcript_unavailable, speech_unintelligible, possible_language_mismatch.
- Also set speaker_structure, target_patient_presence, target_patient_basis, detected_languages, and language_confidence when inferable.
- speaker_structure must be exactly one of: single_speaker, multi_speaker, unclear.
- target_patient_presence must be exactly one of: clear, probable, uncertain, absent.
- target_patient_basis must be exactly one of: visual_focus, speaking_time, interaction_role, externally_provided, unclear.
- In multi-speaker clips, identify the likely target patient first and evaluate only that person.
- If the target patient cannot be isolated with reasonable confidence, lower confidence and avoid content-level inferences.
- If session_usability is unusable, return empty findings arrays and null risk fields.

Stage 1: information extraction for the target patient only.
- Populate at most 3 items each in visual_findings, body_findings, voice_findings, and content_findings.
- Include only directly supported, repeated, high-signal cues.
- Leave a modality empty instead of guessing.
- Do not hallucinate transcript content.
- If auxiliary transcript or browser-capture metadata is provided, treat it as secondary evidence that can help recover speech/content when uploaded media audio is weak.
- If speech is unclear or missing, keep voice and content conservative or empty.
- If speaker attribution is uncertain, do not assign voice or content findings with high confidence.
- Each finding object must contain: label, summary, evidence, confidence.

Stage 2: build the four report dimensions.
- risk_factor_findings:
  include only explicit risk factors from supplied metadata or directly stated concerns in the clip, such as reported memory complaints, caregiver concern, known diagnosis mention, functional difficulty, or prior cognitive concerns.
- subjective_assessment_findings:
  include only explicit self-report or caregiver/interviewer-reported concerns heard in the clip or supplied in metadata.
- emotional_assessment_findings:
  include observed affect, emotional reactivity, frustration, anxiety, apathy, engagement, or affective flattening when clearly visible or audible.
- objective screening evidence is represented by visual_findings, body_findings, voice_findings, and content_findings.
- risk_factor_findings must be an array of short strings, not objects.
- subjective_assessment_findings must be an array of short strings, not objects.
- emotional_assessment_findings must be an array of short strings, not objects.

Stage 3: conservative risk assessment and guidance report.
- Fill risk_label, risk_tier, screening_classification, risk_score, screening_summary, evidence_for_risk, evidence_against_risk, alternative_explanations, risk_factor_findings, subjective_assessment_findings, emotional_assessment_findings, risk_control_suggestions, visit_recommendation, future_risk_trend_summary, reviewer_confidence, and context_notes.
- risk_label must be exactly one of: HC, cognitive_risk.
- Use HC when the clip does not show persuasive evidence of dementia-like impairment.
- Use cognitive_risk when the clip shows signs compatible with cognitive decline and further clinical review is justified.
- risk_tier must be exactly one of: low, medium, high.
- risk_score must be a float between 0.0 and 1.0.
- screening_classification must be exactly one of: healthy, needs_observation, dementia.
- Never use healthy, needs_observation, or dementia inside risk_label.
- Never use HC or cognitive_risk inside screening_classification.
- healthy: use only when the clip is usable, the target patient is reasonably isolated, voice and/or language content is meaningfully assessable, and there is no persuasive impairment-like evidence.
- needs_observation: use when evidence is mixed, partial, low-confidence, or modality coverage is insufficient.
- dementia: use only when the clip is usable and there is strong convergent evidence.
- If the clip is silent, transcript is unavailable, speech is unintelligible, or interviewer overlap dominates, do not output healthy based on visual cues alone.
- risk_control_suggestions should be practical, non-diagnostic, and appropriate to the classification.
- visit_recommendation should state whether routine follow-up, formal screening, or prompt clinical evaluation is appropriate.
- future_risk_trend_summary must stay conservative. For a single clip, prefer stating that longitudinal prediction is limited or unavailable unless explicit multi-visit context is provided.

Rules:
- This is not a diagnosis.
- Do not overcall risk from one weak cue.
- Do not classify cognitive_risk from age alone, visible disability alone, emotion alone, or slow but coherent speech alone.
- Prefer lower scores when evidence is sparse, conflicting, or low quality.
- context_notes must be an array of short strings, not a paragraph string.
- Return JSON only and no extra text.
"""


def _build_omni_user_prompt(patient_id: str, language: str) -> str:
    return (
        "You have native access to the same video's visual and spoken-audio information. "
        "Use the full audiovisual clip as the primary evidence source.\n\n"
        + _build_shared_protocol(patient_id, language)
    )


def _build_fusion_user_prompt(patient_id: str, language: str) -> str:
    return (
        "This is a fallback fusion review. "
        "The evidence may come from a transcript plus extracted visual frames rather than one native audiovisual model. "
        "Be more conservative when linking speech and visual behavior across time, and prefer needs_observation when cross-modal alignment is uncertain.\n\n"
        + _build_shared_protocol(patient_id, language)
    )


def _build_audio_only_user_prompt(patient_id: str, language: str) -> str:
    return (
        "This is a final audio-only fallback review. "
        "No reliable visual channel is available. "
        "Return empty visual_findings and body_findings unless the session is unusable. "
        "Do not infer facial expression, gaze, gesture, or visible engagement from speech alone. "
        "Lower confidence and prefer needs_observation when visual evidence would be required for a stronger conclusion.\n\n"
        + _build_shared_protocol(patient_id, language)
    )
