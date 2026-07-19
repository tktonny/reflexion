"""Verification-only Qwen Omni audio classifier with dialogue-role guidance."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Literal

from backend.src.app.core.json_utils import parse_json_object
from backend.src.app.core.errors import ProviderError
from backend.src.app.models import (
    ProviderAssessmentPayload,
    ProviderCapabilities,
    ProviderContext,
    ProviderPrompt,
    ProviderRawResult,
    normalize_quality_flags,
)
from clinic.configs.settings import get_settings
from clinic.intelligence.prompts import build_provider_prompt
from clinic.intelligence.providers.base import BaseProvider
from verification.config import get_verification_settings
from verification.models import VerificationRecord
from verification.qwen_audio_only import QwenAudioOnlyVerifier


PromptReviewLanguage = Literal["en", "zh"]


def detect_review_language(
    *,
    asr_transcript: str | None,
    expected_language: str,
) -> PromptReviewLanguage:
    transcript = str(asr_transcript or "").strip()
    cjk_chars = sum(1 for char in transcript if _is_cjk_char(char))
    latin_chars = sum(1 for char in transcript if char.isascii() and char.isalpha())

    if cjk_chars >= 8 and cjk_chars >= latin_chars:
        return "zh"
    if latin_chars >= 20 and latin_chars >= cjk_chars * 2:
        return "en"

    normalized_expected = expected_language.strip().lower()
    if normalized_expected.startswith(("zh", "cmn", "yue")):
        return "zh"
    return "en"


def build_qwen_omni_audio_context_note(
    record: VerificationRecord,
    *,
    review_language: PromptReviewLanguage,
) -> str:
    notes = [
        "This patient only provides conversational audio in this verification sample.",
        "The video portion is missing for this patient and should be treated as unavailable rather than as a recording failure.",
        "The audio may contain both the patient and an examiner or tester because this is a dialogue-style screening interaction.",
        "You must identify which speech belongs to the examiner and which belongs to the patient, then evaluate only the patient.",
        "Brief examiner prompts, acknowledgements, or task instructions are expected and do not by themselves make the patient unclear.",
        "The patient is the primary evaluation subject and main respondent in this recording, so you may use externally_provided or interaction_role to isolate the patient when needed.",
        "You must first internally transcribe or summarize the patient speech from the raw audio before making the assessment.",
        "In dialogue-style screening tasks, examiner prompts are neutral context. The screening signal comes from the patient's own replies, especially their coherence, task relevance, lexical retrieval, repetitions, pauses, and narrative organization.",
        "Use the native raw audio directly. If the patient speech is understandable enough to summarize, do not mark transcript_unavailable or speech_unintelligible.",
        "Return empty visual_findings and body_findings because video is unavailable, not because the session is unusable.",
        "Do not mark the session unusable only because video is missing or because the examiner speaks briefly.",
        "If the patient speech is intelligible enough to reconstruct multiple clauses, do not mark transcript_unavailable or speech_unintelligible just because the audio is noisy or accented.",
    ]
    if task_hint := _infer_task_hint(record):
        notes.append(f"Known task context: {task_hint}.")

    risk_features = [
        "word-finding pauses, circumlocution, or frequent filler terms such as thing, stuff, that one",
        "semantic vagueness, low-information descriptions, or generic naming instead of specific scene details",
        "repetition, self-correction, restarting, or abandoned utterances",
        "disorganized sequencing, weak causal links, or failure to connect the main events in the scene",
        "missing core task elements, partial descriptions, or weak informativeness despite prompting",
        "examiner dependence, needing repeated prompting, or giving very short answers to a rich picture-description task",
        "tangential additions, uncertainty, contradictions, or confusion about roles and actions in the scene",
    ]

    decision_rules = build_language_specific_decision_rules(
        record,
        review_language=review_language,
    )

    lines = [f"- {note}" for note in notes]
    lines.append(f"- ASR-confirmed operating language for prompt selection: {_language_label(review_language)}.")
    lines.append("- Language-specific calibration:")
    lines.extend(
        f"  - {rule}" for rule in build_language_specific_calibration(record, review_language=review_language)
    )
    lines.append("- High-signal patient-audio risk features to look for:")
    lines.extend(f"  - {feature}" for feature in risk_features)
    lines.append("- Decision guidance for this verification setting:")
    lines.extend(f"  - {rule}" for rule in decision_rules)
    return "\n".join(lines)


def build_qwen_omni_audio_prompt(record: VerificationRecord) -> ProviderPrompt:
    return build_qwen_omni_audio_prompt_with_asr_assist(record, asr_transcript=None)


def build_qwen_omni_audio_prompt_with_asr_assist(
    record: VerificationRecord,
    *,
    asr_transcript: str | None,
) -> ProviderPrompt:
    base_prompt = build_provider_prompt(record.case_id, record.language, provider_mode="audio_only")
    review_language = detect_review_language(
        asr_transcript=asr_transcript,
        expected_language=record.language,
    )
    asr_note = build_qwen_omni_asr_assist_note(
        asr_transcript,
        review_language=review_language,
    )
    return ProviderPrompt(
        system_prompt=base_prompt.system_prompt,
        user_prompt=(
            f"{base_prompt.user_prompt}\n\n"
            "This verification run uses qwen_omni on native conversational audio.\n"
            f"{build_qwen_omni_audio_output_contract()}\n\n"
            f"{build_qwen_omni_audio_few_shot_examples(record, review_language=review_language)}\n\n"
            f"{asr_note}\n\n"
            f"{build_qwen_omni_audio_context_note(record, review_language=review_language)}"
        ),
        response_schema=base_prompt.response_schema,
    )


def build_qwen_omni_audio_output_contract() -> str:
    return "\n".join(
        [
            "Before final classification, explicitly reconstruct the patient-only content from the dialogue audio.",
            "Return exactly one JSON object.",
            "Keep all normal clinic assessment fields from the original audio_only schema.",
            "For verification, also include these extra top-level fields:",
            '- "patient_only_transcript": a short patient-only reconstructed transcript or faithful summary of what the patient said.',
            '- "speaker_turn_summary": an array of short strings describing the likely examiner turns and patient turns in order.',
            '- "patient_cue_summary": an array of short strings capturing the main patient-language cues you used for the decision.',
            "Use the patient-only reconstruction and cue summary first, then decide the final risk fields.",
        ]
    )


def build_qwen_omni_asr_assist_note(
    asr_transcript: str | None,
    *,
    review_language: PromptReviewLanguage,
) -> str:
    cleaned = str(asr_transcript or "").strip()
    if not cleaned:
        cleaned = "[ASR did not recover a usable draft transcript.]"
    return "\n".join(
        [
            "Auxiliary ASR evidence:",
            f"- Use the ASR-confirmed language for prompt calibration: {_language_label(review_language)}.",
            "- A first-pass ASR transcript is provided below as noisy support for low-quality audio.",
            "- Use the raw audio as primary evidence and the ASR text as auxiliary evidence.",
            "- The ASR draft may mix examiner and patient turns together; separate them before final assessment.",
            "- Preserve meaningful hesitations, fillers, restarts, repetitions, and pause markers from the ASR draft when reconstructing the patient-only transcript.",
            "- If the ASR draft clearly contains patient speech, do not mark transcript_unavailable unless the content is still unusable after cross-checking the audio.",
            "ASR draft transcript:",
            cleaned,
        ]
    )


def build_qwen_omni_audio_few_shot_examples(
    record: VerificationRecord,
    *,
    review_language: PromptReviewLanguage,
) -> str:
    task_hint = _infer_task_hint(record) or "structured dialogue task"
    if review_language == "zh":
        return "\n".join(
            [
                "Few-shot contrast examples for this verification setting:",
                f"Example A (more likely HC in a {task_hint}):",
                '- patient_only_transcript: "阿嬷带孙子捞鱼，这边两个小朋友在玩骰子，那边有人偷拿背包里的东西，冰淇淋掉下来，小朋友在旁边看。" ',
                '- patient_cue_summary: ["scene elements are intelligible", "topic-chaining style is understandable", "some repetition or discourse particles are normal colloquial Mandarin", "multiple actors and actions are identified without abandonment"]',
                '- likely decision: risk_label=HC, screening_classification=healthy or needs_observation, low risk_score',
                "Example B (more likely cognitive_risk in a similar Chinese scene-description task):",
                '- patient_only_transcript: "这个...那个...小孩在那边...嗯...不知道...这个人那个...就这样。" ',
                '- patient_cue_summary: ["severe vague wording blocks reconstruction", "abandoned clauses", "cannot maintain a stable scene description", "multiple missing core task elements", "examiner would need to rescue the narrative"]',
                '- likely decision: risk_label=cognitive_risk, screening_classification=needs_observation or dementia, low-to-medium risk_score',
                "Important contrast for Chinese: discourse particles, topic-comment phrasing, repeated nouns, omitted subjects, and generic referents such as 这个/那个 do not by themselves indicate impairment when the overall scene remains understandable.",
            ]
        )
    if "fluency" in task_hint:
        return "\n".join(
            [
                "Few-shot contrast examples for this verification setting:",
                "Example A (more likely HC in an English verbal-fluency task):",
                '- patient_only_transcript: "apple, orange, banana, grapes, peaches, pears, watermelon... snake, sun, soup, sandwich, shoes."',
                '- patient_cue_summary: ["good item productivity", "category stays on target", "little repetition", "patient sustains the task without rescue"]',
                '- likely decision: risk_label=HC, screening_classification=healthy or needs_observation, low risk_score',
                "Example B (more likely cognitive_risk in an English verbal-fluency task):",
                '- patient_only_transcript: "apple... uh... orange... apple... um... soup... I can\'t think...".',
                '- patient_cue_summary: ["low productivity", "repetition", "long pauses", "category drift", "retrieval difficulty"]',
                '- likely decision: risk_label=cognitive_risk, screening_classification=needs_observation or dementia, low-to-medium risk_score',
                "Important contrast for fluency: count the patient output quality and productivity. A few correct words are not enough for HC if the task quickly breaks down.",
            ]
        )
    if "sentence" in task_hint:
        return "\n".join(
            [
                "Few-shot contrast examples for this verification setting:",
                "Example A (more likely HC in an English sentence-construction task):",
                '- patient_only_transcript: "The child went to the hospital. It is a cold winter day. The doctor sat in the chair. The bureau drawer is open."',
                '- patient_cue_summary: ["task-compliant sentence generation", "clear transformation beyond bare prompt echo", "stable fluency", "no retrieval failure"]',
                '- likely decision: risk_label=HC, screening_classification=healthy or needs_observation, low risk_score',
                "Example B (more likely cognitive_risk in an English sentence-construction task):",
                '- patient_only_transcript: "Child... hospital... the child, uh, hospital. Cold winter. Doctor sit chair."',
                '- patient_cue_summary: ["reduced sentence construction", "prompt-bound wording", "hesitation", "simplified or incomplete syntax", "weak self-generated language"]',
                '- likely decision: risk_label=cognitive_risk, screening_classification=needs_observation or dementia, low-to-medium risk_score',
                "Important contrast for sentence tasks: short answers are normal, but merely echoing the prompt words with little self-generated structure is weaker evidence than fluent sentence construction.",
            ]
        )
    return "\n".join(
        [
            "Few-shot contrast examples for this verification setting:",
            f"Example A (more likely HC in a {task_hint}):",
            '- patient_only_transcript: "The boy is reaching for the cookies, the stool is tipping, the mother is at the sink, and the water is running over. That is about all I see."',
            '- patient_cue_summary: ["brief but accurate scene description", "intelligible speech", "core events identified", "limited elaboration alone is not enough for cognitive_risk"]',
            '- likely decision: risk_label=HC, screening_classification=healthy or needs_observation, low risk_score',
            "Example B (more likely cognitive_risk in a similar task):",
            '- patient_only_transcript: "Uh... the boy... the cookies... I cannot think... the water is running... and the mother... uh..."',
            '- patient_cue_summary: ["word-finding pauses", "vague low-information wording", "fragmented organization", "abandoned or restarted utterances", "examiner-dependent continuation"]',
            '- likely decision: risk_label=cognitive_risk, screening_classification=needs_observation or dementia, low-to-medium risk_score',
            "Important contrast for English: a short but correct response can still be HC, especially in constrained tasks. The stronger signal is whether the patient can sustain the task without clear breakdown, not whether the response is verbose.",
        ]
    )


def build_language_specific_calibration(
    record: VerificationRecord,
    *,
    review_language: PromptReviewLanguage,
) -> list[str]:
    task_hint = _infer_task_hint(record) or ""
    if review_language == "zh":
        calibration = [
            "In Mandarin or Taiwan-influenced spontaneous speech, repetitions, discourse particles such as 嗯/啊/啦/哦, topic-comment phrasing, omitted subjects, and generic referents such as 这个/那个 are often normal and should not alone trigger cognitive_risk.",
            "Chinese scene descriptions often list visible people and actions without strong causal narration; listing several correct scene elements in an intelligible way can still be compatible with HC.",
            "Do not treat accent, dialect mixture, Taiwanese lexical choices, or colloquial grammar as dementia evidence.",
            "For Chinese HC, require stronger convergent cues before calling cognitive_risk: repeated reconstruction failure, obvious naming collapse, major contradictions, severe abandonment, or pervasive unintelligibility not explained by recording quality.",
            "If the patient remains understandable and names multiple relevant scene elements, prefer HC unless impairment-like cues are clearly stronger than ordinary colloquial speech patterns.",
        ]
    else:
        calibration = [
            "In English cookie-description tasks, brief but accurate content, mild uncertainty, or ending with phrases like 'that's all' are not enough by themselves for cognitive_risk.",
            "Do not overcall cognitive_risk from limited elaboration alone when the patient still names core actors, actions, and relationships intelligibly.",
            "English dementia samples can still mention many correct scene elements; look for breakdowns in organization, retrieval, sustained fluency, tangentiality, or task control rather than keyword count alone.",
            "In structured fluency and sentence tasks, correct completion is positive evidence, but low productivity, repeated restarts, very formulaic reuse of the prompt, or heavy examiner scaffolding can still support cognitive_risk when clearly present.",
            "In English structured tasks, carefully separate examiner examples or task instructions from the patient's actual generated responses before judging quality.",
        ]

    if "fluency" in task_hint:
        calibration.append(
            "For fluency tasks, judge productivity, pauses, repetitions, category drift, and restarts; do not decide from a few correct items alone."
        )
    if "sentence" in task_hint:
        calibration.append(
            "For sentence tasks, brief answers are expected, so brevity alone is not impairment. Only up-rank risk when the speech also shows hesitancy, initiation failure, formulaic minimality, or clear retrieval difficulty."
        )
    if any(token in task_hint for token in ("cookie", "scene description", "market", "park", "daddy")):
        calibration.append(
            "For scene-description tasks, correct identification of several actors, actions, and objects is meaningful counterevidence; do not ignore that counterevidence just because the narrative is not elegant."
        )
    return calibration


def build_language_specific_decision_rules(
    record: VerificationRecord,
    *,
    review_language: PromptReviewLanguage,
) -> list[str]:
    task_hint = _infer_task_hint(record) or ""
    if review_language == "zh":
        rules = [
            "For Chinese audio, do not assign cognitive_risk from repetition, vague pronouns, fillers, or short clause chains alone if the patient remains understandable and identifies multiple relevant scene elements.",
            "Use cognitive_risk in Chinese only when multiple stronger cues co-occur, such as persistent naming failure, major loss of referential clarity, repeated abandonment, severe disorganization, or content that cannot be reliably reconstructed.",
            "When the evidence is mixed but still understandable in Chinese, prefer risk_label=HC with screening_classification=needs_observation rather than cognitive_risk.",
            "If you can reconstruct who is doing what in several parts of the scene, that is substantial counterevidence against cognitive_risk in Chinese HC screening clips.",
        ]
    else:
        rules = [
            "cognitive_risk does not require dementia-level certainty, but it should be supported by more than mere brevity, mild uncertainty, or a non-verbose speaking style.",
            "For English picture description, do not assign cognitive_risk when the patient is intelligible, task-relevant, and names core scene elements unless there are additional breakdown cues.",
            "Use cognitive_risk when several English impairment-like cues converge: retrieval problems, repeated restarts, examiner rescue, tangential drift, weak task control, or clear failure to cover core task demands.",
            "When evidence is mixed but the patient remains coherent and task-relevant in English, prefer HC or HC plus needs_observation rather than cognitive_risk.",
        ]

    if "fluency" in task_hint:
        rules.append(
            "In fluency tasks, low item count, repetition, long pauses, semantic drift, or restarting are stronger evidence than surface correctness."
        )
    if "sentence" in task_hint:
        rules.append(
            "In sentence tasks, do not call HC only because the patient formed grammatical sentences; also consider initiation, flexibility, and whether the response is merely a minimal echo of the prompt."
        )
    return rules


def _language_label(review_language: PromptReviewLanguage) -> str:
    return "Chinese" if review_language == "zh" else "English"


def _infer_task_hint(record: VerificationRecord) -> str | None:
    source_relative_path = str(record.metadata.get("source_relative_path", "")).lower()
    case_id = record.case_id.lower()
    combined = f"{source_relative_path} {case_id}"
    if "cookie" in combined:
        return "cookie-theft picture description; the patient should describe the pictured scene"
    if "market" in combined:
        return "market scene description with multiple simultaneous actions and characters"
    if "park" in combined:
        return "park scene description with multiple simultaneous actions and characters"
    if "daddy" in combined:
        return "home hazard scene description with multiple simultaneous actions and characters"
    if "fluency" in combined:
        return "verbal fluency task with the patient generating category items"
    if "recall" in combined:
        return "recall task focused on remembered information"
    if "sentence" in combined:
        return "structured sentence-level language task"
    return None


def _is_cjk_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


class QwenOmniAudioVerifier(BaseProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.verification_settings = get_verification_settings()
        super().__init__(
            settings,
            name="qwen_omni",
            description="Verification-only Qwen Omni native audio reviewer.",
            model_name=settings.qwen_omni_model,
            capabilities=ProviderCapabilities(
                native_video=False,
                native_audio_in_video=False,
                structured_output=True,
                requires_preprocessing=True,
            ),
        )
        self.api_key = settings.qwen_omni_api_key
        self.base_url = settings.qwen_omni_base_url.rstrip("/")
        self.asr_verifier = QwenAudioOnlyVerifier(self.verification_settings)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def analyze(
        self,
        provider_input: dict[str, Any],
        context: ProviderContext,
    ) -> ProviderRawResult:
        raise NotImplementedError("Verification calls classify_audio directly")

    async def classify_audio(
        self,
        *,
        record: VerificationRecord,
        audio_path: str,
        mime_type: str,
    ) -> tuple[dict[str, Any], str | None, str | None, str | None]:
        if not self.api_key:
            raise ProviderError(
                "provider_unavailable",
                "qwen_omni is not configured. Set QWEN_API_KEY or DASHSCOPE_API_KEY.",
            )

        asr_transcript, asr_request_id = await self.asr_verifier.transcribe_audio(
            audio_path=audio_path,
            mime_type=mime_type,
            language=record.language,
        )
        prompt = build_qwen_omni_audio_prompt_with_asr_assist(
            record,
            asr_transcript=asr_transcript,
        )
        response = await self._request_json(
            method="POST",
            url=f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompt.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.user_prompt,
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": self._encode_audio_data_url(Path(audio_path), mime_type),
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.1,
                "stream": False,
            },
            timeout=300.0,
        )
        text = self._extract_chat_text(response)
        payload = self._parse_verification_text_response(text)
        return payload, response.get("id"), asr_transcript, asr_request_id

    def _parse_verification_text_response(self, text: str) -> dict[str, Any]:
        try:
            raw = parse_json_object(text)
        except ValueError as exc:
            raise ProviderError(
                "invalid_provider_output",
                f"{self.name} response did not contain a valid JSON object",
                debug_details=self._build_text_debug_details(text),
            ) from exc

        candidate = self._extract_assessment_candidate(raw)
        normalized = self._coerce_verification_payload(candidate)
        try:
            payload = ProviderAssessmentPayload.model_validate(normalized)
        except Exception as exc:  # noqa: BLE001
            debug_details = self._build_text_debug_details(text)
            debug_details["parsed_json_preview"] = json.dumps(normalized, ensure_ascii=False)[:12000]
            raise ProviderError(
                "invalid_provider_output",
                f"{self.name} returned an invalid assessment payload",
                debug_details=debug_details,
            ) from exc

        result = payload.model_dump(mode="json")
        patient_only_transcript = self._coerce_transcript_text(
            self._find_nested_value(
                raw,
                (
                    "patient_only_transcript",
                    "patient_transcript",
                    "patient_only_summary",
                    "patient_summary",
                    "patient_reconstructed_transcript",
                    "patient_reconstruction",
                ),
            )
        )
        if patient_only_transcript:
            result["patient_only_transcript"] = patient_only_transcript

        speaker_turn_summary = self._coerce_turn_summary(
            self._find_nested_value(
                raw,
                (
                    "speaker_turn_summary",
                    "speaker_turns",
                    "dialogue_turn_summary",
                    "turn_summary",
                ),
            )
        )
        if speaker_turn_summary:
            result["speaker_turn_summary"] = speaker_turn_summary

        cue_summary = self._coerce_string_list(
            self._find_nested_value(
                raw,
                (
                    "patient_cue_summary",
                    "patient_cues",
                    "cue_summary",
                    "patient_language_cues",
                ),
            )
        )
        if cue_summary:
            result["patient_cue_summary"] = cue_summary

        return result

    def _extract_assessment_candidate(self, raw: dict[str, Any]) -> dict[str, Any]:
        best = raw
        best_score = self._assessment_key_score(raw)

        for candidate in self._iter_nested_dicts(raw):
            score = self._assessment_key_score(candidate)
            if score > best_score:
                best = candidate
                best_score = score

        if best is raw:
            return dict(raw)

        merged = dict(best)
        for key in (
            "patient_only_transcript",
            "patient_transcript",
            "patient_only_summary",
            "patient_summary",
            "patient_reconstructed_transcript",
            "patient_reconstruction",
            "speaker_turn_summary",
            "speaker_turns",
            "dialogue_turn_summary",
            "turn_summary",
            "patient_cue_summary",
            "patient_cues",
            "cue_summary",
            "patient_language_cues",
        ):
            if key in raw and key not in merged:
                merged[key] = raw[key]
        return merged

    def _assessment_key_score(self, payload: dict[str, Any]) -> int:
        score = 0
        for key in (
            "risk_label",
            "risk_score",
            "risk_tier",
            "screening_classification",
            "session_usability",
            "quality_flags",
        ):
            if key in payload:
                score += 3
        for key in (
            "visual_findings",
            "body_findings",
            "voice_findings",
            "content_findings",
            "speaker_structure",
            "target_patient_presence",
            "target_patient_basis",
            "evidence_for_risk",
            "evidence_against_risk",
            "context_notes",
        ):
            if key in payload:
                score += 1
        return score

    def _iter_nested_dicts(self, value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        def visit(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                found.append(node)
                for child in node.values():
                    visit(child, depth + 1)
            elif isinstance(node, list):
                for child in node:
                    visit(child, depth + 1)

        visit(value, 0)
        return found

    def _coerce_verification_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_assessment_payload_dict(dict(raw))

        for field_name, label_prefix in (
            ("visual_findings", "visual_finding"),
            ("body_findings", "body_finding"),
            ("voice_findings", "voice_finding"),
            ("content_findings", "content_finding"),
        ):
            normalized[field_name] = self._coerce_findings(
                self._find_nested_value(raw, (field_name,)),
                label_prefix,
            )

        string_list_fields = (
            "detected_languages",
            "evidence_for_risk",
            "evidence_against_risk",
            "alternative_explanations",
            "risk_factor_findings",
            "subjective_assessment_findings",
            "emotional_assessment_findings",
            "risk_control_suggestions",
            "context_notes",
        )
        for field_name in string_list_fields:
            normalized[field_name] = self._coerce_string_list(self._find_nested_value(raw, (field_name,)))

        quality_flags = self._coerce_string_list(self._find_nested_value(raw, ("quality_flags",)))
        normalized["quality_flags"] = normalize_quality_flags(quality_flags)

        speaker_structure = self._coerce_speaker_structure(
            self._find_nested_value(raw, ("speaker_structure",))
        )
        if speaker_structure is not None:
            normalized["speaker_structure"] = speaker_structure

        target_patient_presence = self._coerce_target_patient_presence(
            self._find_nested_value(raw, ("target_patient_presence",))
        )
        if target_patient_presence is not None:
            normalized["target_patient_presence"] = target_patient_presence

        target_patient_basis = self._coerce_target_patient_basis(
            self._find_nested_value(raw, ("target_patient_basis",))
        )
        if target_patient_basis is not None:
            normalized["target_patient_basis"] = target_patient_basis

        session_usability = self._coerce_session_usability(
            self._find_nested_value(raw, ("session_usability",))
        )
        if session_usability is not None:
            normalized["session_usability"] = session_usability

        risk_label = self._normalize_risk_label(
            self._find_nested_value(
                raw,
                ("risk_label", "predicted_label", "final_label", "label", "prediction"),
            )
        )
        if risk_label is not None:
            normalized["risk_label"] = risk_label

        screening_classification = self._normalize_screening_classification(
            self._find_nested_value(
                raw,
                (
                    "screening_classification",
                    "classification",
                    "screening_result",
                    "clinical_classification",
                ),
            )
        )
        if screening_classification is not None:
            normalized["screening_classification"] = screening_classification

        risk_tier = self._normalize_risk_tier(
            self._find_nested_value(raw, ("risk_tier", "tier", "severity"))
        )
        if risk_tier is not None:
            normalized["risk_tier"] = risk_tier

        for field_name, aliases in (
            ("risk_score", ("risk_score", "overall_risk_score", "final_risk_score")),
            ("reviewer_confidence", ("reviewer_confidence", "overall_confidence", "final_confidence")),
            ("language_confidence", ("language_confidence",)),
        ):
            value = self._coerce_probability(self._find_nested_value(raw, aliases))
            if value is not None:
                normalized[field_name] = value

        for field_name, aliases in (
            ("screening_summary", ("screening_summary", "assessment_summary", "final_summary")),
            ("visit_recommendation", ("visit_recommendation",)),
            ("future_risk_trend_summary", ("future_risk_trend_summary", "trend_summary")),
            ("disclaimer", ("disclaimer",)),
        ):
            text_value = self._coerce_text(self._find_nested_value(raw, aliases))
            if text_value is not None:
                normalized[field_name] = text_value

        return normalized

    def _find_nested_value(self, value: Any, keys: tuple[str, ...], *, max_depth: int = 4) -> Any:
        def visit(node: Any, depth: int) -> Any:
            if depth > max_depth:
                return None
            if isinstance(node, dict):
                for key in keys:
                    if key in node and not self._is_empty(node[key]):
                        return node[key]
                for child in node.values():
                    match = visit(child, depth + 1)
                    if match is not None:
                        return match
            elif isinstance(node, list):
                for child in node:
                    match = visit(child, depth + 1)
                    if match is not None:
                        return match
            return None

        if isinstance(value, dict):
            for key in keys:
                if key in value and not self._is_empty(value[key]):
                    return value[key]
        return visit(value, 0)

    def _coerce_findings(self, value: Any, label_prefix: str) -> list[dict[str, Any]]:
        if self._is_empty(value):
            return []

        items = value
        if isinstance(value, dict):
            if any(key in value for key in ("items", "findings", "results")):
                items = value.get("items") or value.get("findings") or value.get("results")
            else:
                items = [value]

        if not isinstance(items, list):
            items = [items]

        findings: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            finding = self._coerce_finding(item, label_prefix, index)
            if finding is not None:
                findings.append(finding)
        return findings

    def _coerce_finding(
        self,
        value: Any,
        label_prefix: str,
        index: int,
    ) -> dict[str, Any] | None:
        if self._is_empty(value):
            return None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            label, summary = self._split_labeled_text(text)
            return {
                "label": label or f"{label_prefix}_{index}",
                "summary": summary or text,
            }

        if not isinstance(value, dict):
            text = self._coerce_text(value)
            if text is None:
                return None
            return {
                "label": f"{label_prefix}_{index}",
                "summary": text,
            }

        label = self._coerce_text(
            self._first_present(
                value,
                ("label", "name", "title", "finding", "type", "category"),
            )
        )
        summary = self._coerce_text(
            self._first_present(
                value,
                ("summary", "description", "text", "content", "observation"),
            )
        )
        evidence = self._coerce_text(
            self._first_present(
                value,
                ("evidence", "rationale", "quote", "supporting_evidence"),
            )
        )
        confidence = self._coerce_probability(
            self._first_present(value, ("confidence", "score", "probability"))
        )

        if summary is None:
            summary_candidates = self._coerce_string_list(value)
            if summary_candidates:
                summary = "; ".join(summary_candidates)

        if label is None and summary is not None:
            label = f"{label_prefix}_{index}"
        if summary is None and label is not None:
            summary = label
        if label is None or summary is None:
            return None

        finding: dict[str, Any] = {"label": label, "summary": summary}
        if evidence is not None:
            finding["evidence"] = evidence
        if confidence is not None:
            finding["confidence"] = confidence
        return finding

    def _coerce_string_list(self, value: Any) -> list[str]:
        if self._is_empty(value):
            return []

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            parsed_json = self._try_parse_json_fragment(stripped)
            if parsed_json is not None and parsed_json is not value:
                return self._coerce_string_list(parsed_json)

            chunks = [
                part.strip(" -*\t")
                for line in stripped.splitlines()
                for part in line.split(";")
            ]
            return [chunk for chunk in chunks if chunk]

        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                items.extend(self._coerce_string_list(item))
            return items

        if isinstance(value, dict):
            for key in ("items", "values", "points", "summary", "text", "content"):
                if key in value and not self._is_empty(value[key]):
                    return self._coerce_string_list(value[key])

            items: list[str] = []
            for key, item in value.items():
                item_text = self._coerce_text(item)
                if item_text is not None:
                    if str(key).isdigit():
                        items.append(item_text)
                    else:
                        items.append(f"{key}: {item_text}")
            return items

        text = self._coerce_text(value)
        return [text] if text is not None else []

    def _coerce_turn_summary(self, value: Any) -> list[str]:
        if self._is_empty(value):
            return []

        if isinstance(value, list):
            turns: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    speaker = self._coerce_text(
                        self._first_present(item, ("speaker", "role", "who"))
                    )
                    utterance = self._coerce_text(
                        self._first_present(item, ("text", "utterance", "summary", "content"))
                    )
                    if speaker and utterance:
                        turns.append(f"{speaker}: {utterance}")
                        continue
                turns.extend(self._coerce_string_list(item))
            return turns

        if isinstance(value, dict):
            if any(key in value for key in ("speaker", "role", "who")):
                return self._coerce_turn_summary([value])
            if any(key in value for key in ("items", "turns", "dialogue")):
                return self._coerce_turn_summary(
                    value.get("items") or value.get("turns") or value.get("dialogue")
                )

        return self._coerce_string_list(value)

    def _coerce_transcript_text(self, value: Any) -> str | None:
        if self._is_empty(value):
            return None
        if isinstance(value, dict):
            for key in ("transcript", "summary", "text", "content"):
                if key in value and not self._is_empty(value[key]):
                    return self._coerce_transcript_text(value[key])
        parts = self._coerce_string_list(value)
        if not parts:
            return None
        return " ".join(parts)

    def _coerce_probability(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            normalized = stripped.lower().replace("-", "_").replace(" ", "_")
            textual_aliases = {
                "very_low": 0.1,
                "low": 0.25,
                "low_confidence": 0.25,
                "medium": 0.5,
                "moderate": 0.5,
                "medium_confidence": 0.5,
                "moderate_confidence": 0.5,
                "high": 0.75,
                "high_confidence": 0.75,
                "very_high": 0.9,
                "very_high_confidence": 0.9,
            }
            if normalized in textual_aliases:
                return textual_aliases[normalized]
            is_percent = stripped.endswith("%")
            stripped = stripped.rstrip("%").strip()
            try:
                numeric = float(stripped)
            except ValueError:
                return None
            if is_percent or numeric > 1.0:
                numeric /= 100.0
        else:
            return None
        return max(0.0, min(1.0, numeric))

    def _coerce_speaker_structure(self, value: Any) -> str | None:
        text = self._coerce_text(value)
        if text is None:
            return None
        cleaned = text.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "single": "single_speaker",
            "single_speaker": "single_speaker",
            "one_speaker": "single_speaker",
            "multi": "multi_speaker",
            "multiple": "multi_speaker",
            "multi_speaker": "multi_speaker",
            "multiple_speakers": "multi_speaker",
            "dialogue": "multi_speaker",
            "unclear": "unclear",
            "unknown": "unclear",
        }
        return aliases.get(cleaned)

    def _coerce_target_patient_presence(self, value: Any) -> str | None:
        text = self._coerce_text(value)
        if text is None:
            return None
        cleaned = text.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "clear": "clear",
            "present": "clear",
            "probable": "probable",
            "likely": "probable",
            "uncertain": "uncertain",
            "unclear": "uncertain",
            "unknown": "uncertain",
            "absent": "absent",
            "missing": "absent",
        }
        return aliases.get(cleaned)

    def _coerce_target_patient_basis(self, value: Any) -> str | None:
        text = self._coerce_text(value)
        if text is None:
            return None
        cleaned = text.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "visual_focus": "visual_focus",
            "speaking_time": "speaking_time",
            "interaction_role": "interaction_role",
            "role": "interaction_role",
            "speaker_role": "interaction_role",
            "externally_provided": "externally_provided",
            "external": "externally_provided",
            "metadata": "externally_provided",
            "provided_metadata": "externally_provided",
            "unclear": "unclear",
            "unknown": "unclear",
        }
        return aliases.get(cleaned)

    def _coerce_session_usability(self, value: Any) -> str | None:
        text = self._coerce_text(value)
        if text is None:
            return None
        cleaned = text.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "usable": "usable",
            "usable_with_caveats": "usable_with_caveats",
            "usable_but_limited": "usable_with_caveats",
            "usable_but_noisy": "usable_with_caveats",
            "partially_usable": "usable_with_caveats",
            "limited_but_usable": "usable_with_caveats",
            "unusable": "unusable",
            "not_usable": "unusable",
        }
        return aliases.get(cleaned)

    def _first_present(self, value: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in value and not self._is_empty(value[key]):
                return value[key]
        return None

    def _split_labeled_text(self, value: str) -> tuple[str | None, str | None]:
        if ":" not in value:
            return None, value
        label, summary = value.split(":", 1)
        clean_label = label.strip()
        clean_summary = summary.strip()
        if len(clean_label) <= 60 and clean_summary:
            return clean_label, clean_summary
        return None, value

    def _coerce_text(self, value: Any) -> str | None:
        if self._is_empty(value):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return None

    def _try_parse_json_fragment(self, value: str) -> Any | None:
        if not value.startswith(("{", "[")):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def _is_empty(self, value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    def _encode_audio_data_url(self, path: Path, mime_type: str) -> str:
        raw = path.read_bytes()
        max_inline_bytes = self.settings.max_inline_video_mb * 1024 * 1024
        if len(raw) > max_inline_bytes:
            raise ProviderError(
                "unsupported_media",
                f"qwen_omni inline audio exceeds {self.settings.max_inline_video_mb}MB limit",
            )
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
