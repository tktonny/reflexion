from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from verification.config import VerificationSettings
from verification.media import prepare_omni_media_artifact
from verification.models import VerificationRecord
from verification.qwen_omni_audio import (
    build_qwen_omni_audio_context_note,
    build_qwen_omni_audio_few_shot_examples,
    build_qwen_omni_audio_output_contract,
    build_qwen_omni_audio_prompt,
    build_qwen_omni_asr_assist_note,
    QwenOmniAudioVerifier,
)
from verification.run_qwen_omni_verification import select_records


def make_settings(tmp_path: Path) -> VerificationSettings:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    prepared_dir = tmp_path / "prepared"
    data_dir.mkdir()
    results_dir.mkdir()
    prepared_dir.mkdir()
    return VerificationSettings(
        project_root=tmp_path,
        data_dir=data_dir,
        results_dir=results_dir,
        prepared_dir=prepared_dir,
        qwen_api_key="qwen-key",
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        coding_plan_api_key="coding-key",
        coding_plan_base_url="https://coding.dashscope.aliyuncs.com/v1",
        qwen_text_model="qwen3.5-plus",
        qwen_asr_model="qwen3-asr-flash",
        request_timeout_seconds=60.0,
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        talkbank_auth_base_url="https://sla2.talkbank.org",
        talkbank_media_base_url="https://media.talkbank.org",
        talkbank_email=None,
        talkbank_password=None,
    )


def make_records() -> list[VerificationRecord]:
    return [
        VerificationRecord(
            case_id="hc-first",
            dataset="demo",
            split="test",
            label="HC",
            media_path="/tmp/hc-first.mp3",
            media_type="audio",
            language="en",
        ),
        VerificationRecord(
            case_id="risk-first",
            dataset="demo",
            split="test",
            label="cognitive_risk",
            media_path="/tmp/risk-first.mp3",
            media_type="audio",
            language="en",
        ),
        VerificationRecord(
            case_id="hc-second",
            dataset="demo",
            split="test",
            label="HC",
            media_path="/tmp/hc-second.mp3",
            media_type="audio",
            language="en",
        ),
    ]


def test_select_records_first_per_label_keeps_first_manifest_match() -> None:
    selected = select_records(
        make_records(),
        labels=["HC", "cognitive_risk"],
        first_per_label=True,
    )

    assert [record.case_id for record in selected] == ["hc-first", "risk-first"]


def test_select_records_case_ids_preserve_requested_order() -> None:
    selected = select_records(
        make_records(),
        case_ids=["risk-first", "hc-second"],
    )

    assert [record.case_id for record in selected] == ["risk-first", "hc-second"]


def test_qwen_omni_audio_context_note_mentions_missing_video_and_dialogue_roles() -> None:
    record = VerificationRecord(
        case_id="english__pitt__control__cookie__054_0",
        dataset="talkbank_english_pitt",
        split="full",
        label="HC",
        media_path="/tmp/054-0.mp3",
        media_type="audio",
        language="en",
        metadata={"source_relative_path": "Control/cookie/054-0.mp3"},
    )

    note = build_qwen_omni_audio_context_note(record)

    assert "video portion is missing" in note
    assert "examiner or tester" in note
    assert "evaluate only the patient" in note
    assert "internally transcribe" in note
    assert "screening signal comes from the patient's own replies" in note
    assert "Do not default to HC just because the patient mentions a few correct keywords" in note
    assert "High-signal patient-audio risk features to look for" in note
    assert "word-finding pauses, circumlocution" in note
    assert "Decision guidance for this verification setting" in note
    assert "cookie-theft picture description" in note


def test_qwen_omni_audio_prompt_preserves_original_audio_only_protocol() -> None:
    record = VerificationRecord(
        case_id="english__pitt__control__cookie__054_0",
        dataset="talkbank_english_pitt",
        split="full",
        label="HC",
        media_path="/tmp/054-0.mp3",
        media_type="audio",
        language="en",
        metadata={"source_relative_path": "Control/cookie/054-0.mp3"},
    )

    prompt = build_qwen_omni_audio_prompt(record)

    assert "This is a final audio-only fallback review." in prompt.user_prompt
    assert "Return exactly one JSON object matching the schema." in prompt.user_prompt
    assert "Do not infer facial expression, gaze, gesture, or visible engagement from speech alone." in prompt.user_prompt
    assert "The video portion is missing for this patient" in prompt.user_prompt
    assert "identify which speech belongs to the examiner and which belongs to the patient" in prompt.user_prompt
    assert "Do not default to HC just because the patient mentions a few correct keywords" in prompt.user_prompt
    assert "cognitive_risk does not require dementia-level certainty" in prompt.user_prompt
    assert '"patient_only_transcript"' in prompt.user_prompt
    assert '"speaker_turn_summary"' in prompt.user_prompt
    assert '"patient_cue_summary"' in prompt.user_prompt
    assert "Few-shot contrast examples" in prompt.user_prompt
    assert "Auxiliary ASR evidence:" in prompt.user_prompt


def test_qwen_omni_audio_prompt_contract_and_few_shot_are_explicit() -> None:
    record = VerificationRecord(
        case_id="english__pitt__control__cookie__054_0",
        dataset="talkbank_english_pitt",
        split="full",
        label="HC",
        media_path="/tmp/054-0.mp3",
        media_type="audio",
        language="en",
        metadata={"source_relative_path": "Control/cookie/054-0.mp3"},
    )

    contract = build_qwen_omni_audio_output_contract()
    examples = build_qwen_omni_audio_few_shot_examples(record)

    assert "Return exactly one JSON object." in contract
    assert "patient-only reconstructed transcript" in contract
    assert "speaker_turn_summary" in contract
    assert "patient_cue_summary" in contract
    assert "Example A (more likely HC" in examples
    assert "Example B (more likely cognitive_risk" in examples
    assert "correct keywords like cookie, sink, mother, or stool" in examples


def test_qwen_omni_asr_assist_note_mentions_turns_and_pauses() -> None:
    note = build_qwen_omni_asr_assist_note(
        "Examiner: Tell me what is happening... Patient: Uh... the boy is taking cookies."
    )

    assert "Auxiliary ASR evidence:" in note
    assert "mix examiner and patient turns together" in note
    assert "hesitations, fillers, restarts, repetitions, and pause markers" in note
    assert "Examiner: Tell me what is happening..." in note


def test_qwen_omni_audio_parser_preserves_verification_extras() -> None:
    verifier = QwenOmniAudioVerifier()
    payload = verifier._parse_verification_text_response(
        """
        {
          "patient_only_transcript": "The boy is taking cookies and the sink is overflowing.",
          "speaker_turn_summary": ["Examiner prompts the task.", "Patient gives a short scene description with hesitation."],
          "patient_cue_summary": ["coherent scene description", "specific content words"],
          "voice_findings": [],
          "content_findings": [],
          "risk_label": "HC",
          "risk_tier": "low",
          "screening_classification": "healthy",
          "risk_score": 0.2,
          "reviewer_confidence": 0.7,
          "quality_flags": [],
          "session_usability": "usable_with_caveats",
          "context_notes": []
        }
        """
    )

    assert payload["patient_only_transcript"] == "The boy is taking cookies and the sink is overflowing."
    assert payload["speaker_turn_summary"] == [
        "Examiner prompts the task.",
        "Patient gives a short scene description with hesitation.",
    ]
    assert payload["patient_cue_summary"] == ["coherent scene description", "specific content words"]
    assert payload["risk_label"] == "HC"


def test_qwen_omni_audio_parser_salvages_wrapped_and_shaky_payloads() -> None:
    verifier = QwenOmniAudioVerifier()
    payload = verifier._parse_verification_text_response(
        """
        {
          "output": {
            "assessment": {
              "voice_findings": {
                "summary": "Frequent pauses and vague wording.",
                "confidence": "68%"
              },
              "content_findings": [
                "Fragmented scene description with missing links."
              ],
              "risk_label": "cognitive decline risk",
              "risk_tier": "moderate",
              "screening_classification": "monitoring",
              "risk_score": "71%",
              "reviewer_confidence": "82%",
              "quality_flags": "multiple people\\npoor audio",
              "context_notes": {
                "audio": "Noisy clip with overlap.",
                "video": "Missing by design."
              },
              "session_usability": "usable but limited"
            },
            "patient_only_transcript": [
              "uh...",
              "the boy...",
              "cookies..."
            ],
            "speaker_turn_summary": [
              {
                "speaker": "Examiner",
                "text": "Tell me what's happening in the picture."
              },
              {
                "speaker": "Patient",
                "utterance": "Uh... the boy... cookies..."
              }
            ],
            "patient_cue_summary": {
              "1": "word-finding pauses",
              "2": "fragmented organization"
            }
          }
        }
        """
    )

    assert payload["risk_label"] == "cognitive_risk"
    assert payload["risk_tier"] == "medium"
    assert payload["screening_classification"] == "needs_observation"
    assert payload["risk_score"] == 0.71
    assert payload["reviewer_confidence"] == 0.82
    assert payload["session_usability"] == "usable_with_caveats"
    assert payload["quality_flags"] == ["multiple_people", "poor_audio"]
    assert payload["patient_only_transcript"] == "uh... the boy... cookies..."
    assert payload["speaker_turn_summary"] == [
        "Examiner: Tell me what's happening in the picture.",
        "Patient: Uh... the boy... cookies...",
    ]
    assert payload["patient_cue_summary"] == ["word-finding pauses", "fragmented organization"]
    assert payload["voice_findings"][0]["label"] == "voice_finding_1"
    assert payload["voice_findings"][0]["summary"] == "Frequent pauses and vague wording."
    assert payload["voice_findings"][0]["confidence"] == 0.68
    assert payload["content_findings"][0]["label"] == "content_finding_1"
    assert payload["content_findings"][0]["summary"] == "Fragmented scene description with missing links."
    assert payload["context_notes"] == ["audio: Noisy clip with overlap.", "video: Missing by design."]


def test_prepare_omni_media_artifact_wraps_audio_as_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake-audio")
    settings = make_settings(tmp_path)

    record = VerificationRecord(
        case_id="sample-case",
        dataset="demo",
        split="test",
        label="HC",
        media_path=str(audio_path),
        media_type="audio",
        language="en",
    )

    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool) -> SimpleNamespace:
        target = Path(command[-1])
        if command[0] == "ffmpeg":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-mp4")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")

    monkeypatch.setattr("verification.media.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("verification.media.subprocess.run", fake_run)

    prepared = prepare_omni_media_artifact(record, settings)

    assert prepared.standardized_path.endswith("omni_input.mp4")
    assert prepared.mime_type == "video/mp4"
    assert prepared.duration_seconds == 12.5
    assert Path(prepared.standardized_path).exists()


def test_prepare_omni_media_artifact_requires_ffmpeg_for_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake-audio")
    settings = make_settings(tmp_path)
    record = VerificationRecord(
        case_id="sample-case",
        dataset="demo",
        split="test",
        label="HC",
        media_path=str(audio_path),
        media_type="audio",
        language="en",
    )

    monkeypatch.setattr("verification.media.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        prepare_omni_media_artifact(record, settings)
