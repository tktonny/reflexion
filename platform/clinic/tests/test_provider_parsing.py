"""Unit tests for provider response normalization before schema validation."""

from __future__ import annotations

from pathlib import Path

from clinic.configs.settings import Settings
from backend.src.app.models import PreparedMedia, ProviderCapabilities, ProviderContext, ProviderRawResult
from clinic.intelligence.providers.base import BaseProvider


class ParsingTestProvider(BaseProvider):
    """Minimal concrete provider used to test shared parsing helpers."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            name="qwen_omni",
            description="test",
            model_name="test-model",
            capabilities=ProviderCapabilities(
                native_video=True,
                native_audio_in_video=True,
                structured_output=True,
                requires_preprocessing=False,
            ),
        )

    def is_configured(self) -> bool:
        return True

    async def analyze(self, provider_input: dict[str, object], context: ProviderContext) -> ProviderRawResult:
        raise NotImplementedError


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="test",
        storage_dir=tmp_path,
        uploads_dir=tmp_path / "uploads",
        prepared_dir=tmp_path / "prepared",
        assessments_dir=tmp_path / "assessments",
        server_host="127.0.0.1",
        server_port=8000,
        server_reload=False,
        max_upload_mb=100,
        max_inline_video_mb=5,
        allow_mock_providers=False,
        default_provider="qwen_omni",
        fallback_order=("qwen_omni", "gemini", "fusion", "audio_only"),
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        qwen_omni_api_key=None,
        qwen_omni_base_url="https://example.com/qwen",
        qwen_omni_model="qwen3.5-omni-plus",
        gemini_api_key=None,
        gemini_base_url="https://example.com/gemini",
        gemini_model="gemini-3.1-pro-preview",
        openai_api_key=None,
        openai_base_url="https://example.com/openai",
        openai_fusion_model="gpt-4.1",
        openai_text_model="gpt-4.1",
        openai_transcription_model="gpt-4o-transcribe",
    )


def test_parse_text_response_swaps_common_label_mixup(tmp_path: Path) -> None:
    provider = ParsingTestProvider(make_settings(tmp_path))
    raw_text = """
    {
      "session_usability": "usable",
      "quality_flags": [],
      "visual_findings": [],
      "body_findings": [],
      "voice_findings": [],
      "content_findings": [],
      "risk_label": "needs_observation",
      "risk_tier": "moderate",
      "screening_classification": "cognitive_risk",
      "risk_score": 0.45
    }
    """

    payload = provider._parse_text_response(raw_text)

    assert payload.risk_label == "cognitive_risk"
    assert payload.screening_classification == "needs_observation"
    assert payload.risk_tier == "medium"


def test_parse_text_response_infers_missing_risk_fields_from_classification(tmp_path: Path) -> None:
    provider = ParsingTestProvider(make_settings(tmp_path))
    raw_text = """
    {
      "session_usability": "usable_with_caveats",
      "quality_flags": ["limited_speaking_time"],
      "visual_findings": [],
      "body_findings": [],
      "voice_findings": [],
      "content_findings": [],
      "risk_tier": "medium",
      "screening_classification": "needs_observation"
    }
    """

    payload = provider._parse_text_response(raw_text)

    assert payload.risk_score == 0.5
    assert payload.risk_label == "cognitive_risk"
    assert payload.risk_tier == "medium"
    assert payload.screening_classification == "needs_observation"


def test_parse_text_response_normalizes_percent_risk_score_and_infers_label(tmp_path: Path) -> None:
    provider = ParsingTestProvider(make_settings(tmp_path))
    raw_text = """
    {
      "session_usability": "usable",
      "quality_flags": [],
      "visual_findings": [],
      "body_findings": [],
      "voice_findings": [],
      "content_findings": [],
      "risk_score": "20%",
      "screening_classification": "healthy"
    }
    """

    payload = provider._parse_text_response(raw_text)

    assert payload.risk_score == 0.2
    assert payload.risk_label == "HC"
    assert payload.screening_classification == "healthy"
    assert payload.risk_tier == "low"


def test_augment_payload_synthesizes_voice_findings_from_session_record(tmp_path: Path) -> None:
    provider = ParsingTestProvider(make_settings(tmp_path))
    payload = provider._parse_text_response(
        """
        {
          "session_usability": "usable",
          "quality_flags": [],
          "visual_findings": [],
          "body_findings": [],
          "voice_findings": [],
          "content_findings": [],
          "risk_score": 0.42,
          "risk_label": "cognitive_risk",
          "screening_classification": "needs_observation"
        }
        """
    )
    context = ProviderContext(
        assessment_id="a1",
        patient_id="patient-001",
        language="en",
        media=PreparedMedia(
            original_path="/tmp/input.mp4",
            standardized_path="/tmp/standardized.mp4",
            mime_type="video/mp4",
            size_bytes=1024,
        ),
        session_record={
            "qualityControl": {"flags": []},
            "derivedFeatures": {
                "speech": {
                    "speechSeconds": 24.0,
                    "utteranceCount": 5,
                    "averageTurnSeconds": 2.8,
                    "averageRms": 0.012,
                    "peakRms": 0.054,
                    "voicedChunkRatio": 0.41,
                    "transcriptTurns": [
                        {"role": "patient", "text": "Um, I had breakfast and then went outside."},
                        {"role": "patient", "text": "I am not sure what happened after that."},
                    ],
                },
                "interactionTiming": {
                    "turnDurationsSeconds": [1.8, 3.1, 2.7, 4.4, 2.0],
                },
            },
        },
    )

    augmented = provider.augment_payload(payload, context)

    assert augmented.voice_findings
    assert any(item.label == "transcript_derived_hesitation_pattern" for item in augmented.voice_findings)
    assert "Voice findings were supplemented from live-session transcript and speech-timing metadata." in augmented.context_notes


def test_augment_payload_does_not_invent_voice_findings_without_speech_evidence(tmp_path: Path) -> None:
    provider = ParsingTestProvider(make_settings(tmp_path))
    payload = provider._parse_text_response(
        """
        {
          "session_usability": "usable_with_caveats",
          "quality_flags": ["transcript_unavailable"],
          "visual_findings": [],
          "body_findings": [],
          "voice_findings": [],
          "content_findings": [],
          "risk_score": 0.5,
          "risk_label": "cognitive_risk",
          "screening_classification": "needs_observation"
        }
        """
    )
    context = ProviderContext(
        assessment_id="a2",
        patient_id="patient-001",
        language="en",
        media=PreparedMedia(
            original_path="/tmp/input.mp4",
            standardized_path="/tmp/standardized.mp4",
            mime_type="video/mp4",
            size_bytes=1024,
        ),
        session_record={
            "qualityControl": {"flags": ["transcript_unavailable"]},
            "derivedFeatures": {
                "speech": {
                    "speechSeconds": 2.0,
                    "utteranceCount": 0,
                    "transcriptTurns": [],
                }
            },
        },
    )

    augmented = provider.augment_payload(payload, context)

    assert augmented.voice_findings == []
