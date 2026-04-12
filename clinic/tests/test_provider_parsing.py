"""Unit tests for provider response normalization before schema validation."""

from __future__ import annotations

from pathlib import Path

from clinic.configs.settings import Settings
from backend.src.app.models import ProviderCapabilities, ProviderContext, ProviderRawResult
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
