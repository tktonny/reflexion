"""Unit tests for qwen_omni media guards."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError
from backend.src.app.models import PreparedMedia, ProviderContext
from clinic.intelligence.providers.qwen_omni import QwenOmniProvider


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
        qwen_omni_api_key="test-key",
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


def test_qwen_prepare_rejects_browser_webm_upload(tmp_path: Path) -> None:
    provider = QwenOmniProvider(make_settings(tmp_path))
    context = ProviderContext(
        assessment_id="a1",
        patient_id="patient-001",
        language="en",
        media=PreparedMedia(
            original_path="/tmp/input.webm",
            standardized_path="/tmp/standardized.webm",
            mime_type="video/webm",
            size_bytes=1024,
        ),
    )

    with pytest.raises(ProviderError, match="Install ffmpeg on the server"):
        asyncio.run(provider.prepare(context))


def test_qwen_prepare_rejects_inline_video_over_provider_limit(tmp_path: Path) -> None:
    provider = QwenOmniProvider(make_settings(tmp_path))
    context = ProviderContext(
        assessment_id="a2",
        patient_id="patient-001",
        language="en",
        media=PreparedMedia(
            original_path="/tmp/input.mp4",
            standardized_path="/tmp/standardized.mp4",
            mime_type="video/mp4",
            size_bytes=(10 * 1024 * 1024) + 1,
        ),
    )

    with pytest.raises(ProviderError, match="provider limit of 10MB"):
        asyncio.run(provider.prepare(context))
