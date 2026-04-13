"""Unit tests for provider mesh routing and fallback behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError, RoutingError
from backend.src.app.models import (
    PreparedMedia,
    ProviderAssessmentPayload,
    ProviderCapabilities,
    ProviderContext,
    ProviderName,
    ProviderRawResult,
    ProviderStatus,
)
from backend.src.app.services.assessment_service import ProviderMeshRouter
from backend.src.app.services.media_preparer import MediaPreparer


class FakeProvider:
    """Minimal provider stub for routing tests."""

    def __init__(
        self,
        name: ProviderName,
        *,
        result: ProviderRawResult | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self.name = name
        self.model_name = f"{name}-model"
        self.result = result
        self.error = error
        self.capabilities = ProviderCapabilities(
            native_video=True,
            native_audio_in_video=True,
            structured_output=True,
            requires_preprocessing=False,
        )

    async def healthcheck(self, fallback_rank: int) -> ProviderStatus:
        return ProviderStatus(
            provider=self.name,
            available=True,
            configured=True,
            mock_mode=False,
            fallback_rank=fallback_rank,
            description=self.name,
            capabilities=self.capabilities,
        )

    async def prepare(self, context: ProviderContext) -> dict[str, object]:
        del context
        return {}

    async def analyze(
        self,
        provider_input: dict[str, object],
        context: ProviderContext,
    ) -> ProviderRawResult:
        del provider_input, context
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("FakeProvider requires either a result or an error.")
        return self.result

    def normalize(self, raw_result: ProviderRawResult) -> ProviderAssessmentPayload:
        return raw_result.payload


def make_settings(tmp_path: Path) -> Settings:
    """Build a minimal settings object for router unit tests."""

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


def make_result(**payload_overrides: object) -> ProviderRawResult:
    """Build a normalized provider result with sensible test defaults."""

    payload_fields: dict[str, object] = {
        "visual_findings": [],
        "body_findings": [],
        "voice_findings": [],
        "content_findings": [],
        "risk_score": 0.5,
        "risk_label": "cognitive_risk",
        "quality_flags": [],
        "session_usability": "usable_with_caveats",
    }
    payload_fields.update(payload_overrides)
    payload = ProviderAssessmentPayload(**payload_fields)
    return ProviderRawResult(payload=payload)


def make_router(
    settings: Settings,
    providers: dict[ProviderName, FakeProvider],
    media_preparer: MediaPreparer | None = None,
) -> ProviderMeshRouter:
    """Build a router with injected fake providers."""

    router = ProviderMeshRouter(settings, media_preparer=media_preparer)
    router.providers = providers
    return router


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def provider_context() -> ProviderContext:
    return ProviderContext(
        assessment_id="a1",
        patient_id="p1",
        language="en",
        media=PreparedMedia(
            original_path="/tmp/input.mp4",
            standardized_path="/tmp/standardized.mp4",
            mime_type="video/mp4",
            size_bytes=10,
        ),
    )


def test_router_falls_back_after_provider_error(
    settings: Settings,
    provider_context: ProviderContext,
) -> None:
    router = make_router(
        settings,
        {
            "qwen_omni": FakeProvider(
                "qwen_omni",
                error=ProviderError("timeout", "boom"),
            ),
            "gemini": FakeProvider(
                "gemini",
                result=make_result(
                    visual_findings=[{"label": "v", "summary": "ok"}],
                    risk_score=0.61,
                    risk_label="cognitive_risk",
                    session_usability="usable",
                ),
            ),
            "fusion": FakeProvider("fusion", result=make_result()),
            "audio_only": FakeProvider("audio_only", result=make_result()),
        },
    )

    assessment = asyncio.run(router.analyze(provider_context))

    assert assessment.provider_meta.final_provider == "gemini"
    assert assessment.provider_trace[0].failure_reason == "timeout"
    assert assessment.fallback_message == "Processed by gemini after qwen_omni timeout."


def test_router_accepts_first_usable_result_with_caveats(
    settings: Settings,
    provider_context: ProviderContext,
) -> None:
    router = make_router(
        settings,
        {
            "qwen_omni": FakeProvider(
                "qwen_omni",
                result=make_result(
                    risk_score=0.33,
                    risk_label="HC",
                    quality_flags=["poor_audio"],
                    session_usability="usable_with_caveats",
                ),
            ),
            "gemini": FakeProvider("gemini", result=make_result()),
            "fusion": FakeProvider("fusion", result=make_result()),
            "audio_only": FakeProvider("audio_only", result=make_result()),
        },
    )

    assessment = asyncio.run(router.analyze(provider_context))

    assert assessment.provider_meta.final_provider == "qwen_omni"
    assert len(assessment.provider_trace) == 1
    assert assessment.fallback_message is None


def test_router_falls_back_after_unusable_payload(
    settings: Settings,
    provider_context: ProviderContext,
) -> None:
    router = make_router(
        settings,
        {
            "qwen_omni": FakeProvider(
                "qwen_omni",
                result=make_result(
                    risk_score=None,
                    risk_label=None,
                    session_usability="unusable",
                ),
            ),
            "gemini": FakeProvider(
                "gemini",
                result=make_result(
                    risk_score=0.2,
                    risk_label="HC",
                    session_usability="usable",
                ),
            ),
            "fusion": FakeProvider("fusion", result=make_result()),
            "audio_only": FakeProvider("audio_only", result=make_result()),
        },
    )

    assessment = asyncio.run(router.analyze(provider_context))

    assert assessment.provider_meta.final_provider == "gemini"
    assert assessment.provider_trace[0].failure_reason == "unusable_result"
    assert assessment.fallback_message == "Processed by gemini after qwen_omni unusable_result."


def test_router_preserves_unusable_payload_debug_details(
    settings: Settings,
    provider_context: ProviderContext,
) -> None:
    router = make_router(
        settings,
        {
            "qwen_omni": FakeProvider(
                "qwen_omni",
                result=ProviderRawResult(
                    payload=ProviderAssessmentPayload(
                        visual_findings=[],
                        body_findings=[],
                        voice_findings=[],
                        content_findings=[],
                        risk_score=None,
                        risk_label=None,
                        session_usability="usable_with_caveats",
                        quality_flags=["limited_speaking_time"],
                    ),
                    request_id="req-qwen-1",
                    raw_status="ok",
                    debug_details={
                        "provider": "qwen_omni",
                        "raw_text_preview": "{\"session_usability\":\"usable_with_caveats\"}",
                    },
                ),
            ),
            "gemini": FakeProvider(
                "gemini",
                error=ProviderError("provider_unavailable", "gemini is not configured"),
            ),
            "fusion": FakeProvider(
                "fusion",
                error=ProviderError("provider_unavailable", "fusion provider is not configured"),
            ),
            "audio_only": FakeProvider(
                "audio_only",
                error=ProviderError("provider_unavailable", "audio_only provider is not configured"),
            ),
        },
    )

    with pytest.raises(RoutingError) as exc_info:
        asyncio.run(router.analyze(provider_context))

    trace = exc_info.value.provider_trace
    assert trace[0]["failure_reason"] == "unusable_result"
    assert trace[0]["request_id"] == "req-qwen-1"
    assert trace[0]["debug_details"]["missing_required_fields"] == ["risk_score", "risk_label"]
    assert trace[0]["debug_details"]["raw_text_preview"].startswith("{\"session_usability\"")


def test_router_strict_provider_stops_on_failure(
    settings: Settings,
    provider_context: ProviderContext,
) -> None:
    router = make_router(
        settings,
        {
            "qwen_omni": FakeProvider(
                "qwen_omni",
                error=ProviderError("timeout", "boom"),
            ),
            "gemini": FakeProvider("gemini", result=make_result()),
            "fusion": FakeProvider("fusion", result=make_result()),
            "audio_only": FakeProvider("audio_only", result=make_result()),
        },
    )
    strict_context = provider_context.model_copy(
        update={"preferred_provider": "qwen_omni", "strict_provider": True}
    )

    with pytest.raises(RoutingError):
        asyncio.run(router.analyze(strict_context))


class FakeMediaPreparer:
    """Track which provider requests fallback artifacts during routing."""

    def __init__(self) -> None:
        self.calls: list[ProviderName] = []

    def prepare_for_provider(
        self,
        provider_name: ProviderName,
        media: PreparedMedia,
    ) -> PreparedMedia:
        self.calls.append(provider_name)
        if provider_name == "fusion":
            return media.model_copy(
                update={
                    "extracted_audio_path": "/tmp/audio.wav",
                    "frame_paths": ["/tmp/frame-001.jpg"],
                }
            )
        if provider_name == "audio_only":
            return media.model_copy(update={"extracted_audio_path": "/tmp/audio.wav"})
        return media


def test_router_prepares_heavy_artifacts_only_for_fallback_modes(
    settings: Settings,
    provider_context: ProviderContext,
) -> None:
    media_preparer = FakeMediaPreparer()
    router = make_router(
        settings,
        {
            "qwen_omni": FakeProvider(
                "qwen_omni",
                error=ProviderError("timeout", "boom"),
            ),
            "gemini": FakeProvider(
                "gemini",
                error=ProviderError("timeout", "boom"),
            ),
            "fusion": FakeProvider(
                "fusion",
                result=make_result(
                    risk_score=0.4,
                    risk_label="cognitive_risk",
                    session_usability="usable",
                ),
            ),
            "audio_only": FakeProvider("audio_only", result=make_result()),
        },
        media_preparer=media_preparer,
    )

    assessment = asyncio.run(router.analyze(provider_context))

    assert assessment.provider_meta.final_provider == "fusion"
    assert media_preparer.calls == ["qwen_omni", "gemini", "fusion"]
