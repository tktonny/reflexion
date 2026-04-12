"""Shared environment-backed settings for the clinic product."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from backend.src.app.models.assessment import ProviderName


SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = (
    "qwen_omni",
    "gemini",
    "fusion",
    "audio_only",
)


def _load_local_env() -> None:
    settings_path = Path(__file__).resolve()
    project_root = settings_path.parents[2]
    for env_path in (
        project_root / ".secret" / ".env",
        project_root / ".env",
        settings_path.with_name(".env"),
    ):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_provider_name(value: str, *, field_name: str) -> ProviderName:
    clean_value = value.strip()
    if clean_value not in SUPPORTED_PROVIDERS:
        supported_values = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"{field_name} must be one of: {supported_values}")
    return clean_value  # type: ignore[return-value]


def _parse_fallback_order(raw_value: str) -> tuple[ProviderName, ...]:
    fallback_order = tuple(
        _parse_provider_name(item, field_name="REFLEXION_FALLBACK_ORDER")
        for item in raw_value.split(",")
        if item.strip()
    )
    if not fallback_order:
        raise ValueError("REFLEXION_FALLBACK_ORDER must contain at least one provider")
    return fallback_order


@dataclass(frozen=True)
class Settings:
    app_name: str
    storage_dir: Path
    uploads_dir: Path
    prepared_dir: Path
    assessments_dir: Path
    server_host: str
    server_port: int
    server_reload: bool
    max_upload_mb: int
    max_inline_video_mb: int
    allow_mock_providers: bool
    default_provider: ProviderName
    fallback_order: tuple[ProviderName, ...]
    ffmpeg_binary: str
    ffprobe_binary: str
    qwen_omni_api_key: str | None
    qwen_omni_base_url: str
    qwen_omni_model: str
    gemini_api_key: str | None
    gemini_base_url: str
    gemini_model: str
    openai_api_key: str | None
    openai_base_url: str
    openai_fusion_model: str
    openai_text_model: str
    openai_transcription_model: str
    qwen_omni_realtime_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    qwen_omni_realtime_model: str = "qwen3-omni-flash-realtime"
    qwen_omni_realtime_transcription_model: str = "gummy-realtime-v1"
    qwen_omni_realtime_default_voice: str = "Cherry"
    realtime_flow_path: Path | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_local_env()
    project_root = Path(__file__).resolve().parents[2]
    storage_dir = Path(
        os.getenv("REFLEXION_STORAGE_DIR", str(project_root / "data"))
    ).resolve()
    uploads_dir = storage_dir / "uploads"
    prepared_dir = storage_dir / "prepared"
    assessments_dir = storage_dir / "assessments"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    assessments_dir.mkdir(parents=True, exist_ok=True)

    fallback_order = _parse_fallback_order(
        os.getenv(
            "REFLEXION_FALLBACK_ORDER",
            ",".join(SUPPORTED_PROVIDERS),
        )
    )

    return Settings(
        app_name="Reflexion Clinic Demo",
        storage_dir=storage_dir,
        uploads_dir=uploads_dir,
        prepared_dir=prepared_dir,
        assessments_dir=assessments_dir,
        server_host=os.getenv("REFLEXION_SERVER_HOST", "0.0.0.0"),
        server_port=int(os.getenv("REFLEXION_SERVER_PORT", "8000")),
        server_reload=_as_bool(os.getenv("REFLEXION_SERVER_RELOAD"), default=True),
        max_upload_mb=int(os.getenv("REFLEXION_MAX_UPLOAD_MB", "250")),
        max_inline_video_mb=int(os.getenv("REFLEXION_MAX_INLINE_VIDEO_MB", "15")),
        allow_mock_providers=_as_bool(
            os.getenv("REFLEXION_ALLOW_MOCK_PROVIDERS"),
            default=False,
        ),
        default_provider=_parse_provider_name(
            os.getenv("REFLEXION_DEFAULT_PROVIDER", "qwen_omni"),
            field_name="REFLEXION_DEFAULT_PROVIDER",
        ),
        fallback_order=fallback_order,
        ffmpeg_binary=os.getenv("REFLEXION_FFMPEG_BINARY", "ffmpeg"),
        ffprobe_binary=os.getenv("REFLEXION_FFPROBE_BINARY", "ffprobe"),
        qwen_omni_api_key=os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        qwen_omni_base_url=os.getenv(
            "REFLEXION_QWEN_OMNI_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        qwen_omni_model=os.getenv("REFLEXION_QWEN_OMNI_MODEL", "qwen3.5-omni-plus"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_base_url=os.getenv(
            "REFLEXION_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com",
        ),
        gemini_model=os.getenv("REFLEXION_GEMINI_MODEL", "gemini-3.1-pro-preview"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("REFLEXION_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_fusion_model=os.getenv("REFLEXION_OPENAI_FUSION_MODEL", "gpt-4.1"),
        openai_text_model=os.getenv("REFLEXION_OPENAI_TEXT_MODEL", "gpt-4.1"),
        openai_transcription_model=os.getenv(
            "REFLEXION_OPENAI_TRANSCRIPTION_MODEL",
            "gpt-4o-transcribe",
        ),
        qwen_omni_realtime_url=os.getenv(
            "REFLEXION_QWEN_OMNI_REALTIME_URL",
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        ),
        qwen_omni_realtime_model=os.getenv(
            "REFLEXION_QWEN_OMNI_REALTIME_MODEL",
            "qwen3-omni-flash-realtime",
        ),
        qwen_omni_realtime_transcription_model=os.getenv(
            "REFLEXION_QWEN_OMNI_REALTIME_TRANSCRIPTION_MODEL",
            "gummy-realtime-v1",
        ),
        qwen_omni_realtime_default_voice=os.getenv(
            "REFLEXION_QWEN_OMNI_REALTIME_DEFAULT_VOICE",
            "Cherry",
        ),
        realtime_flow_path=(
            Path(raw_path).resolve()
            if (raw_path := os.getenv("REFLEXION_REALTIME_FLOW_PATH"))
            else None
        ),
    )
