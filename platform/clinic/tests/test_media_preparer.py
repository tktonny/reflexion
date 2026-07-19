"""Unit tests for staged media preparation behavior."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from clinic.configs.settings import Settings
from backend.src.app.models import PreparedMedia
from backend.src.app.services.media_preparer import MediaPreparer


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


@pytest.fixture
def base_media(tmp_path: Path) -> PreparedMedia:
    return PreparedMedia(
        original_path="/tmp/input.mp4",
        standardized_path=str(tmp_path / "prepared" / "a1" / "standardized.mp4"),
        mime_type="video/mp4",
        size_bytes=1024,
        duration_seconds=12.3,
    )


def test_prepare_for_provider_keeps_omni_lightweight(
    tmp_path: Path,
    base_media: PreparedMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = MediaPreparer(make_settings(tmp_path))
    calls: list[str] = []

    def fake_ensure_audio(media: PreparedMedia) -> PreparedMedia:
        calls.append("audio")
        return media

    def fake_ensure_frames(media: PreparedMedia) -> PreparedMedia:
        calls.append("frames")
        return media

    monkeypatch.setattr(preparer, "ensure_audio_artifact", fake_ensure_audio)
    monkeypatch.setattr(preparer, "ensure_frame_artifacts", fake_ensure_frames)

    qwen_media = preparer.prepare_for_provider("qwen_omni", base_media)
    gemini_media = preparer.prepare_for_provider("gemini", base_media)

    assert qwen_media == base_media
    assert gemini_media == base_media
    assert calls == []


def test_prepare_for_provider_builds_only_needed_fallback_artifacts(
    tmp_path: Path,
    base_media: PreparedMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = MediaPreparer(make_settings(tmp_path))
    calls: list[str] = []

    def fake_ensure_audio(media: PreparedMedia) -> PreparedMedia:
        calls.append("audio")
        return media.model_copy(update={"extracted_audio_path": "/tmp/audio.wav"})

    def fake_ensure_frames(media: PreparedMedia) -> PreparedMedia:
        calls.append("frames")
        return media.model_copy(update={"frame_paths": ["/tmp/frame-001.jpg"]})

    monkeypatch.setattr(preparer, "ensure_audio_artifact", fake_ensure_audio)
    monkeypatch.setattr(preparer, "ensure_frame_artifacts", fake_ensure_frames)

    fusion_media = preparer.prepare_for_provider("fusion", base_media)
    audio_only_media = preparer.prepare_for_provider("audio_only", base_media)

    assert fusion_media.extracted_audio_path == "/tmp/audio.wav"
    assert fusion_media.frame_paths == ["/tmp/frame-001.jpg"]
    assert audio_only_media.extracted_audio_path == "/tmp/audio.wav"
    assert audio_only_media.frame_paths == []
    assert calls == ["audio", "frames", "audio"]


def test_prepare_base_preserves_non_mp4_suffix_without_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = MediaPreparer(make_settings(tmp_path))
    source = tmp_path / "input.webm"
    source.write_bytes(b"webm-bytes")

    original_which = shutil.which

    def fake_which(binary: str) -> str | None:
        if binary in {preparer.settings.ffmpeg_binary, preparer.settings.ffprobe_binary}:
            return None
        return original_which(binary)

    monkeypatch.setattr(shutil, "which", fake_which)

    prepared = preparer.prepare_base("a1", source)

    assert prepared.standardized_path.endswith("standardized.webm")
    assert prepared.mime_type == "video/webm"
    assert Path(prepared.standardized_path).read_bytes() == b"webm-bytes"
    assert prepared.duration_seconds is None


def test_prepare_base_keeps_mp4_passthrough_name_without_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = MediaPreparer(make_settings(tmp_path))
    source = tmp_path / "input.mp4"
    source.write_bytes(b"mp4-bytes")

    original_which = shutil.which

    def fake_which(binary: str) -> str | None:
        if binary in {preparer.settings.ffmpeg_binary, preparer.settings.ffprobe_binary}:
            return None
        return original_which(binary)

    monkeypatch.setattr(shutil, "which", fake_which)

    prepared = preparer.prepare_base("a2", source)

    assert prepared.standardized_path.endswith("standardized.mp4")
    assert prepared.mime_type == "video/mp4"
    assert Path(prepared.standardized_path).read_bytes() == b"mp4-bytes"


def test_estimated_standardized_size_stays_within_target_window(tmp_path: Path) -> None:
    preparer = MediaPreparer(make_settings(tmp_path))

    short_clip_bytes = preparer._estimate_standardized_total_bytes(8.0)
    typical_clip_bytes = preparer._estimate_standardized_total_bytes(30.0)
    long_clip_bytes = preparer._estimate_standardized_total_bytes(120.0)

    assert 2 * 1024 * 1024 <= short_clip_bytes <= 10 * 1024 * 1024
    assert 2 * 1024 * 1024 <= typical_clip_bytes <= 10 * 1024 * 1024
    assert long_clip_bytes == 10 * 1024 * 1024


def test_build_standardize_command_uses_adaptive_bitrate_targets(tmp_path: Path) -> None:
    preparer = MediaPreparer(make_settings(tmp_path))

    command = preparer._build_standardize_command(
        tmp_path / "input.webm",
        tmp_path / "prepared" / "a1" / "standardized.mp4",
        duration_seconds=30.0,
    )

    assert "-b:v" in command
    assert "-maxrate" in command
    assert "-bufsize" in command
    assert "libx264" in command
    assert "aac" in command
    assert "96k" in command
    assert "scale='min(854,iw)':-2" in command
    assert "-crf" not in command
