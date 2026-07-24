"""Media preparation utilities for verification datasets."""

from __future__ import annotations

import mimetypes
import shutil
import subprocess
from pathlib import Path

from backend.src.app.models import PreparedMedia
from verification.config import VerificationSettings
from verification.models import PreparedAudioArtifact, VerificationRecord


def prepare_audio_artifact(
    record: VerificationRecord,
    settings: VerificationSettings,
) -> PreparedAudioArtifact:
    source_path = Path(record.media_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"media file not found: {source_path}")

    prepared_dir = settings.prepared_dir / record.case_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_path = prepared_dir / "audio.mp3"

    if output_path.exists():
        return PreparedAudioArtifact(
            case_id=record.case_id,
            source_path=str(source_path),
            audio_path=str(output_path),
            mime_type="audio/mpeg",
            size_bytes=output_path.stat().st_size,
            duration_seconds=_probe_duration(output_path, settings),
        )

    if shutil.which(settings.ffmpeg_binary):
        command = [
            settings.ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            str(output_path),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffmpeg audio extraction failed")
    elif record.media_type == "audio":
        shutil.copy2(source_path, output_path)
    else:
        raise RuntimeError("ffmpeg is required to extract audio from video inputs")

    mime_type = mimetypes.guess_type(output_path.name)[0] or "audio/mpeg"
    return PreparedAudioArtifact(
        case_id=record.case_id,
        source_path=str(source_path),
        audio_path=str(output_path),
        mime_type=mime_type,
        size_bytes=output_path.stat().st_size,
        duration_seconds=_probe_duration(output_path, settings),
    )


def prepare_omni_media_artifact(
    record: VerificationRecord,
    settings: VerificationSettings,
) -> PreparedMedia:
    source_path = Path(record.media_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"media file not found: {source_path}")

    prepared_dir = settings.prepared_dir / record.case_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_path = prepared_dir / "omni_input.mp4"

    if output_path.exists():
        return PreparedMedia(
            original_path=str(source_path),
            standardized_path=str(output_path),
            mime_type="video/mp4",
            size_bytes=output_path.stat().st_size,
            duration_seconds=_probe_duration(output_path, settings),
        )

    ffmpeg_binary = shutil.which(settings.ffmpeg_binary)
    if not ffmpeg_binary:
        if record.media_type == "video" and source_path.suffix.lower() == ".mp4":
            shutil.copy2(source_path, output_path)
        else:
            raise RuntimeError("ffmpeg is required to build qwen_omni verification media")
    elif record.media_type == "audio":
        command = [
            ffmpeg_binary,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:r=1",
            "-i",
            str(source_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "stillimage",
            "-crf",
            "35",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "32k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        _run(command, "ffmpeg omni audio-to-video packaging failed")
    else:
        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-vf",
            "scale='min(854,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(output_path),
        ]
        _run(command, "ffmpeg omni video standardization failed")

    return PreparedMedia(
        original_path=str(source_path),
        standardized_path=str(output_path),
        mime_type="video/mp4",
        size_bytes=output_path.stat().st_size,
        duration_seconds=_probe_duration(output_path, settings),
    )


def _probe_duration(path: Path, settings: VerificationSettings) -> float | None:
    if not shutil.which(settings.ffprobe_binary):
        return None
    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _run(command: list[str], message: str) -> None:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or message)
