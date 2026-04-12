"""Media preparation utilities for clinic video uploads."""

from __future__ import annotations

import mimetypes
import shutil
import subprocess
from pathlib import Path

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError
from backend.src.app.models import PreparedMedia


class MediaPreparer:
    """Standardize video inputs and extract artifacts for fallback providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_base(self, assessment_id: str, input_path: Path) -> PreparedMedia:
        """Create the standardized video and basic metadata used by every provider."""

        prepared_dir = self.settings.prepared_dir / assessment_id
        prepared_dir.mkdir(parents=True, exist_ok=True)

        standardized_path = prepared_dir / "standardized.mp4"
        mime_type = mimetypes.guess_type(input_path.name)[0] or "video/mp4"

        if shutil.which(self.settings.ffmpeg_binary):
            command = [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(input_path),
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
                str(standardized_path),
            ]
            self._run(command, "unsupported_media", "ffmpeg standardization failed")
        else:
            shutil.copy2(input_path, standardized_path)

        duration_seconds = None
        if shutil.which(self.settings.ffprobe_binary):
            try:
                duration_seconds = self._probe_duration(standardized_path)
            except ProviderError:
                duration_seconds = None

        size_bytes = standardized_path.stat().st_size
        final_mime = mimetypes.guess_type(standardized_path.name)[0] or mime_type

        return PreparedMedia(
            original_path=str(input_path),
            standardized_path=str(standardized_path),
            mime_type=final_mime,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
        )

    def prepare(self, assessment_id: str, input_path: Path) -> PreparedMedia:
        """Backward-compatible alias for the base media preparation step."""

        return self.prepare_base(assessment_id, input_path)

    def prepare_for_provider(
        self,
        provider_name: str,
        media: PreparedMedia,
    ) -> PreparedMedia:
        """Materialize only the artifacts required by the selected provider."""

        if provider_name == "fusion":
            return self.ensure_frame_artifacts(self.ensure_audio_artifact(media))
        if provider_name == "audio_only":
            return self.ensure_audio_artifact(media)
        return media

    def ensure_audio_artifact(self, media: PreparedMedia) -> PreparedMedia:
        """Extract a mono wav file when a fallback provider needs audio input."""

        if media.extracted_audio_path:
            return media
        if not shutil.which(self.settings.ffmpeg_binary):
            return media

        audio_path = Path(media.standardized_path).with_name("audio.wav")
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            media.standardized_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        try:
            self._run(command, "audio_extract_failed", "audio extraction failed")
        except ProviderError:
            return media
        return media.model_copy(update={"extracted_audio_path": str(audio_path)})

    def ensure_frame_artifacts(self, media: PreparedMedia) -> PreparedMedia:
        """Extract sparse key frames when the fusion fallback needs visual snapshots."""

        if media.frame_paths:
            return media
        if not shutil.which(self.settings.ffmpeg_binary):
            return media

        prepared_dir = Path(media.standardized_path).parent
        frames_glob = prepared_dir / "frame-%03d.jpg"
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            media.standardized_path,
            "-vf",
            "fps=1/8",
            "-frames:v",
            "6",
            str(frames_glob),
        ]
        try:
            self._run(command, "frame_extract_failed", "frame extraction failed")
        except ProviderError:
            return media

        frame_paths = [str(path) for path in sorted(prepared_dir.glob("frame-*.jpg"))]
        return media.model_copy(update={"frame_paths": frame_paths})

    def _probe_duration(self, path: Path) -> float:
        command = [
            self.settings.ffprobe_binary,
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
            raise ProviderError("ffprobe_failed", proc.stderr.strip() or "ffprobe failed")
        return float(proc.stdout.strip())

    def _run(self, command: list[str], code: str, message: str) -> None:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise ProviderError(code, proc.stderr.strip() or message)
