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

    STANDARDIZED_MAX_WIDTH = 854
    STANDARDIZED_VIDEO_BITRATE_MIN_KBPS = 420
    STANDARDIZED_VIDEO_BITRATE_MAX_KBPS = 2800
    STANDARDIZED_CONTAINER_OVERHEAD_BYTES = 160 * 1024

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_base(self, assessment_id: str, input_path: Path) -> PreparedMedia:
        """Create the standardized video and basic metadata used by every provider."""

        prepared_dir = self.settings.prepared_dir / assessment_id
        prepared_dir.mkdir(parents=True, exist_ok=True)

        input_mime = mimetypes.guess_type(input_path.name)[0] or "video/mp4"
        input_duration_seconds = None
        if shutil.which(self.settings.ffprobe_binary):
            try:
                input_duration_seconds = self._probe_duration(input_path)
            except ProviderError:
                input_duration_seconds = None

        if shutil.which(self.settings.ffmpeg_binary):
            standardized_path = prepared_dir / "standardized.mp4"
            command = self._build_standardize_command(
                input_path,
                standardized_path,
                duration_seconds=input_duration_seconds,
            )
            self._run(command, "unsupported_media", "ffmpeg standardization failed")
        else:
            standardized_path = self._build_passthrough_target(prepared_dir, input_path)
            shutil.copy2(input_path, standardized_path)

        duration_seconds = None
        if shutil.which(self.settings.ffprobe_binary):
            try:
                duration_seconds = self._probe_duration(standardized_path)
            except ProviderError:
                duration_seconds = None

        size_bytes = standardized_path.stat().st_size
        final_mime = mimetypes.guess_type(standardized_path.name)[0] or input_mime

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

    def _estimate_standardized_total_bytes(self, duration_seconds: float | None) -> int:
        min_bytes = max(1, self.settings.standardized_video_min_mb) * 1024 * 1024
        max_bytes = max(self.settings.standardized_video_max_mb, self.settings.standardized_video_min_mb) * 1024 * 1024
        if not duration_seconds or duration_seconds <= 0:
            return min(max_bytes, max(min_bytes, 6 * 1024 * 1024))
        target_total_kbps = max(600, self.settings.standardized_video_target_total_kbps)
        estimated = int(duration_seconds * target_total_kbps * 1000 / 8)
        return max(min_bytes, min(max_bytes, estimated))

    def _standardized_bitrate_profile(self, duration_seconds: float | None) -> tuple[int, int]:
        audio_kbps = max(48, self.settings.standardized_audio_bitrate_kbps)
        if not duration_seconds or duration_seconds <= 0:
            return 1400, audio_kbps

        target_total_bytes = self._estimate_standardized_total_bytes(duration_seconds)
        audio_bytes = int(duration_seconds * audio_kbps * 1000 / 8)
        video_bytes = max(
            256 * 1024,
            target_total_bytes - audio_bytes - self.STANDARDIZED_CONTAINER_OVERHEAD_BYTES,
        )
        video_kbps = int((video_bytes * 8) / max(duration_seconds, 1.0) / 1000)
        video_kbps = max(
            self.STANDARDIZED_VIDEO_BITRATE_MIN_KBPS,
            min(self.STANDARDIZED_VIDEO_BITRATE_MAX_KBPS, video_kbps),
        )
        return video_kbps, audio_kbps

    def _build_standardize_command(
        self,
        input_path: Path,
        standardized_path: Path,
        *,
        duration_seconds: float | None,
    ) -> list[str]:
        video_kbps, audio_kbps = self._standardized_bitrate_profile(duration_seconds)
        maxrate_kbps = int(video_kbps * 1.15)
        bufsize_kbps = max(video_kbps * 2, 900)
        return [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            f"scale='min({self.STANDARDIZED_MAX_WIDTH},iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            f"{video_kbps}k",
            "-maxrate",
            f"{maxrate_kbps}k",
            "-bufsize",
            f"{bufsize_kbps}k",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            f"{audio_kbps}k",
            str(standardized_path),
        ]

    def _run(self, command: list[str], code: str, message: str) -> None:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise ProviderError(code, proc.stderr.strip() or message)

    def _build_passthrough_target(self, prepared_dir: Path, input_path: Path) -> Path:
        suffix = input_path.suffix.lower()
        if suffix == ".mp4":
            return prepared_dir / "standardized.mp4"
        if not suffix:
            suffix = ".bin"
        return prepared_dir / f"standardized{suffix}"
