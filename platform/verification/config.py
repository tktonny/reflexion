"""Environment-backed settings for the verification pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from clinic.configs.settings import _load_local_env


_load_local_env()


@dataclass(frozen=True)
class VerificationSettings:
    project_root: Path
    data_dir: Path
    results_dir: Path
    prepared_dir: Path
    qwen_api_key: str | None
    qwen_base_url: str
    coding_plan_api_key: str | None
    coding_plan_base_url: str
    qwen_text_model: str
    qwen_asr_model: str
    request_timeout_seconds: float
    ffmpeg_binary: str
    ffprobe_binary: str
    talkbank_auth_base_url: str
    talkbank_media_base_url: str
    talkbank_email: str | None
    talkbank_password: str | None


@lru_cache(maxsize=1)
def get_verification_settings() -> VerificationSettings:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(
        os.getenv("REFLEXION_VERIFICATION_DATA_DIR", str(project_root / "verification" / "data"))
    ).resolve()
    results_dir = Path(
        os.getenv("REFLEXION_VERIFICATION_RESULTS_DIR", str(project_root / "verification" / "results"))
    ).resolve()
    prepared_dir = data_dir / "prepared"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    return VerificationSettings(
        project_root=project_root,
        data_dir=data_dir,
        results_dir=results_dir,
        prepared_dir=prepared_dir,
        qwen_api_key=(
            os.getenv("QWEN_API_KEY")
            or os.getenv("REFLEXION_QWEN_API_KEY")
        ),
        qwen_base_url=os.getenv(
            "REFLEXION_QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/"),
        coding_plan_api_key=(
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("CODING_PLAN_API_KEY")
            or os.getenv("REFLEXION_CODING_PLAN_API_KEY")
        ),
        coding_plan_base_url=os.getenv(
            "REFLEXION_CODING_PLAN_BASE_URL",
            "https://coding.dashscope.aliyuncs.com/v1",
        ).rstrip("/"),
        qwen_text_model=os.getenv("REFLEXION_VERIFICATION_QWEN_TEXT_MODEL", "qwen3.5-plus"),
        qwen_asr_model=os.getenv("REFLEXION_VERIFICATION_QWEN_ASR_MODEL", "qwen3-asr-flash"),
        request_timeout_seconds=float(
            os.getenv("REFLEXION_VERIFICATION_TIMEOUT_SECONDS", "300")
        ),
        ffmpeg_binary=os.getenv("REFLEXION_FFMPEG_BINARY", "ffmpeg"),
        ffprobe_binary=os.getenv("REFLEXION_FFPROBE_BINARY", "ffprobe"),
        talkbank_auth_base_url=os.getenv(
            "REFLEXION_TALKBANK_AUTH_BASE_URL",
            "https://sla2.talkbank.org",
        ).rstrip("/"),
        talkbank_media_base_url=os.getenv(
            "REFLEXION_TALKBANK_MEDIA_BASE_URL",
            "https://media.talkbank.org",
        ).rstrip("/"),
        talkbank_email=os.getenv("TALKBANK_EMAIL"),
        talkbank_password=os.getenv("TALKBANK_PASSWORD"),
    )
