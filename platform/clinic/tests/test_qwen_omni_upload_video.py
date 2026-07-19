"""Manual CLI harness for one end-to-end clinic assessment run."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_FOLDER = PROJECT_ROOT / "data" / "sample_videos" / "yt_demtalk"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "clinic" / "tests" / "output"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinic.configs.settings import SUPPORTED_PROVIDERS, get_settings
from backend.src.app.core.errors import RoutingError
from backend.src.app.models import ProviderContext
from backend.src.app.services.assessment_service import ProviderMeshRouter
from backend.src.app.services.media_preparer import MediaPreparer


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".flv",
    ".wmv",
    ".m4v",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for a single local assessment run."""

    parser = argparse.ArgumentParser(
        description=(
            "Select one video from a folder, run one clinic assessment flow, "
            "and write the returned assessment JSON to disk."
        )
    )
    parser.add_argument(
        "--video-folder",
        default=str(DEFAULT_VIDEO_FOLDER),
        help="Folder containing input videos.",
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="Optional exact video filename inside the folder. If omitted, the first supported video is used.",
    )
    parser.add_argument(
        "--provider",
        default="qwen_omni",
        choices=SUPPORTED_PROVIDERS,
        help="Provider to use. Defaults to qwen_omni.",
    )
    parser.add_argument(
        "--allow-fallback",
        dest="strict",
        action="store_false",
        help="Enable fallback instead of running strict single-provider validation.",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--patient-id",
        default="demo-patient",
        help="Patient identifier stored in the request metadata.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Expected spoken language.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where assessment JSON files will be written.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep prepared upload/media artifacts under data/uploads and data/prepared.",
    )
    return parser.parse_args()


def select_video(video_folder: Path, video_name: str | None) -> Path:
    """Pick one supported input video from a folder."""

    if not video_folder.exists() or not video_folder.is_dir():
        raise FileNotFoundError(f"video folder not found: {video_folder}")

    if video_name:
        target = (video_folder / video_name).resolve()
        if not target.exists():
            raise FileNotFoundError(f"video not found: {target}")
        return target

    candidates = sorted(
        path
        for path in video_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"no supported video files found in: {video_folder}")
    return candidates[0].resolve()


async def run_cli() -> int:
    """Run a manual end-to-end assessment and persist the resulting JSON."""

    args = parse_args()
    get_settings.cache_clear()
    settings = get_settings()

    if args.provider == "qwen_omni" and not settings.qwen_omni_api_key:
        raise RuntimeError(
            "qwen_omni is not configured. Set QWEN_API_KEY or DASHSCOPE_API_KEY in .secret/.env."
        )

    video_path = select_video(Path(args.video_folder).expanduser().resolve(), args.video_name)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assessment_id = uuid.uuid4().hex
    upload_dir = settings.uploads_dir / assessment_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / video_path.name
    shutil.copy2(video_path, stored_path)

    media_preparer = MediaPreparer(settings)
    prepared = media_preparer.prepare_base(assessment_id, stored_path)

    router = ProviderMeshRouter(settings)
    context = ProviderContext(
        assessment_id=assessment_id,
        patient_id=args.patient_id,
        language=args.language,
        preferred_provider=args.provider,
        strict_provider=args.strict,
        media=prepared,
    )

    target_stem = f"{video_path.stem}.{args.provider}"
    success_path = output_dir / f"{target_stem}.assessment.json"
    error_path = output_dir / f"{target_stem}.error.json"

    try:
        assessment = await router.analyze(context)
    except RoutingError as exc:
        payload = {
            "error": exc.code,
            "message": exc.message,
            "provider_trace": exc.provider_trace,
            "video_path": str(video_path),
            "provider": args.provider,
            "strict_provider": args.strict,
            "prepared_media": {
                "standardized_path": prepared.standardized_path,
                "mime_type": prepared.mime_type,
                "size_bytes": prepared.size_bytes,
                "duration_seconds": prepared.duration_seconds,
                "extracted_audio_path": prepared.extracted_audio_path,
                "frame_count": len(prepared.frame_paths),
            },
            "error_file": str(error_path),
        }
        error_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(settings.uploads_dir / assessment_id, ignore_errors=True)
            shutil.rmtree(settings.prepared_dir / assessment_id, ignore_errors=True)

    success_path.write_text(assessment.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "video_path": str(video_path),
                "provider": args.provider,
                "result_file": str(success_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_cli()))
