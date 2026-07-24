"""CLI smoke test for a single clinic provider against a local sample video."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinic.configs.settings import SUPPORTED_PROVIDERS, get_settings
from backend.src.app.core.errors import RoutingError
from backend.src.app.models import ProviderContext
from backend.src.app.services.assessment_service import ProviderMeshRouter
from backend.src.app.services.media_preparer import MediaPreparer


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for a single-provider smoke test."""

    parser = argparse.ArgumentParser(description="Smoke-test a single clinic demo provider.")
    parser.add_argument("--video", required=True, help="Path to a local input video")
    parser.add_argument(
        "--provider",
        default="qwen_omni",
        choices=SUPPORTED_PROVIDERS,
        help="Preferred provider.",
    )
    parser.add_argument("--strict", action="store_true", help="Disable fallback")
    parser.add_argument("--patient-id", default="smoke-patient")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep prepared media and saved assessment files",
    )
    return parser.parse_args()


async def run_cli() -> int:
    """Run one provider against one local video and print the normalized result."""

    args = parse_args()
    settings = get_settings()
    router = ProviderMeshRouter(settings)
    media_preparer = MediaPreparer(settings)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(json.dumps({"error": "missing_video", "message": str(video_path)}, indent=2))
        return 2

    assessment_id = uuid.uuid4().hex
    upload_dir = settings.uploads_dir / assessment_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / video_path.name
    shutil.copy2(video_path, stored_path)

    prepared = media_preparer.prepare_base(assessment_id, stored_path)
    context = ProviderContext(
        assessment_id=assessment_id,
        patient_id=args.patient_id,
        language=args.language,
        preferred_provider=args.provider,
        strict_provider=args.strict,
        media=prepared,
    )

    try:
        assessment = await router.analyze(context)
    except RoutingError as exc:
        print(
            json.dumps(
                {
                    "error": exc.code,
                    "message": exc.message,
                    "provider_trace": exc.provider_trace,
                },
                indent=2,
            )
        )
        return 1
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(settings.uploads_dir / assessment_id, ignore_errors=True)
            shutil.rmtree(settings.prepared_dir / assessment_id, ignore_errors=True)

    print(assessment.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_cli()))
