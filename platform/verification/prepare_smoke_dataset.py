"""Prepare a tiny local smoke dataset from bundled YT-DemTalk clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.config import get_verification_settings
from verification.datasets import normalize_label, write_manifest_jsonl
from verification.models import VerificationRecord


def parse_args() -> argparse.Namespace:
    settings = get_verification_settings()
    parser = argparse.ArgumentParser(
        description="Create a smoke-test manifest from data/sample_videos/yt_demtalk/manifest.json"
    )
    parser.add_argument(
        "--output",
        default=str(settings.data_dir / "smoke" / "manifest.jsonl"),
        help="Output manifest path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_verification_settings()
    source_manifest = settings.project_root / "data" / "sample_videos" / "yt_demtalk" / "manifest.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    records: list[VerificationRecord] = []
    for item in payload:
        local_path = settings.project_root / item["local_path"]
        records.append(
            VerificationRecord(
                case_id=Path(local_path).stem,
                dataset="yt_demtalk_smoke",
                split=item.get("split") or "unknown",
                label=normalize_label(item["label"]),
                media_path=str(local_path.resolve()),
                media_type="video",
                language="en",
                metadata={
                    "source_url": item.get("url"),
                    "clip_window": item.get("clip_window"),
                },
            )
        )
    output_path = Path(args.output).expanduser().resolve()
    write_manifest_jsonl(records, output_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "record_count": len(records),
                "manifest_path": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
