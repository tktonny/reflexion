"""Build an ADReSSo manifest once audio files are available locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.config import get_verification_settings
from verification.datasets import build_adresso_records, write_manifest_jsonl


def parse_args() -> argparse.Namespace:
    settings = get_verification_settings()
    parser = argparse.ArgumentParser(description="Create verification/data/adresso/manifest.jsonl")
    parser.add_argument(
        "--audio-root",
        default=str(settings.data_dir / "adresso" / "audio"),
        help="Root directory containing ADReSSo audio files.",
    )
    parser.add_argument(
        "--labels",
        default=str(settings.data_dir / "adresso" / "groundtruth" / "task1.csv"),
        help="ADReSSo label CSV, e.g. task1.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(settings.data_dir / "adresso" / "manifest.jsonl"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split name stored in the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_adresso_records(
        Path(args.audio_root),
        Path(args.labels),
        split=args.split,
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
