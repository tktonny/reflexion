"""Prune duplicate TalkBank audio variants while keeping one preferred copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.talkbank_verification_data import build_logical_source_key


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
EXTENSION_RANK = {
    ".mp3": 0,
    ".m4a": 1,
    ".aac": 2,
    ".ogg": 3,
    ".opus": 4,
    ".flac": 5,
    ".wav": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove duplicate audio variants in a TalkBank corpus root.")
    parser.add_argument(
        "--root",
        required=True,
        help="Corpus root directory, e.g. verification/data/talkbank/corpora/english/pitt",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset slug used for logical keying, e.g. talkbank_english_pitt",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete redundant files. Default is dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    by_key: dict[tuple[str, ...], list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        key = build_logical_source_key(args.dataset, rel)
        by_key.setdefault(key, []).append(path)

    delete_paths: list[Path] = []
    kept_paths: list[Path] = []
    for paths in by_key.values():
        if len(paths) == 1:
            kept_paths.extend(paths)
            continue
        ordered = sorted(
            paths,
            key=lambda item: (
                EXTENSION_RANK.get(item.suffix.lower(), 99),
                item.stat().st_size,
                str(item),
            ),
        )
        kept_paths.append(ordered[0])
        delete_paths.extend(ordered[1:])

    reclaimed_bytes = sum(path.stat().st_size for path in delete_paths if path.exists())
    if args.apply:
        for path in delete_paths:
            path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "root": str(root),
                "dataset": args.dataset,
                "duplicate_logical_items": sum(1 for paths in by_key.values() if len(paths) > 1),
                "kept_file_count": len(kept_paths),
                "delete_file_count": len(delete_paths),
                "reclaimed_gib": round(reclaimed_bytes / (1024**3), 2),
                "applied": args.apply,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
