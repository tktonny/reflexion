"""List or download a shared Google Drive folder into verification/data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gdown

from verification.config import get_verification_settings


def parse_args() -> argparse.Namespace:
    settings = get_verification_settings()
    parser = argparse.ArgumentParser(
        description="List or download a shared Google Drive folder for verification datasets."
    )
    parser.add_argument(
        "url",
        help="Shared Google Drive folder URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.data_dir / "imports"),
        help="Destination directory when downloading.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only inspect folder contents without downloading files.",
    )
    parser.add_argument(
        "--remaining-ok",
        action="store_true",
        help="Allow gdown's max-50-files folder handling without interactive confirmation.",
    )
    parser.add_argument(
        "--no-cookies",
        action="store_true",
        help="Disable cookie usage while accessing Google Drive.",
    )
    parser.add_argument(
        "--no-check-certificate",
        action="store_true",
        help="Disable TLS certificate verification for the Drive request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = gdown.download_folder(
            url=args.url,
            output=str(output_dir) + "/",
            quiet=True,
            remaining_ok=args.remaining_ok,
            skip_download=args.list_only,
            use_cookies=not args.no_cookies,
            verify=not args.no_check_certificate,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "operation": "list" if args.list_only else "download",
                    "url": args.url,
                    "output_dir": str(output_dir),
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1

    if files is None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "operation": "list" if args.list_only else "download",
                    "url": args.url,
                    "output_dir": str(output_dir),
                    "error": "gdown returned no result",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1

    if args.list_only:
        preview = [
            {
                "id": item.id,
                "path": item.path,
                "local_path": item.local_path,
            }
            for item in files[:100]
        ]
        payload = {
            "status": "ok",
            "operation": "list",
            "url": args.url,
            "count": len(files),
            "preview": preview,
        }
    else:
        payload = {
            "status": "ok",
            "operation": "download",
            "url": args.url,
            "count": len(files),
            "downloaded_paths": files,
        }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
