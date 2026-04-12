"""Download TalkBank verification data and build runnable manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tarfile
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.config import get_verification_settings
from verification.datasets import build_adresso_records, write_manifest_jsonl
from verification.models import VerificationRecord
from verification.talkbank import TalkBankClient, write_json
from verification.talkbank_catalog import write_talkbank_catalog


PUBLIC_ADRESSO_TASK1_URL = "https://talkbank.org/dementia/ADReSSo-2021/groundtruth/task1.csv"
ADRESSO_MEDIA_PATH = "dementia/English/0extra/ADReSSo/"
PITT_CONTROL_MEDIA_PATH = "dementia/English/Pitt/Control/cookie/"
PITT_DEMENTIA_MEDIA_PATH = "dementia/English/Pitt/Dementia/cookie/"
DIAGNOSIS_ARCHIVES = {
    "ADReSSo21-diagnosis-train.tgz": "diagnosis_train",
    "ADReSSo21-diagnosis-test.tgz": "diagnosis_test",
}
PITT_GROUPS = (
    ("Control", "HC", PITT_CONTROL_MEDIA_PATH, "control"),
    ("Dementia", "cognitive_risk", PITT_DEMENTIA_MEDIA_PATH, "dementia"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TalkBank verification data and build runnable manifests."
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Optional TalkBank email. Falls back to TALKBANK_EMAIL.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Optional TalkBank password. Falls back to TALKBANK_PASSWORD.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload archives or media that already exist locally.",
    )
    parser.add_argument(
        "--skip-pitt",
        action="store_true",
        help="Skip Pitt cookie audio download and manifest refresh.",
    )
    parser.add_argument(
        "--skip-adresso",
        action="store_true",
        help="Skip ADReSSo archive download and manifest refresh.",
    )
    return parser.parse_args()


def load_public_task1_rows(task1_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    task1_source = "downloaded"
    warning: str | None = None
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(PUBLIC_ADRESSO_TASK1_URL)
            response.raise_for_status()
            task1_path.write_bytes(response.content)
    except httpx.HTTPError as exc:
        if not task1_path.exists():
            raise
        task1_source = "cached"
        warning = f"Using cached task1.csv because refresh failed: {exc}"

    rows = list(csv.DictReader(task1_path.read_text(encoding="utf-8").splitlines()))
    payload: dict[str, Any] = {
        "adresso_task1_groundtruth": str(task1_path),
        "task1_case_count": len(rows),
        "task1_source": task1_source,
    }
    if warning:
        payload["task1_warning"] = warning
    return rows, payload


def media_url(settings: Any, relative_path: str) -> str:
    return f"{settings.talkbank_media_base_url}/{relative_path.lstrip('/')}"


def count_audio_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    )


def count_manifest_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def extract_archive(archive_path: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_count = count_audio_files(output_dir)
    marker_path = output_dir / ".extracted.json"
    if marker_path.exists() and not overwrite:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        payload["source"] = "cached"
        return payload

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output_dir)

    payload = {
        "archive": str(archive_path),
        "extract_dir": str(output_dir),
        "audio_file_count_before": before_count,
        "audio_file_count_after": count_audio_files(output_dir),
        "source": "extracted",
    }
    marker_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def download_pitt_group(
    client: TalkBankClient,
    *,
    settings: Any,
    group_name: str,
    label: str,
    relative_path: str,
    dest_group_name: str,
    overwrite: bool,
) -> tuple[list[VerificationRecord], dict[str, Any]]:
    group_url = media_url(settings, relative_path)
    entries = client.list_directory(group_url)
    audio_entries = [
        entry for entry in entries if Path(entry.name).suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
    ]

    dest_dir = settings.data_dir / "talkbank" / "pitt_cookie" / dest_group_name / "audio"
    records: list[VerificationRecord] = []
    downloaded_count = 0
    reused_count = 0
    for entry in audio_entries:
        dest_path = dest_dir / entry.name
        existed_before = dest_path.exists()
        client.download_to_path(entry.url, dest_path, overwrite=overwrite)
        if existed_before and not overwrite:
            reused_count += 1
        else:
            downloaded_count += 1

        records.append(
            VerificationRecord(
                case_id=Path(entry.name).stem,
                dataset="dementiabank_pitt_cookie",
                split="unknown",
                label=label,
                media_path=str(dest_path.resolve()),
                media_type="audio",
                language="en",
                metadata={
                    "source_group": group_name,
                    "source_url": entry.url,
                    "filename": entry.name,
                    "size_bytes": dest_path.stat().st_size,
                },
            )
        )

    return records, {
        "group": group_name,
        "group_url": group_url,
        "listed_count": len(audio_entries),
        "downloaded_count": downloaded_count,
        "reused_count": reused_count,
        "local_dir": str(dest_dir.resolve()),
    }


def download_adresso_archives(
    client: TalkBankClient,
    *,
    settings: Any,
    overwrite: bool,
) -> dict[str, Any]:
    adresso_root_url = media_url(settings, ADRESSO_MEDIA_PATH)
    entries = client.list_directory(adresso_root_url)
    entry_map = {entry.name: entry for entry in entries}

    raw_dir = settings.data_dir / "adresso" / "raw"
    audio_root = settings.data_dir / "adresso" / "audio"
    archive_reports: dict[str, Any] = {}
    for archive_name, extract_dir_name in DIAGNOSIS_ARCHIVES.items():
        if archive_name not in entry_map:
            raise FileNotFoundError(f"TalkBank directory did not list expected archive: {archive_name}")
        archive_entry = entry_map[archive_name]
        archive_path = raw_dir / archive_name
        existed_before = archive_path.exists()
        client.download_to_path(archive_entry.url, archive_path, overwrite=overwrite)
        extraction = extract_archive(
            archive_path,
            audio_root / extract_dir_name,
            overwrite=overwrite,
        )
        archive_reports[archive_name] = {
            "url": archive_entry.url,
            "archive_path": str(archive_path.resolve()),
            "downloaded": overwrite or not existed_before,
            "size_bytes": archive_path.stat().st_size,
            "extraction": extraction,
        }

    return {
        "root_url": adresso_root_url,
        "archives": archive_reports,
        "audio_root": str(audio_root.resolve()),
        "audio_file_count": count_audio_files(audio_root),
    }


def main() -> int:
    args = parse_args()
    settings = get_verification_settings()

    adresso_dir = settings.data_dir / "adresso"
    groundtruth_dir = adresso_dir / "groundtruth"
    groundtruth_dir.mkdir(parents=True, exist_ok=True)

    task1_path = groundtruth_dir / "task1.csv"
    rows, public_assets = load_public_task1_rows(task1_path)
    write_json(
        {
            "status": "ok",
            "public_assets": public_assets,
        },
        adresso_dir / "public_assets.json",
    )

    email = args.email or settings.talkbank_email
    password = args.password or settings.talkbank_password
    if not email or not password:
        print(
            json.dumps(
                {
                    "status": "partial",
                    "message": "Public ADReSSo labels downloaded. TalkBank credentials were not provided.",
                    "task1_groundtruth": str(task1_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    client = TalkBankClient(settings)
    try:
        try:
            login_payload = client.login(email=email, password=password)
            tree = client.get_anno_path_trees()
            write_json(tree, settings.data_dir / "talkbank" / "anno_path_tree.json")
            catalog = write_talkbank_catalog(
                tree,
                settings.data_dir / "talkbank" / "corpus_catalog.json",
            )

            pitt_records: list[VerificationRecord] = []
            pitt_report: list[dict[str, Any]] = []
            pitt_manifest_path = settings.data_dir / "talkbank" / "pitt_cookie" / "manifest.jsonl"
            pitt_record_count = count_manifest_records(pitt_manifest_path) if args.skip_pitt else 0
            adresso_manifest_path = settings.data_dir / "adresso" / "manifest.jsonl"
            if not args.skip_pitt:
                for group_name, label, relative_path, dest_group_name in PITT_GROUPS:
                    records, report = download_pitt_group(
                        client,
                        settings=settings,
                        group_name=group_name,
                        label=label,
                        relative_path=relative_path,
                        dest_group_name=dest_group_name,
                        overwrite=args.overwrite,
                    )
                    pitt_records.extend(records)
                    pitt_report.append(report)
                write_manifest_jsonl(pitt_records, pitt_manifest_path)
                pitt_record_count = len(pitt_records)

            adresso_report: dict[str, Any] | None = None
            adresso_manifest_records: list[VerificationRecord] = []
            adresso_record_count = count_manifest_records(adresso_manifest_path) if args.skip_adresso else 0
            if not args.skip_adresso:
                adresso_report = download_adresso_archives(
                    client,
                    settings=settings,
                    overwrite=args.overwrite,
                )
                adresso_manifest_records = build_adresso_records(
                    settings.data_dir / "adresso" / "audio",
                    task1_path,
                    split="test",
                )
                write_manifest_jsonl(adresso_manifest_records, adresso_manifest_path)
                adresso_record_count = len(adresso_manifest_records)

            report = {
                "login": login_payload,
                "paths": {
                    "adresso": client.session_has_auth(ADRESSO_MEDIA_PATH),
                    "pitt_control_cookie": client.session_has_auth(PITT_CONTROL_MEDIA_PATH),
                    "pitt_dementia_cookie": client.session_has_auth(PITT_DEMENTIA_MEDIA_PATH),
                },
                "task1_groundtruth": str(task1_path),
                "talkbank_catalog": {
                    "catalog_path": str((settings.data_dir / "talkbank" / "corpus_catalog.json").resolve()),
                    "summary": catalog["summary"],
                    "verification_candidates": catalog["verification_candidates"],
                },
                "pitt_cookie_manifest": {
                    "manifest_path": str(pitt_manifest_path.resolve()),
                    "record_count": pitt_record_count,
                    "groups": pitt_report,
                    "skipped": args.skip_pitt,
                },
                "adresso_manifest": {
                    "manifest_path": str(adresso_manifest_path.resolve()),
                    "record_count": adresso_record_count,
                    "skipped": args.skip_adresso,
                },
                "adresso_download": adresso_report,
            }
            write_json(report, settings.data_dir / "talkbank" / "access_report.json")

            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        except httpx.HTTPError as exc:
            error_report = {
                "status": "error",
                "stage": "talkbank_auth",
                "message": "TalkBank connection failed before authorization could be checked.",
                "error": str(exc),
                "task1_groundtruth": str(task1_path),
                "task1_source": public_assets["task1_source"],
            }
            write_json(error_report, settings.data_dir / "talkbank" / "access_report.json")
            print(json.dumps(error_report, indent=2, ensure_ascii=False))
            return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
