"""Dataset manifest helpers for verification runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from verification.models import GroundTruthLabel, VerificationRecord


def normalize_label(raw_value: str) -> GroundTruthLabel:
    value = raw_value.strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"hc", "control", "healthy", "normotypical", "normal"}:
        return "HC"
    if value in {
        "cognitive_risk",
        "dementia",
        "probablead",
        "probable_ad",
        "ad",
        "alzheimers",
        "alzheimer",
    }:
        return "cognitive_risk"
    raise ValueError(f"unsupported label: {raw_value}")


def load_manifest(path: Path) -> list[VerificationRecord]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            VerificationRecord.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list of records")
        return [VerificationRecord.model_validate(item) for item in payload]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [VerificationRecord.model_validate(row) for row in reader]
    raise ValueError(f"unsupported manifest format: {path.suffix}")


def write_manifest_jsonl(records: Iterable[VerificationRecord], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [record.model_dump_json() for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_adresso_records(
    audio_root: Path,
    label_csv_path: Path,
    *,
    dataset_name: str = "adresso_2021",
    split: str = "test",
) -> list[VerificationRecord]:
    audio_root = audio_root.expanduser().resolve()
    label_csv_path = label_csv_path.expanduser().resolve()
    if not label_csv_path.exists():
        raise FileNotFoundError(f"ADReSSo label file not found: {label_csv_path}")

    with label_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    records: list[VerificationRecord] = []
    for row in rows:
        case_id = (row.get("ID") or row.get("id") or "").strip()
        if not case_id:
            continue
        matched_path = _find_audio_case(audio_root, case_id)
        if matched_path is None:
            continue
        label_value = row.get("Dx") or row.get("dx") or row.get("label") or ""
        records.append(
            VerificationRecord(
                case_id=case_id,
                dataset=dataset_name,
                split=split,
                label=normalize_label(label_value),
                media_path=str(matched_path),
                media_type="audio",
                language="en",
                metadata={"source": "ADReSSo 2021"},
            )
        )
    return records


def _find_audio_case(audio_root: Path, case_id: str) -> Path | None:
    candidates = sorted(audio_root.rglob(f"{case_id}.*"))
    for candidate in candidates:
        if candidate.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}:
            return candidate.resolve()
    return None
