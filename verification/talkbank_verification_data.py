"""Prepare selected TalkBank corpora for verification manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verification.datasets import write_manifest_jsonl
from verification.models import VerificationRecord
from verification.talkbank import TalkBankClient, write_json


DEFAULT_VERIFICATION_CORPORA = (
    "English/Lu",
    "English/Pitt",
    "Korean/Kang",
    "Mandarin/Chou",
    "Spanish/Ivanova",
)
LANGUAGE_CODE_MAP = {
    "English": "en",
    "German": "de",
    "Greek": "el",
    "Korean": "ko",
    "Mandarin": "zh",
    "Spanish": "es",
    "Taiwanese": "nan",
}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
FORMAT_DIRECTORY_SLUGS = {"0wav", "wav", "0mp3", "mp3", "flac", "m4a", "aac", "ogg", "opus"}
EXTENSION_PREFERENCE = {
    ".mp3": 0,
    ".m4a": 1,
    ".aac": 2,
    ".ogg": 3,
    ".opus": 4,
    ".flac": 5,
    ".wav": 6,
}


@dataclass(frozen=True)
class CorpusTarget:
    language: str
    corpus: str
    label_groups: list[dict[str, Any]]
    audio_file_count: int

    @property
    def spec(self) -> str:
        return f"{self.language}/{self.corpus}"


def load_selected_targets(
    catalog: dict[str, Any],
    corpus_specs: list[str] | tuple[str, ...],
) -> list[CorpusTarget]:
    candidates = {
        (item["language"], item["corpus"]): item
        for item in catalog.get("verification_candidates", [])
    }
    targets: list[CorpusTarget] = []
    for spec in corpus_specs:
        language, corpus = parse_corpus_spec(spec)
        payload = candidates.get((language, corpus))
        if payload is None:
            raise KeyError(f"TalkBank verification candidate not found in catalog: {spec}")
        targets.append(
            CorpusTarget(
                language=language,
                corpus=corpus,
                label_groups=list(payload.get("label_groups", [])),
                audio_file_count=int(payload.get("audio_file_count", 0)),
            )
        )
    return targets


def prepare_selected_corpora(
    client: TalkBankClient,
    *,
    settings: Any,
    targets: list[CorpusTarget],
    overwrite: bool = False,
) -> dict[str, Any]:
    all_records: list[VerificationRecord] = []
    corpus_reports: list[dict[str, Any]] = []

    for target in targets:
        corpus_root = (
            settings.data_dir
            / "talkbank"
            / "corpora"
            / slugify(target.language)
            / slugify(target.corpus)
        )
        corpus_records: list[VerificationRecord] = []
        group_reports: list[dict[str, Any]] = []
        for group in target.label_groups:
            group_root_url = media_url(settings, group["relative_path"])
            group_name = str(group["group_name"])
            group_local_root = corpus_root / group_name
            audio_entries = select_preferred_remote_entries(
                list_remote_audio_files(client, group_root_url),
                dataset_name=build_dataset_name(target.language, target.corpus),
                group_name=group_name,
            )
            downloaded_count = 0
            reused_count = 0
            for entry in audio_entries:
                relative_path = Path(*entry["relative_parts"])
                dest_path = group_local_root / relative_path
                existed_before = dest_path.exists()
                client.download_to_path(entry["url"], dest_path, overwrite=overwrite)
                if existed_before and not overwrite:
                    reused_count += 1
                else:
                    downloaded_count += 1

                source_relative_path = Path(group_name) / relative_path
                record = VerificationRecord(
                    case_id=build_case_id(target.language, target.corpus, source_relative_path),
                    dataset=build_dataset_name(target.language, target.corpus),
                    split="full",
                    label=group["mapped_label"],
                    media_path=str(dest_path.resolve()),
                    media_type="audio",
                    language=LANGUAGE_CODE_MAP.get(target.language, "en"),
                    metadata={
                        "source_language": target.language,
                        "source_corpus": target.corpus,
                        "source_group": group_name,
                        "source_relative_path": source_relative_path.as_posix(),
                        "source_url": entry["url"],
                        "expected_group_audio_count": group["audio_file_count"],
                    },
                )
                corpus_records.append(record)

            group_reports.append(
                {
                    "group_name": group_name,
                    "mapped_label": group["mapped_label"],
                    "expected_audio_file_count": group["audio_file_count"],
                    "downloaded_audio_file_count": len(audio_entries),
                    "downloaded_count": downloaded_count,
                    "reused_count": reused_count,
                    "remote_root_path": group["relative_path"],
                    "local_root": str(group_local_root.resolve()),
                }
            )

        manifest_path = corpus_root / "manifest.jsonl"
        deduped_records = dedupe_records(corpus_records)
        write_manifest_jsonl(deduped_records, manifest_path)
        corpus_report = {
            "language": target.language,
            "corpus": target.corpus,
            "spec": target.spec,
            "dataset": build_dataset_name(target.language, target.corpus),
            "expected_audio_file_count": target.audio_file_count,
            "record_count": len(deduped_records),
            "downloaded_file_count": len(corpus_records),
            "manifest_path": str(manifest_path.resolve()),
            "local_root": str(corpus_root.resolve()),
            "groups": group_reports,
        }
        corpus_reports.append(corpus_report)
        all_records.extend(deduped_records)

    combined_root = settings.data_dir / "talkbank" / "verification_ready"
    combined_manifest_path = combined_root / "selected_corpora_manifest.jsonl"
    write_manifest_jsonl(all_records, combined_manifest_path)
    index = {
        "selected_corpora": [target.spec for target in targets],
        "combined_manifest_path": str(combined_manifest_path.resolve()),
        "combined_record_count": len(all_records),
        "corpora": corpus_reports,
    }
    write_json(index, combined_root / "index.json")
    return index


def list_remote_audio_files(
    client: TalkBankClient,
    root_url: str,
    *,
    _relative_parts: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    entries = client.list_directory(root_url)
    audio_entries: list[dict[str, Any]] = []
    for entry in entries:
        suffix = Path(entry.name).suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            audio_entries.append(
                {
                    "relative_parts": _relative_parts + (entry.name,),
                    "url": entry.url,
                }
            )
            continue
        if suffix:
            continue
        audio_entries.extend(
            list_remote_audio_files(
                client,
                entry.url,
                _relative_parts=_relative_parts + (entry.name,),
            )
        )
    return audio_entries


def parse_corpus_spec(spec: str) -> tuple[str, str]:
    cleaned = spec.strip().strip("/")
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"Corpus spec must look like Language/Corpus, got: {spec}")
    return parts[0], parts[1]


def slugify(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    chars = [ch if ch.isalnum() or ch == "_" else "_" for ch in value]
    collapsed = "".join(chars)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_")


def build_dataset_name(language: str, corpus: str) -> str:
    return f"talkbank_{slugify(language)}_{slugify(corpus)}"


def build_case_id(language: str, corpus: str, relative_path: Path) -> str:
    parts = [slugify(language), slugify(corpus)]
    for part in relative_path.parts[:-1]:
        parts.append(slugify(part))
    parts.append(slugify(relative_path.stem))
    return "__".join(part for part in parts if part)


def media_url(settings: Any, relative_path: str) -> str:
    return f"{settings.talkbank_media_base_url}/{relative_path.lstrip('/')}"


def dedupe_records(records: list[VerificationRecord]) -> list[VerificationRecord]:
    preferred: dict[tuple[str, ...], VerificationRecord] = {}
    for record in records:
        relative_path = Path(str(record.metadata.get("source_relative_path", record.media_path)))
        key = build_logical_source_key(record.dataset, relative_path)
        current = preferred.get(key)
        if current is None or compare_record_preference(record, current) < 0:
            preferred[key] = record
    return sorted(preferred.values(), key=lambda item: item.case_id)


def build_logical_source_key(dataset: str, relative_path: Path) -> tuple[str, ...]:
    normalized_parts = [slugify(dataset)]
    for part in relative_path.parts[:-1]:
        part_slug = slugify(part)
        if part_slug in FORMAT_DIRECTORY_SLUGS:
            continue
        normalized_parts.append(part_slug)
    normalized_parts.append(slugify(relative_path.stem))
    return tuple(normalized_parts)


def compare_record_preference(left: VerificationRecord, right: VerificationRecord) -> int:
    left_ext = Path(left.media_path).suffix.lower()
    right_ext = Path(right.media_path).suffix.lower()
    left_rank = EXTENSION_PREFERENCE.get(left_ext, 99)
    right_rank = EXTENSION_PREFERENCE.get(right_ext, 99)
    if left_rank != right_rank:
        return -1 if left_rank < right_rank else 1
    return -1 if left.media_path < right.media_path else (1 if left.media_path > right.media_path else 0)


def select_preferred_remote_entries(
    entries: list[dict[str, Any]],
    *,
    dataset_name: str,
    group_name: str,
) -> list[dict[str, Any]]:
    preferred: dict[tuple[str, ...], dict[str, Any]] = {}
    for entry in entries:
        relative_path = Path(group_name) / Path(*entry["relative_parts"])
        key = build_logical_source_key(dataset_name, relative_path)
        current = preferred.get(key)
        if current is None or compare_remote_entry_preference(entry, current) < 0:
            preferred[key] = entry
    return sorted(preferred.values(), key=lambda item: "/".join(item["relative_parts"]))


def compare_remote_entry_preference(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_ext = Path(*left["relative_parts"]).suffix.lower()
    right_ext = Path(*right["relative_parts"]).suffix.lower()
    left_rank = EXTENSION_PREFERENCE.get(left_ext, 99)
    right_rank = EXTENSION_PREFERENCE.get(right_ext, 99)
    if left_rank != right_rank:
        return -1 if left_rank < right_rank else 1
    left_key = "/".join(left["relative_parts"])
    right_key = "/".join(right["relative_parts"])
    return -1 if left_key < right_key else (1 if left_key > right_key else 0)
