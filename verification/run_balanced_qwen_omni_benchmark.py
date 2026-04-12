"""Select and run a balanced multilingual qwen_omni verification benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.config import get_verification_settings
from verification.datasets import load_manifest, write_manifest_jsonl
from verification.media import prepare_audio_artifact
from verification.metrics import compute_metrics
from verification.models import (
    PreparedAudioArtifact,
    VerificationCaseResult,
    VerificationRecord,
    VerificationSummary,
)
from verification.qwen_audio_only import MAX_DATA_URL_AUDIO_BYTES
from verification.qwen_omni_audio import QwenOmniAudioVerifier


@dataclass(frozen=True)
class LanguageQuota:
    name: str
    manifests: tuple[str, ...]
    total: int

    @property
    def per_label(self) -> int:
        if self.total % 2 != 0:
            raise ValueError(f"{self.name} total must be even to keep 50/50 labels: {self.total}")
        return self.total // 2


def parse_args() -> argparse.Namespace:
    settings = get_verification_settings()
    parser = argparse.ArgumentParser(
        description="Build and run a balanced multilingual qwen_omni verification benchmark."
    )
    parser.add_argument(
        "--english-manifest",
        action="append",
        default=[
            str(settings.data_dir / "talkbank" / "corpora" / "english" / "pitt" / "manifest.jsonl"),
            str(settings.data_dir / "talkbank" / "corpora" / "english" / "lu" / "manifest.jsonl"),
        ],
        help="English manifest path. Repeat to combine multiple pools.",
    )
    parser.add_argument(
        "--chinese-manifest",
        action="append",
        default=[
            str(settings.data_dir / "talkbank" / "corpora" / "mandarin" / "chou" / "manifest.jsonl"),
        ],
        help="Chinese manifest path. Repeat to combine multiple pools.",
    )
    parser.add_argument(
        "--english-total",
        type=int,
        default=50,
        help="Total English cases to select. Must be even.",
    )
    parser.add_argument(
        "--chinese-total",
        type=int,
        default=30,
        help="Total Chinese cases to select. Must be even.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260410,
        help="Selection seed for reproducible sampling.",
    )
    parser.add_argument(
        "--exclude-case-id",
        action="append",
        default=[],
        help="Case id to exclude from sampling. Repeat to add more exclusions.",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        help="Text file with one case_id per line to exclude.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.results_dir / "qwen_omni_balanced_multilingual_seed20260410"),
        help="Directory for selected manifest, per-case outputs, summaries, and report files.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Only build the selected manifest and report preflight diagnostics without running inference.",
    )
    return parser.parse_args()


def load_case_exclusions(args: argparse.Namespace) -> set[str]:
    excluded = {case_id.strip() for case_id in args.exclude_case_id if case_id and case_id.strip()}
    for path_text in args.exclude_file:
        path = Path(path_text).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"exclude file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                excluded.add(clean)
    return excluded


def load_records_from_manifests(manifest_paths: list[str]) -> list[VerificationRecord]:
    records: list[VerificationRecord] = []
    seen_case_ids: set[str] = set()
    for manifest_text in manifest_paths:
        manifest_path = Path(manifest_text).expanduser().resolve()
        for record in load_manifest(manifest_path):
            if record.case_id in seen_case_ids:
                continue
            seen_case_ids.add(record.case_id)
            records.append(record)
    return records


def enrich_record_metadata(record: VerificationRecord, *, language_group: str, seed: int) -> VerificationRecord:
    metadata = dict(record.metadata)
    metadata["benchmark_language_group"] = language_group
    metadata["benchmark_seed"] = seed
    return record.model_copy(update={"metadata": metadata})


def preflight_record(
    record: VerificationRecord,
    *,
    settings,
) -> tuple[PreparedAudioArtifact | None, str | None]:
    try:
        prepared = prepare_audio_artifact(record, settings)
    except Exception as exc:  # noqa: BLE001
        return None, f"prepare_failed:{type(exc).__name__}"

    if prepared.size_bytes > MAX_DATA_URL_AUDIO_BYTES:
        return None, "prepared_audio_too_large"
    return prepared, None


def select_balanced_records(
    *,
    quotas: list[LanguageQuota],
    excluded_case_ids: set[str],
    seed: int,
) -> tuple[list[VerificationRecord], dict[str, PreparedAudioArtifact], dict[str, Any]]:
    settings = get_verification_settings()
    selected_records: list[VerificationRecord] = []
    prepared_cache: dict[str, PreparedAudioArtifact] = {}
    selection_summary: dict[str, Any] = {
        "seed": seed,
        "excluded_case_ids_count": len(excluded_case_ids),
        "by_language": {},
        "skipped_reasons": {},
    }
    skipped_reason_counter: Counter[str] = Counter()

    for quota in quotas:
        all_records = load_records_from_manifests(list(quota.manifests))
        all_records = [
            enrich_record_metadata(record, language_group=quota.name, seed=seed)
            for record in all_records
            if record.case_id not in excluded_case_ids
        ]

        language_info: dict[str, Any] = {
            "requested_total": quota.total,
            "requested_per_label": quota.per_label,
            "selected_case_ids": [],
            "pool_size": len(all_records),
            "label_pool_sizes": {},
        }

        for label in ("HC", "cognitive_risk"):
            candidates = [record for record in all_records if record.label == label]
            language_info["label_pool_sizes"][label] = len(candidates)

            shuffled = sorted(candidates, key=lambda record: record.case_id)
            random.Random(f"{seed}:{quota.name}:{label}").shuffle(shuffled)

            accepted_count = 0
            for record in shuffled:
                if record.case_id in prepared_cache:
                    prepared = prepared_cache[record.case_id]
                else:
                    prepared, skip_reason = preflight_record(record, settings=settings)
                    if skip_reason is not None:
                        skipped_reason_counter[skip_reason] += 1
                        continue
                    assert prepared is not None
                    prepared_cache[record.case_id] = prepared

                selected_records.append(record)
                language_info["selected_case_ids"].append(record.case_id)
                accepted_count += 1
                if accepted_count >= quota.per_label:
                    break

            if accepted_count < quota.per_label:
                raise RuntimeError(
                    f"not enough runnable {quota.name} {label} cases after exclusions and preflight: "
                    f"needed {quota.per_label}, found {accepted_count}"
                )

        selection_summary["by_language"][quota.name] = language_info

    selection_summary["skipped_reasons"] = dict(sorted(skipped_reason_counter.items()))
    return selected_records, prepared_cache, selection_summary


async def run_benchmark(
    *,
    records: list[VerificationRecord],
    prepared_cache: dict[str, PreparedAudioArtifact],
    output_dir: Path,
) -> VerificationSummary:
    verification_settings = get_verification_settings()
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    verifier = QwenOmniAudioVerifier()
    if not verifier.is_configured():
        raise RuntimeError("qwen_omni is not configured. Set QWEN_API_KEY or DASHSCOPE_API_KEY.")

    results: list[VerificationCaseResult] = []
    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        prepared_media_path: str | None = None
        case_output_path = cases_dir / f"{index:03d}__{record.case_id}.json"
        try:
            prepared_audio = prepared_cache.get(record.case_id) or prepare_audio_artifact(
                record,
                verification_settings,
            )
            prepared_media_path = prepared_audio.audio_path
            assessment, classification_request_id, asr_transcript, asr_request_id = await verifier.classify_audio(
                record=record,
                audio_path=prepared_audio.audio_path,
                mime_type=prepared_audio.mime_type,
            )
            result = VerificationCaseResult(
                case_id=record.case_id,
                dataset=record.dataset,
                split=record.split,
                ground_truth_label=record.label,
                predicted_label=assessment.get("risk_label"),
                predicted_screening_classification=assessment.get("screening_classification"),
                risk_score=assessment.get("risk_score"),
                reviewer_confidence=assessment.get("reviewer_confidence"),
                transcript=asr_transcript,
                patient_only_transcript=assessment.get("patient_only_transcript"),
                speaker_turn_summary=assessment.get("speaker_turn_summary") or [],
                patient_cue_summary=assessment.get("patient_cue_summary") or [],
                transcript_request_id=asr_request_id,
                classification_request_id=classification_request_id,
                transcript_model=verifier.asr_verifier.asr_model,
                classifier_model=verifier.model_name,
                audio_path=prepared_audio.audio_path,
                prepared_media_path=prepared_media_path,
                assessment=assessment,
                latency_ms=int((time.perf_counter() - started) * 1000),
                metadata=record.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            result = VerificationCaseResult(
                case_id=record.case_id,
                dataset=record.dataset,
                split=record.split,
                ground_truth_label=record.label,
                transcript_model=verifier.asr_verifier.asr_model,
                classifier_model=verifier.model_name,
                audio_path=prepared_media_path or (record.media_path if record.media_type == "audio" else None),
                prepared_media_path=prepared_media_path,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
                metadata=record.metadata,
            )

        case_output_path.write_text(
            result.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        results.append(result)

    metrics = compute_metrics(results)
    dataset_names = sorted({result.dataset for result in results})
    summary = VerificationSummary(
        dataset=dataset_names[0] if len(dataset_names) == 1 else "mixed",
        manifest_path=str((output_dir / "selected_manifest.jsonl").resolve()),
        results_path=str(output_dir),
        metrics=metrics,
        cases=results,
    )
    return summary


def build_case_report_rows(summary: VerificationSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(summary.cases, start=1):
        metadata = case.metadata or {}
        rows.append(
            {
                "index": index,
                "language_group": metadata.get("benchmark_language_group"),
                "case_id": case.case_id,
                "dataset": case.dataset,
                "ground_truth_label": case.ground_truth_label,
                "predicted_label": case.predicted_label,
                "predicted_screening_classification": case.predicted_screening_classification,
                "risk_score": case.risk_score,
                "reviewer_confidence": case.reviewer_confidence,
                "source_url": metadata.get("source_url"),
                "source_relative_path": metadata.get("source_relative_path"),
                "payload": case.assessment,
                "error": case.error,
            }
        )
    return rows


def build_language_metrics(summary: VerificationSummary) -> dict[str, Any]:
    by_language: dict[str, Any] = {}
    grouped: dict[str, list[VerificationCaseResult]] = {}
    for case in summary.cases:
        language_group = str((case.metadata or {}).get("benchmark_language_group") or "unknown")
        grouped.setdefault(language_group, []).append(case)

    for language_group, results in grouped.items():
        by_language[language_group] = compute_metrics(results).model_dump(mode="json")
    return by_language


def write_benchmark_report(
    *,
    output_dir: Path,
    summary: VerificationSummary,
    selection_summary: dict[str, Any],
    requested_english_total: int,
    requested_chinese_total: int,
) -> None:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": {
            "english_total": requested_english_total,
            "chinese_total": requested_chinese_total,
            "combined_total": requested_english_total + requested_chinese_total,
            "balanced_labels": True,
        },
        "selection_summary": selection_summary,
        "overall_metrics": summary.metrics.model_dump(mode="json"),
        "metrics_by_language": build_language_metrics(summary),
        "cases": build_case_report_rows(summary),
    }

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Balanced Qwen Omni Benchmark",
        "",
        f"- English cases: {requested_english_total}",
        f"- Chinese cases: {requested_chinese_total}",
        f"- Combined cases: {requested_english_total + requested_chinese_total}",
        "",
        "## Overall Metrics",
        "",
        f"- Accuracy: {summary.metrics.accuracy}",
        f"- Precision: {summary.metrics.precision}",
        f"- Recall: {summary.metrics.recall}",
        f"- Specificity: {summary.metrics.specificity}",
        f"- F1: {summary.metrics.f1}",
        f"- Confusion Matrix: {json.dumps(summary.metrics.confusion_matrix, ensure_ascii=False)}",
        "",
        "## Metrics By Language",
        "",
    ]

    for language_group, metrics in report["metrics_by_language"].items():
        md_lines.extend(
            [
                f"### {language_group.title()}",
                "",
                f"- Accuracy: {metrics.get('accuracy')}",
                f"- Precision: {metrics.get('precision')}",
                f"- Recall: {metrics.get('recall')}",
                f"- Specificity: {metrics.get('specificity')}",
                f"- F1: {metrics.get('f1')}",
                f"- Confusion Matrix: {json.dumps(metrics.get('confusion_matrix', {}), ensure_ascii=False)}",
                "",
            ]
        )

    md_lines.extend(
        [
            "## Case Table",
            "",
            "| # | Group | Case ID | Truth | Pred | URL |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["cases"]:
        url = row["source_url"] or ""
        md_lines.append(
            f"| {row['index']} | {row['language_group']} | {row['case_id']} | "
            f"{row['ground_truth_label']} | {row['predicted_label'] or row['error'] or ''} | {url} |"
        )

    (output_dir / "benchmark_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


async def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quotas = [
        LanguageQuota(
            name="english",
            manifests=tuple(args.english_manifest),
            total=args.english_total,
        ),
        LanguageQuota(
            name="chinese",
            manifests=tuple(args.chinese_manifest),
            total=args.chinese_total,
        ),
    ]
    excluded_case_ids = load_case_exclusions(args)
    selected_records, prepared_cache, selection_summary = select_balanced_records(
        quotas=quotas,
        excluded_case_ids=excluded_case_ids,
        seed=args.seed,
    )

    selected_manifest_path = output_dir / "selected_manifest.jsonl"
    write_manifest_jsonl(selected_records, selected_manifest_path)
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.selection_only:
        print(
            json.dumps(
                {
                    "selected_manifest": str(selected_manifest_path),
                    "selection_summary": selection_summary,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    summary = await run_benchmark(
        records=selected_records,
        prepared_cache=prepared_cache,
        output_dir=output_dir,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    write_benchmark_report(
        output_dir=output_dir,
        summary=summary,
        selection_summary=selection_summary,
        requested_english_total=args.english_total,
        requested_chinese_total=args.chinese_total,
    )
    print(summary.model_dump_json(indent=2, exclude_none=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
