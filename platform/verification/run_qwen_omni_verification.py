"""Run batch qwen_omni verification on native audio manifests."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.config import get_verification_settings
from verification.datasets import load_manifest
from verification.media import prepare_audio_artifact
from verification.metrics import compute_metrics
from verification.models import VerificationCaseResult, VerificationRecord, VerificationSummary
from verification.qwen_omni_audio import QwenOmniAudioVerifier


def parse_args() -> argparse.Namespace:
    settings = get_verification_settings()
    parser = argparse.ArgumentParser(
        description="Batch-run qwen_omni verification on verification manifests."
    )
    parser.add_argument(
        "--manifest",
        default=str(settings.data_dir / "smoke" / "manifest.jsonl"),
        help="Manifest path in jsonl/json/csv format.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.results_dir / "qwen_omni_latest"),
        help="Directory for per-case outputs and summary.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of selected cases to run.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Exact case_id to run. Repeat to run multiple specific cases.",
    )
    parser.add_argument(
        "--label",
        action="append",
        choices=("HC", "cognitive_risk"),
        default=None,
        help="Ground-truth label to include. Repeat to include both labels.",
    )
    parser.add_argument(
        "--first-per-label",
        action="store_true",
        help="After label filtering, keep only the first manifest match for each requested label.",
    )
    return parser.parse_args()


def select_records(
    records: list[VerificationRecord],
    *,
    case_ids: list[str] | None = None,
    labels: list[str] | None = None,
    first_per_label: bool = False,
    limit: int | None = None,
) -> list[VerificationRecord]:
    selected = list(records)

    if case_ids:
        by_case_id = {record.case_id: record for record in records}
        missing_case_ids = [case_id for case_id in case_ids if case_id not in by_case_id]
        if missing_case_ids:
            missing_preview = ", ".join(missing_case_ids[:5])
            raise ValueError(f"unknown case_id(s): {missing_preview}")
        selected = [by_case_id[case_id] for case_id in case_ids]

    if labels:
        label_set = set(labels)
        selected = [record for record in selected if record.label in label_set]

    if first_per_label:
        if not labels:
            raise ValueError("--first-per-label requires at least one --label")
        deduped_labels = list(dict.fromkeys(labels))
        first_matches: list[VerificationRecord] = []
        missing_labels: list[str] = []
        for label in deduped_labels:
            match = next((record for record in selected if record.label == label), None)
            if match is None:
                missing_labels.append(label)
                continue
            first_matches.append(match)
        if missing_labels:
            missing_preview = ", ".join(missing_labels)
            raise ValueError(f"no manifest records found for label(s): {missing_preview}")
        selected = first_matches

    if limit is not None:
        selected = selected[:limit]
    return selected


async def main() -> int:
    args = parse_args()
    verification_settings = get_verification_settings()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_path)
    records = select_records(
        records,
        case_ids=args.case_id,
        labels=args.label,
        first_per_label=args.first_per_label,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError(f"manifest selection is empty: {manifest_path}")

    verifier = QwenOmniAudioVerifier()
    if not verifier.is_configured():
        raise RuntimeError("qwen_omni is not configured. Set QWEN_API_KEY or DASHSCOPE_API_KEY.")
    results: list[VerificationCaseResult] = []

    for record in records:
        started = time.perf_counter()
        prepared_media_path: str | None = None
        case_output_path = cases_dir / f"{record.case_id}.json"
        try:
            prepared_audio = prepare_audio_artifact(record, verification_settings)
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
        manifest_path=str(manifest_path),
        results_path=str(output_dir),
        metrics=metrics,
        cases=results,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    print(summary.model_dump_json(indent=2, exclude_none=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
