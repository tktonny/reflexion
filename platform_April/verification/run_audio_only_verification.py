"""Run batch audio-only verification with DashScope ASR plus Qwen 3.5."""

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
from verification.models import VerificationCaseResult, VerificationSummary
from verification.qwen_audio_only import QwenAudioOnlyVerifier


def parse_args() -> argparse.Namespace:
    settings = get_verification_settings()
    parser = argparse.ArgumentParser(description="Batch-run the voice-only verification pipeline.")
    parser.add_argument(
        "--manifest",
        default=str(settings.data_dir / "smoke" / "manifest.jsonl"),
        help="Manifest path in jsonl/json/csv format.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.results_dir / "latest"),
        help="Directory for per-case outputs and summary.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases to run.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_verification_settings()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"manifest is empty: {manifest_path}")

    verifier = QwenAudioOnlyVerifier(settings)
    if not verifier.is_configured():
        raise RuntimeError("DashScope is not configured. Set DASHSCOPE_API_KEY or QWEN_API_KEY.")

    results: list[VerificationCaseResult] = []
    for record in records:
        started = time.perf_counter()
        case_output_path = cases_dir / f"{record.case_id}.json"
        try:
            prepared_audio = prepare_audio_artifact(record, settings)
            transcript, transcript_request_id = await verifier.transcribe_audio(
                audio_path=prepared_audio.audio_path,
                mime_type=prepared_audio.mime_type,
                language=record.language,
            )
            assessment, classification_request_id = await verifier.classify_transcript(
                patient_id=record.case_id,
                language=record.language,
                transcript=transcript,
            )
            predicted_label = assessment.get("risk_label")
            result = VerificationCaseResult(
                case_id=record.case_id,
                dataset=record.dataset,
                split=record.split,
                ground_truth_label=record.label,
                predicted_label=predicted_label,
                predicted_screening_classification=assessment.get("screening_classification"),
                risk_score=assessment.get("risk_score"),
                reviewer_confidence=assessment.get("reviewer_confidence"),
                transcript=transcript,
                transcript_request_id=transcript_request_id,
                classification_request_id=classification_request_id,
                transcript_model=settings.qwen_asr_model,
                classifier_model=settings.qwen_text_model,
                audio_path=prepared_audio.audio_path,
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
                transcript_model=settings.qwen_asr_model,
                classifier_model=settings.qwen_text_model,
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
