"""Tests for doctor and care surface aggregation routes."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clinic.configs.settings import Settings
from backend.src.app.api.routes import build_api_router
from backend.src.app.models import (
    ClinicAssessment,
    ProviderMeta,
    ProviderTraceEntry,
)
from backend.src.app.services.assessment_service import ClinicAssessmentService
from backend.src.app.services.identity_service import FaceEvidence, IdentityLinkageService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="test",
        storage_dir=tmp_path,
        uploads_dir=tmp_path / "uploads",
        prepared_dir=tmp_path / "prepared",
        assessments_dir=tmp_path / "assessments",
        server_host="127.0.0.1",
        server_port=8000,
        server_reload=False,
        max_upload_mb=100,
        max_inline_video_mb=5,
        allow_mock_providers=False,
        default_provider="qwen_omni",
        fallback_order=("qwen_omni", "gemini", "fusion", "audio_only"),
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        qwen_omni_api_key=None,
        qwen_omni_base_url="https://example.com/qwen",
        qwen_omni_model="qwen3.5-omni-plus",
        gemini_api_key=None,
        gemini_base_url="https://example.com/gemini",
        gemini_model="gemini-3.1-pro-preview",
        openai_api_key=None,
        openai_base_url="https://example.com/openai",
        openai_fusion_model="gpt-4.1",
        openai_text_model="gpt-4.1",
        openai_transcription_model="gpt-4o-transcribe",
    )


def write_assessment(
    service: ClinicAssessmentService,
    *,
    assessment_id: str,
    patient_id: str,
    created_at: str,
    risk_score: float,
    risk_tier: str,
    screening_classification: str,
) -> ClinicAssessment:
    assessment = ClinicAssessment(
        assessment_id=assessment_id,
        patient_id=patient_id,
        language="en",
        created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(timezone.utc),
        visual_findings=[],
        body_findings=[],
        voice_findings=[],
        content_findings=[],
        speaker_structure="single_speaker",
        target_patient_presence="clear",
        target_patient_basis="visual_focus",
        detected_languages=["en"],
        risk_score=risk_score,
        risk_label="cognitive_risk" if risk_tier != "low" else "HC",
        risk_tier=risk_tier,
        screening_classification=screening_classification,
        screening_summary=f"{patient_id} summary",
        evidence_for_risk=[],
        evidence_against_risk=[],
        alternative_explanations=[],
        risk_factor_findings=[],
        subjective_assessment_findings=[],
        emotional_assessment_findings=[],
        risk_control_suggestions=["Repeat follow-up as needed."],
        visit_recommendation="Coordinate next review.",
        future_risk_trend_summary="Trend unavailable from a single clip.",
        reviewer_confidence=0.81,
        context_notes=[],
        quality_flags=[],
        session_usability="usable",
        provider_meta=ProviderMeta(
            final_provider="qwen_omni",
            model_name="qwen3-omni-flash",
            request_id=f"request-{assessment_id}",
            latency_ms=1200,
            raw_status="ok",
        ),
        provider_trace=[
            ProviderTraceEntry(
                provider="qwen_omni",
                attempt_order=1,
                status="success",
                latency_ms=1200,
            )
        ],
    )
    service.storage.save_assessment(assessment)
    return assessment


def make_face_evidence(track_id: str, descriptor: list[float]) -> FaceEvidence:
    norm = math.sqrt(sum(value * value for value in descriptor)) or 1.0
    return FaceEvidence(
        descriptor=[round(value / norm, 4) for value in descriptor],
        sample_count=4,
        detection_rate=0.86,
        average_face_area=0.22,
        dominant_track_id=track_id,
        method="test-face-descriptor",
    )


def patch_face_sequence(monkeypatch, *sequence: FaceEvidence) -> None:
    queue = list(sequence)

    def _fake_extract(self, media):
        if queue:
            return queue.pop(0)
        return sequence[-1] if sequence else FaceEvidence()

    monkeypatch.setattr(IdentityLinkageService, "_extract_face_evidence", _fake_extract)


def make_client(
    tmp_path: Path,
    *,
    seed_identity: bool = False,
    monkeypatch=None,
) -> TestClient:
    settings = make_settings(tmp_path)
    settings.assessments_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.prepared_dir.mkdir(parents=True, exist_ok=True)
    service = ClinicAssessmentService(settings)

    a_new = write_assessment(
        service,
        assessment_id="a-new",
        patient_id="patient-001",
        created_at="2026-04-07T08:10:00Z",
        risk_score=0.72,
        risk_tier="high",
        screening_classification="dementia",
    )
    a_old = write_assessment(
        service,
        assessment_id="a-old",
        patient_id="patient-002",
        created_at="2026-04-06T08:10:00Z",
        risk_score=0.28,
        risk_tier="low",
        screening_classification="healthy",
    )
    if seed_identity:
        patch_face_sequence(
            monkeypatch,
            make_face_evidence("patient-001-face", [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
            make_face_evidence("patient-002-face", [0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1]),
        )
        service.identity.link_assessment(a_new)
        service.identity.link_assessment(a_old)

    app = FastAPI()
    app.include_router(build_api_router(service), prefix="/api")
    return TestClient(app)


def test_doctor_dashboard_route_returns_aggregated_counts(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/api/doctor/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_assessments"] == 2
    assert payload["total_patients"] == 2
    assert payload["high_risk_assessments"] == 1
    assert payload["recent_assessments"][0]["assessment_id"] == "a-new"


def test_care_routes_return_known_patients_and_patient_dashboard(tmp_path: Path, monkeypatch) -> None:
    client = make_client(tmp_path, seed_identity=True, monkeypatch=monkeypatch)

    patients_response = client.get("/api/care/patients")
    dashboard_response = client.get("/api/care/dashboard/patient-001")

    assert patients_response.status_code == 200
    assert patients_response.json()["patients"] == [
        "demo-dementia-case",
        "demo-healthy-case",
        "patient-001",
        "patient-002",
    ]

    assert dashboard_response.status_code == 200
    payload = dashboard_response.json()
    assert payload["patient_id"] == "patient-001"
    assert payload["sessions_completed"] == 1
    assert payload["signal"] == "urgent"
    assert payload["risk_label"] == "HIGH RISK"
    assert payload["risk_score"] == 0.72
    assert payload["baseline_comparison"].startswith("Compared to your baseline (3 months ago)")
    assert payload["longitudinal_direction"] == "declining"
    assert payload["longitudinal_direction_label"] == "Declining"
    assert payload["chart_x_axis_label"] == "Date"
    assert payload["chart_y_axis_label"] == "Risk Score"
    assert len(payload["longitudinal_points"]) >= 4
    assert payload["top_reasons"] == [
        "Frequent pauses",
        "Reduced fluency",
        "Word-finding difficulty",
    ]
    assert payload["history"][0]["assessment_id"] == "a-new"


def test_care_demo_cases_are_hard_coded_for_reliable_presentations(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    dementia_response = client.get("/api/care/dashboard/demo-dementia-case")
    healthy_response = client.get("/api/care/dashboard/demo-healthy-case")

    assert dementia_response.status_code == 200
    dementia_payload = dementia_response.json()
    assert dementia_payload["risk_label"] == "HIGH RISK"
    assert dementia_payload["risk_score"] >= 0.7
    assert dementia_payload["baseline_comparison"].startswith("Compared to your baseline (3 months ago)")
    assert dementia_payload["longitudinal_direction"] == "declining"
    assert len(dementia_payload["longitudinal_points"]) >= 4
    assert dementia_payload["top_reasons"] == [
        "Frequent pauses",
        "Reduced fluency",
        "Word-finding difficulty",
    ]
    assert dementia_payload["recommendation"] == "Recommend further cognitive assessment."

    assert healthy_response.status_code == 200
    healthy_payload = healthy_response.json()
    assert healthy_payload["risk_label"] == "LOW RISK"
    assert healthy_payload["risk_score"] <= 0.35
    assert healthy_payload["baseline_comparison"].startswith("Compared to your baseline (3 months ago)")
    assert healthy_payload["longitudinal_direction"] == "stable"
    assert len(healthy_payload["longitudinal_points"]) >= 4
    assert healthy_payload["top_reasons"] == [
        "Normal pause duration",
        "Coherent narrative",
        "Typical vocabulary",
    ]
    assert healthy_payload["recommendation"] == "Continue routine monitoring."
