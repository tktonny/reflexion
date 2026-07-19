"""Tests for identity linkage and identity-gated longitudinal tracking."""

from __future__ import annotations

from dataclasses import replace
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clinic.configs.settings import Settings
from clinic.database.storage import LocalStorage
from backend.src.app.api.routes import build_api_router
from backend.src.app.models import ClinicAssessment, ProviderMeta, ProviderTraceEntry
from backend.src.app.models.identity import IdentityProfile
from backend.src.app.services.assessment_service import ClinicAssessmentService
from backend.src.app.services.identity_service import FaceEvidence, IdentityLinkageService
from backend.src.app.services.longitudinal_service import LongitudinalTrackingService
from backend.src.app.services import patient_memory as patient_memory_service


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


def make_assessment(
    *,
    assessment_id: str,
    patient_id: str,
    created_at: datetime,
    risk_score: float,
    screening_classification: str,
    risk_tier: str,
    target_patient_presence: str | None = "clear",
    speaker_structure: str | None = "single_speaker",
) -> ClinicAssessment:
    return ClinicAssessment(
        assessment_id=assessment_id,
        patient_id=patient_id,
        language="en",
        created_at=created_at,
        visual_findings=[],
        body_findings=[],
        voice_findings=[],
        content_findings=[],
        speaker_structure=speaker_structure,
        target_patient_presence=target_patient_presence,
        target_patient_basis="visual_focus" if target_patient_presence else None,
        detected_languages=["en"],
        language_confidence=0.93,
        risk_score=risk_score,
        risk_label="HC" if risk_score <= 0.35 else "cognitive_risk",
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
        reviewer_confidence=0.84,
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


def make_session_record(
    *,
    session_id: str,
    patient_name: str,
    patient_turns: int,
    utterance_count: int,
    face_detection_rate: float | None,
    average_face_area: float | None,
    motion_intensity: float,
    transcript: list[str],
    flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "sessionId": session_id,
        "productMode": "clinic",
        "patient": {
            "patientId": "declared-patient",
            "primaryLanguage": "en",
        },
        "derivedFeatures": {
            "speech": {
                "utteranceCount": utterance_count,
                "speechSeconds": 26.0,
                "averageTurnSeconds": 6.5,
                "transcriptTurns": [
                    {
                        "role": "patient",
                        "stage": "orientation",
                        "text": f"My name is {patient_name}. {transcript[0]}",
                    },
                    *[
                        {
                            "role": "patient",
                            "stage": f"stage-{index}",
                            "text": text,
                        }
                        for index, text in enumerate(transcript[1:], start=1)
                    ],
                ],
            },
            "task": {
                "completedStages": ["orientation", "recent_story", "daily_function", "delayed_recall"],
                "patientTurns": patient_turns,
            },
            "facial": {
                "faceDetectionRate": face_detection_rate,
                "averageFaceArea": average_face_area,
            },
            "interactionTiming": {
                "motionIntensity": motion_intensity,
                "meanBrightness": 0.54,
            },
        },
        "qualityControl": {
            "usability": "usable",
            "audioQualityScore": 0.82,
            "videoQualityScore": 0.80,
            "flags": flags or [],
        },
    }


def make_face_evidence(
    track_id: str,
    descriptor: list[float],
    *,
    sample_count: int = 4,
    detection_rate: float = 0.84,
    average_face_area: float = 0.22,
) -> FaceEvidence:
    norm = math.sqrt(sum(value * value for value in descriptor)) or 1.0
    normalized = [round(value / norm, 4) for value in descriptor]
    return FaceEvidence(
        descriptor=normalized,
        sample_count=sample_count,
        detection_rate=detection_rate,
        average_face_area=average_face_area,
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


def patch_frame_face_sequence(monkeypatch, *sequence: FaceEvidence) -> None:
    queue = list(sequence)

    def _fake_extract_from_frames(self, frames):
        del frames
        if queue:
            return queue.pop(0)
        return sequence[-1] if sequence else FaceEvidence()

    monkeypatch.setattr(IdentityLinkageService, "_extract_face_evidence_from_frames", _fake_extract_from_frames)


def test_identity_service_enrolls_and_confirms_same_patient_across_sessions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    mary_face = make_face_evidence("mary-face", [1.0, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
    patch_face_sequence(monkeypatch, mary_face, mary_face)

    first = make_assessment(
        assessment_id="visit-1",
        patient_id="patient-001",
        created_at=datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc),
        risk_score=0.31,
        risk_tier="low",
        screening_classification="healthy",
    )
    second = make_assessment(
        assessment_id="visit-2",
        patient_id="patient-001",
        created_at=datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
        risk_score=0.36,
        risk_tier="medium",
        screening_classification="needs_observation",
    )

    first_link = identity.link_assessment(
        first,
        session_record=make_session_record(
            session_id="session-1",
            patient_name="Mary Chen",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.86,
            average_face_area=0.24,
            motion_intensity=0.31,
            transcript=[
                "I am at home today.",
                "This morning I made tea and watered the plants.",
                "I manage my own medicine box.",
            ],
        ),
    )
    second_link = identity.link_assessment(
        second,
        session_record=make_session_record(
            session_id="session-2",
            patient_name="Mary Chen",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.83,
            average_face_area=0.23,
            motion_intensity=0.29,
            transcript=[
                "I am speaking from home again today.",
                "This morning I made breakfast and read the newspaper.",
                "I still manage my own appointments and medicine reminders.",
            ],
        ),
    )

    profile = identity.load_profile("patient-001")

    assert first_link.timeline_inclusion == "include"
    assert second_link.timeline_inclusion == "include"
    assert second_link.linkage_verdict in {"confirmed", "probable"}
    assert (second_link.face_match_confidence or 0) >= 0.95
    assert first_link.face_sample_count >= 2
    assert profile.sessions_included == 2
    assert profile.enrollment_assessment_id == "visit-1"
    assert profile.canonical_face_embedding


def test_longitudinal_skips_session_when_identity_is_ambiguous(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    longitudinal = LongitudinalTrackingService(storage)
    david_face = make_face_evidence("david-face", [0.2, 1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.0])
    caregiver_face = make_face_evidence("caregiver-face", [-0.2, -1.0, -0.1, 0.0, -0.05, 0.0, 0.0, 0.0])
    patch_face_sequence(monkeypatch, david_face, caregiver_face)

    included_assessment = make_assessment(
        assessment_id="visit-clear",
        patient_id="patient-002",
        created_at=datetime(2026, 1, 10, 8, 0, tzinfo=timezone.utc),
        risk_score=0.29,
        risk_tier="low",
        screening_classification="healthy",
    )
    excluded_assessment = make_assessment(
        assessment_id="visit-ambiguous",
        patient_id="patient-002",
        created_at=datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
        risk_score=0.75,
        risk_tier="high",
        screening_classification="dementia",
        target_patient_presence="absent",
        speaker_structure="multi_speaker",
    )

    included_link = identity.link_assessment(
        included_assessment,
        session_record=make_session_record(
            session_id="clear-session",
            patient_name="David Wong",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.81,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "I cooked noodles this morning and cleaned the kitchen.",
                "I manage my calendar myself.",
            ],
        ),
    )
    excluded_link = identity.link_assessment(
        excluded_assessment,
        session_record=make_session_record(
            session_id="ambiguous-session",
            patient_name="David Wong",
            patient_turns=1,
            utterance_count=5,
            face_detection_rate=0.05,
            average_face_area=0.04,
            motion_intensity=0.58,
            transcript=[
                "My daughter is helping answer this.",
                "She usually explains my medicine routine for me.",
            ],
            flags=["face_visibility_low"],
        ),
    )

    longitudinal.record_assessment(
        included_assessment,
        session_record=make_session_record(
            session_id="clear-session",
            patient_name="David Wong",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.81,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "I cooked noodles this morning and cleaned the kitchen.",
                "I manage my calendar myself.",
            ],
        ),
        identity_link=included_link,
    )
    profile = longitudinal.record_assessment(
        excluded_assessment,
        session_record=make_session_record(
            session_id="ambiguous-session",
            patient_name="David Wong",
            patient_turns=1,
            utterance_count=5,
            face_detection_rate=0.05,
            average_face_area=0.04,
            motion_intensity=0.58,
            transcript=[
                "My daughter is helping answer this.",
                "She usually explains my medicine routine for me.",
            ],
            flags=["face_visibility_low"],
        ),
        identity_link=excluded_link,
    )

    assert excluded_link.timeline_inclusion == "exclude"
    assert excluded_link.linkage_verdict == "mismatch"
    assert excluded_link.face_match_confidence is not None
    assert excluded_link.face_match_confidence < IdentityLinkageService.FACE_MATCH_EXCLUDE_THRESHOLD
    assert profile.total_sessions_seen == 2
    assert profile.sessions_included == 1
    assert profile.sessions_excluded == 1
    assert len(profile.trend_points) >= 4
    assert profile.gating_summary.startswith("Longitudinal gating:")
    assert profile.latest_risk_score == 0.29


def test_identity_service_holds_session_for_manual_review_when_face_is_missing_after_enrollment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)

    patch_face_sequence(
        monkeypatch,
        make_face_evidence("enrolled-face", [0.0, 0.2, 1.0, 0.1, 0.0, 0.0, 0.1, 0.0]),
        FaceEvidence(
            sample_count=0,
            detection_rate=0.0,
            quality_flags=["face_identity_no_detected_face"],
        ),
    )

    enrolled = make_assessment(
        assessment_id="visit-enrolled",
        patient_id="patient-003",
        created_at=datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc),
        risk_score=0.34,
        risk_tier="low",
        screening_classification="healthy",
    )
    follow_up = make_assessment(
        assessment_id="visit-no-face",
        patient_id="patient-003",
        created_at=datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
        risk_score=0.41,
        risk_tier="medium",
        screening_classification="needs_observation",
    )

    enrolled_link = identity.link_assessment(
        enrolled,
        session_record=make_session_record(
            session_id="session-enrolled",
            patient_name="Helen Zhu",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.82,
            average_face_area=0.22,
            motion_intensity=0.29,
            transcript=[
                "I am at home today.",
                "I made breakfast and read the calendar.",
                "I can manage my reminders myself.",
            ],
        ),
    )
    no_face_link = identity.link_assessment(
        follow_up,
        session_record=make_session_record(
            session_id="session-no-face",
            patient_name="Helen Zhu",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.05,
            average_face_area=0.03,
            motion_intensity=0.41,
            transcript=[
                "The room is darker today.",
                "I am still answering most of the questions myself.",
                "The video is not very clear.",
            ],
            flags=["face_visibility_low"],
        ),
    )
    profile = identity.load_profile("patient-003")

    assert enrolled_link.timeline_inclusion == "include"
    assert no_face_link.timeline_inclusion == "manual-review"
    assert no_face_link.linkage_verdict == "no-face"
    assert profile.sessions_included == 1
    assert profile.sessions_manual_review == 1
    assert profile.sessions_excluded == 0


def test_identity_service_persists_preferred_name_and_memory_from_session_transcript(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    patch_face_sequence(
        monkeypatch,
        make_face_evidence("memory-face", [0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    assessment = make_assessment(
        assessment_id="visit-memory",
        patient_id="patient-memory",
        created_at=datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc),
        risk_score=0.34,
        risk_tier="low",
        screening_classification="healthy",
    )

    identity.link_assessment(
        assessment,
        session_record=make_session_record(
            session_id="session-memory",
            patient_name="Grace Lin",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.82,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "This morning I had breakfast and read the news.",
                "I set reminders on my phone for medicine and appointments.",
            ],
        ),
    )

    profile = identity.load_profile("patient-memory")

    assert profile.preferred_name == "Grace Lin"
    assert profile.memory[0] == "Preferred name: Grace Lin."
    assert any(item.startswith("Recent activity mentioned:") for item in profile.memory)
    assert any(item.startswith("Routine/support detail:") for item in profile.memory)


def test_identity_service_accumulates_memory_across_sessions(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    repeated_face = make_face_evidence("memory-face", [0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    patch_face_sequence(monkeypatch, repeated_face, repeated_face)

    first_assessment = make_assessment(
        assessment_id="visit-memory-1",
        patient_id="patient-memory",
        created_at=datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc),
        risk_score=0.34,
        risk_tier="low",
        screening_classification="healthy",
    )
    second_assessment = make_assessment(
        assessment_id="visit-memory-2",
        patient_id="patient-memory",
        created_at=datetime(2026, 4, 15, 8, 0, tzinfo=timezone.utc),
        risk_score=0.37,
        risk_tier="medium",
        screening_classification="needs_observation",
    )

    identity.link_assessment(
        first_assessment,
        session_record=make_session_record(
            session_id="session-memory-1",
            patient_name="Grace Lin",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.82,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "This morning I had breakfast and read the news.",
                "I set reminders on my phone for medicine and appointments.",
            ],
        ),
    )
    identity.link_assessment(
        second_assessment,
        session_record=make_session_record(
            session_id="session-memory-2",
            patient_name="Grace Lin",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.84,
            average_face_area=0.23,
            motion_intensity=0.27,
            transcript=[
                "I am at home again today.",
                "This afternoon I watered the garden and called my daughter.",
                "My son helps me double-check the family calendar on weekends.",
            ],
        ),
    )

    profile = identity.load_profile("patient-memory")

    assert profile.preferred_name == "Grace Lin"
    assert profile.memory[0] == "Preferred name: Grace Lin."
    assert profile.memory[1] == "Recent activity mentioned: This afternoon I watered the garden and called my daughter."
    assert profile.memory[2] == "Routine/support detail: My son helps me double-check the family calendar on weekends."
    assert "Recent activity mentioned: This morning I had breakfast and read the news." in profile.memory
    assert "Routine/support detail: I set reminders on my phone for medicine and appointments." in profile.memory
    assert len(profile.memory) == 5


def test_identity_service_uses_llm_patient_memory_when_configured(tmp_path: Path, monkeypatch) -> None:
    settings = replace(make_settings(tmp_path), patient_memory_api_key="test-key")
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    patch_face_sequence(
        monkeypatch,
        make_face_evidence("llm-memory-face", [0.2, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    observed: dict[str, str] = {}

    def _fake_request_patient_memory_payload(*, transcript: str, settings) -> dict[str, object]:
        observed["transcript"] = transcript
        observed["model"] = settings.patient_memory_model
        return {
            "preferred_name": "Grace Lin",
            "recent_activity": "Had breakfast and read the news with her sister",
            "routine_support": "Uses phone reminders for medicine and appointments",
            "other_memory_items": ["Prefers afternoon check-ins"],
        }

    monkeypatch.setattr(
        patient_memory_service,
        "_request_patient_memory_payload",
        _fake_request_patient_memory_payload,
    )

    assessment = make_assessment(
        assessment_id="visit-memory-llm",
        patient_id="patient-memory-llm",
        created_at=datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc),
        risk_score=0.34,
        risk_tier="low",
        screening_classification="healthy",
    )

    identity.link_assessment(
        assessment,
        session_record=make_session_record(
            session_id="session-memory-llm",
            patient_name="Grace Lin",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.82,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "This morning I had breakfast and read the news.",
                "I set reminders on my phone for medicine and appointments.",
            ],
        ),
    )

    profile = identity.load_profile("patient-memory-llm")

    assert observed["model"] == "qwen3.5-plus"
    assert "[orientation]" in observed["transcript"]
    assert "[stage-1]" in observed["transcript"]
    assert profile.preferred_name == "Grace Lin"
    assert profile.memory == [
        "Preferred name: Grace Lin.",
        "Recent activity mentioned: Had breakfast and read the news with her sister.",
        "Routine/support detail: Uses phone reminders for medicine and appointments.",
        "Patient detail: Prefers afternoon check-ins.",
    ]


def test_identity_service_falls_back_to_rule_based_memory_when_llm_fails(tmp_path: Path, monkeypatch) -> None:
    settings = replace(make_settings(tmp_path), patient_memory_api_key="test-key")
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    patch_face_sequence(
        monkeypatch,
        make_face_evidence("llm-fallback-face", [0.3, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    def _raise_request_failure(*, transcript: str, settings) -> dict[str, object]:
        del transcript, settings
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        patient_memory_service,
        "_request_patient_memory_payload",
        _raise_request_failure,
    )

    assessment = make_assessment(
        assessment_id="visit-memory-fallback",
        patient_id="patient-memory-fallback",
        created_at=datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc),
        risk_score=0.34,
        risk_tier="low",
        screening_classification="healthy",
    )

    identity.link_assessment(
        assessment,
        session_record=make_session_record(
            session_id="session-memory-fallback",
            patient_name="Grace Lin",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.82,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "This morning I had breakfast and read the news.",
                "I set reminders on my phone for medicine and appointments.",
            ],
        ),
    )

    profile = identity.load_profile("patient-memory-fallback")

    assert profile.preferred_name == "Grace Lin"
    assert profile.memory[0] == "Preferred name: Grace Lin."
    assert profile.memory[1] == "Recent activity mentioned: This morning I had breakfast and read the news."
    assert profile.memory[2] == "Routine/support detail: I set reminders on my phone for medicine and appointments."


def test_realtime_identity_preflight_verifies_enrolled_patient_face(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    identity = IdentityLinkageService(storage)
    enrolled_face = make_face_evidence("realtime-face", [0.1, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    patch_face_sequence(monkeypatch, enrolled_face)

    assessment = make_assessment(
        assessment_id="visit-realtime-enroll",
        patient_id="patient-rt-1",
        created_at=datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc),
        risk_score=0.31,
        risk_tier="low",
        screening_classification="healthy",
    )
    identity.link_assessment(
        assessment,
        session_record=make_session_record(
            session_id="session-realtime-enroll",
            patient_name="Nina Gu",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.84,
            average_face_area=0.22,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "I made breakfast and watered the plants.",
                "I manage my reminders by myself.",
            ],
        ),
    )

    patch_frame_face_sequence(monkeypatch, enrolled_face)
    result = identity.check_realtime_identity(
        "patient-rt-1",
        image_base64_samples=["not-a-real-image"],
    )

    assert result.status == "verified"
    assert result.can_start_session is True
    assert result.requires_patient_reentry is False
    assert (result.face_match_confidence or 0) >= IdentityLinkageService.FACE_MATCH_INCLUDE_THRESHOLD


def test_legacy_face_profile_is_cleared_and_requires_reenrollment(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    storage = LocalStorage(settings)
    storage.save_identity_profile(
        IdentityProfile(
            profile_id="patient-legacy-identity",
            patient_id="patient-legacy",
            created_at=datetime.now(timezone.utc) - timedelta(days=14),
            updated_at=datetime.now(timezone.utc) - timedelta(days=2),
            enrollment_assessment_id="legacy-visit",
            canonical_face_embedding=[round(0.01 + (index * 0.0001), 4) for index in range(176)],
            canonical_voice_embedding=[0.1, 0.0, -0.1, 0.2, -0.2, 0.05, 0.02, -0.03],
            sessions_linked=3,
            sessions_included=2,
        )
    )
    identity = IdentityLinkageService(storage)
    patch_frame_face_sequence(
        monkeypatch,
        make_face_evidence("fresh-face", [0.1, 0.0, 0.2, 1.0, 0.0, 0.1, 0.0, 0.0]),
    )

    result = identity.check_realtime_identity(
        "patient-legacy",
        image_base64_samples=["not-a-real-image"],
    )
    profile = identity.load_profile("patient-legacy")

    assert result.status == "unenrolled"
    assert result.can_start_session is True
    assert result.requires_reenrollment is True
    assert profile.canonical_face_embedding == []
    assert profile.canonical_face_recognition_method is None
    assert profile.enrollment_assessment_id is None
    assert any("Legacy face profile was cleared" in note for note in profile.notes)


def test_identity_routes_expose_profile_and_link_records(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.assessments_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.prepared_dir.mkdir(parents=True, exist_ok=True)
    service = ClinicAssessmentService(settings)
    patch_face_sequence(
        monkeypatch,
        make_face_evidence("route-face", [0.1, 0.0, 0.1, 1.0, 0.0, 0.1, 0.0, 0.0]),
    )

    assessment = make_assessment(
        assessment_id="visit-route",
        patient_id="patient-route",
        created_at=datetime.now(timezone.utc) - timedelta(days=2),
        risk_score=0.34,
        risk_tier="low",
        screening_classification="healthy",
    )
    service.storage.save_assessment(assessment)
    service.identity.link_assessment(
        assessment,
        session_record=make_session_record(
            session_id="session-route",
            patient_name="Grace Lin",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.78,
            average_face_area=0.22,
            motion_intensity=0.27,
            transcript=[
                "I am at home today.",
                "This morning I had breakfast and read the news.",
                "I set reminders on my phone for medicine and appointments.",
            ],
        ),
    )

    app = FastAPI()
    app.include_router(build_api_router(service), prefix="/api")
    client = TestClient(app)

    profile_response = client.get("/api/identity/profile/patient-route")
    link_response = client.get("/api/identity/link/visit-route")

    assert profile_response.status_code == 200
    assert profile_response.json()["sessions_included"] == 1
    assert link_response.status_code == 200
    assert link_response.json()["assessment_id"] == "visit-route"
    assert link_response.json()["timeline_inclusion"] == "include"


def test_realtime_identity_check_route_blocks_mismatched_face(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.assessments_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.prepared_dir.mkdir(parents=True, exist_ok=True)
    service = ClinicAssessmentService(settings)

    enrolled_face = make_face_evidence("route-enrolled-face", [1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    other_face = make_face_evidence("route-other-face", [-1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    patch_face_sequence(monkeypatch, enrolled_face)

    assessment = make_assessment(
        assessment_id="visit-preflight-route",
        patient_id="patient-preflight-route",
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        risk_score=0.33,
        risk_tier="low",
        screening_classification="healthy",
    )
    service.storage.save_assessment(assessment)
    service.identity.link_assessment(
        assessment,
        session_record=make_session_record(
            session_id="session-preflight-route",
            patient_name="Route Patient",
            patient_turns=4,
            utterance_count=4,
            face_detection_rate=0.82,
            average_face_area=0.21,
            motion_intensity=0.28,
            transcript=[
                "I am at home today.",
                "I made tea and checked the calendar.",
                "I take my medicines on my own.",
            ],
        ),
    )
    patch_frame_face_sequence(monkeypatch, other_face)

    app = FastAPI()
    app.include_router(build_api_router(service), prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/clinic/realtime/identity/check",
        json={
            "patient_id": "patient-preflight-route",
            "image_base64_samples": ["not-a-real-image"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "mismatch"
    assert payload["can_start_session"] is False
    assert payload["requires_patient_reentry"] is True


def test_identity_reset_face_route_clears_profile_embedding(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.assessments_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.prepared_dir.mkdir(parents=True, exist_ok=True)
    service = ClinicAssessmentService(settings)
    service.storage.save_identity_profile(
        IdentityProfile(
            profile_id="patient-reset-identity",
            patient_id="patient-reset",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
            updated_at=datetime.now(timezone.utc) - timedelta(days=1),
            enrollment_assessment_id="visit-reset",
            canonical_face_embedding=[0.4, 0.2, 0.1, 0.0],
            canonical_face_recognition_method="test-face-descriptor",
            canonical_voice_embedding=[0.1, 0.0, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0],
            sessions_linked=4,
            sessions_included=3,
        )
    )

    app = FastAPI()
    app.include_router(build_api_router(service), prefix="/api")
    client = TestClient(app)

    response = client.post("/api/identity/profile/patient-reset/reset-face")

    assert response.status_code == 200
    payload = response.json()
    assert payload["canonical_face_embedding"] == []
    assert payload["canonical_face_recognition_method"] is None
    assert payload["enrollment_assessment_id"] is None
