"""Unit tests for the standalone Raspberry Pi audio Flask server."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

import audio_server


@pytest.fixture
def client():
    audio_server.app.config["TESTING"] = True
    with audio_server.app.test_client() as test_client:
        yield test_client


def test_analyze_requires_file_field(client) -> None:
    response = client.post("/analyze", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {
        "ok": False,
        "error": "missing_file",
        "message": "missing WAV file in multipart field 'file'",
    }


def test_analyze_rejects_non_wav_upload(client) -> None:
    response = client.post(
        "/analyze",
        data={"file": (BytesIO(b"not-a-wav"), "sample.mp3")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 415
    assert response.get_json()["error"] == "invalid_file_type"


def test_analyze_returns_audio_only_assessment(client, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_analyze_saved_wav(audio_path: Path, patient_id: str, language: str) -> dict[str, object]:
        assert audio_path.suffix == ".wav"
        assert patient_id == audio_server.DEFAULT_PATIENT_ID
        assert language == audio_server.DEFAULT_LANGUAGE
        return {
            "assessment_id": "assessment-123",
            "patient_id": patient_id,
            "language": language,
            "created_at": "2026-04-15T00:00:00+00:00",
            "visual_findings": [],
            "body_findings": [],
            "voice_findings": [{"label": "hesitation", "summary": "Frequent hesitations"}],
            "content_findings": [],
            "speaker_structure": "single_speaker",
            "target_patient_presence": "clear",
            "target_patient_basis": "externally_provided",
            "detected_languages": ["en"],
            "language_confidence": 0.98,
            "risk_score": 0.41,
            "risk_label": "cognitive_risk",
            "risk_tier": "medium",
            "screening_classification": "needs_observation",
            "screening_summary": "Mixed speech cues warrant observation.",
            "evidence_for_risk": ["Frequent hesitations"],
            "evidence_against_risk": [],
            "alternative_explanations": ["Background stress may contribute."],
            "risk_factor_findings": [],
            "subjective_assessment_findings": [],
            "emotional_assessment_findings": [],
            "risk_control_suggestions": ["Repeat screening with a longer sample."],
            "visit_recommendation": "Consider formal cognitive screening if concerns persist.",
            "future_risk_trend_summary": "Insufficient longitudinal data.",
            "reviewer_confidence": 0.63,
            "context_notes": ["Audio-only review."],
            "quality_flags": [],
            "session_usability": "usable",
            "disclaimer": "Research demo only.",
            "provider_meta": {
                "final_provider": "audio_only",
                "model_name": "gpt-4.1",
                "request_id": "req_123",
                "latency_ms": 111,
                "raw_status": "ok",
            },
            "provider_trace": [
                {
                    "provider": "audio_only",
                    "attempt_order": 1,
                    "status": "success",
                    "latency_ms": 111,
                }
            ],
            "fallback_message": None,
        }

    monkeypatch.setattr(audio_server, "analyze_saved_wav", fake_analyze_saved_wav)

    response = client.post(
        "/analyze",
        data={"file": (BytesIO(b"RIFF....WAVEfmt "), "sample.wav", "audio/wav")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["assessment_id"] == "assessment-123"
    assert payload["risk_label"] == "cognitive_risk"
    assert payload["provider_meta"]["final_provider"] == "audio_only"
    assert payload["received_filename"] == "sample.wav"
