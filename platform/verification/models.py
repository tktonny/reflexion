"""Typed models for verification manifests, outputs, and summaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


GroundTruthLabel = Literal["HC", "cognitive_risk"]
MediaType = Literal["audio", "video"]


class VerificationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    split: str = Field(default="unknown", min_length=1)
    label: GroundTruthLabel
    media_path: str = Field(..., min_length=1)
    media_type: MediaType = "audio"
    language: str = Field(default="en", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreparedAudioArtifact(BaseModel):
    case_id: str
    source_path: str
    audio_path: str
    mime_type: str
    duration_seconds: float | None = None
    size_bytes: int


class VerificationCaseResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    dataset: str
    split: str
    ground_truth_label: GroundTruthLabel
    predicted_label: GroundTruthLabel | None = None
    predicted_screening_classification: str | None = None
    risk_score: float | None = None
    reviewer_confidence: float | None = None
    transcript: str | None = None
    patient_only_transcript: str | None = None
    speaker_turn_summary: list[str] = Field(default_factory=list)
    patient_cue_summary: list[str] = Field(default_factory=list)
    transcript_request_id: str | None = None
    classification_request_id: str | None = None
    transcript_model: str
    classifier_model: str
    audio_path: str | None = None
    prepared_media_path: str | None = None
    assessment: dict[str, Any] | None = None
    latency_ms: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationMetrics(BaseModel):
    evaluated_cases: int
    successful_cases: int
    error_cases: int
    skipped_cases: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    confusion_matrix: dict[str, int] = Field(default_factory=dict)


class VerificationSummary(BaseModel):
    dataset: str
    manifest_path: str
    results_path: str
    metrics: VerificationMetrics
    cases: list[VerificationCaseResult]
