"""Domain models for the realtime conversational demo flow."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.src.app.models.assessment import DEFAULT_DISCLAIMER


ConversationRole = Literal["assistant", "patient"]
ConversationStage = Literal["orientation", "recent_story", "daily_function", "delayed_recall"]
SessionMode = Literal["live_qwen", "guided_demo"]
RiskBand = Literal["low", "medium", "high"]


class RealtimePromptStep(BaseModel):
    key: ConversationStage
    title: str
    goal: str
    prompt: str
    rationale: str
    exit_when: list[str] = Field(default_factory=list)
    max_follow_ups: int = Field(default=1, ge=0, le=3)


class RealtimeConversationFlow(BaseModel):
    flow_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    opening_message: str = Field(..., min_length=1)
    conversation_goal: str = Field(..., min_length=1)
    completion_rule: str = Field(..., min_length=1)
    completion_message: str = Field(..., min_length=1)
    assistant_response_rules: list[str] = Field(default_factory=list)
    processing_steps: list[str] = Field(default_factory=list)
    steps: list[RealtimePromptStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_steps(self) -> "RealtimeConversationFlow":
        expected = {"orientation", "recent_story", "daily_function", "delayed_recall"}
        actual = {step.key for step in self.steps}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(
                "Realtime conversation flow must define exactly the four required stages"
                + (f" ({', '.join(details)})" if details else "")
            )
        if len(self.steps) != len(expected):
            raise ValueError("Realtime conversation flow cannot repeat stages")
        return self


class RealtimeSessionStatus(BaseModel):
    session_mode: SessionMode
    conversation_provider: str
    model_name: str
    live_relay_available: bool
    selected_voice: str | None = None
    selected_language: str | None = None
    voice_selection_source: str | None = None
    flow_id: str
    flow_title: str
    conversation_goal: str
    completion_rule: str
    greeting: str
    prompt_steps: list[RealtimePromptStep]
    processing_steps: list[str]
    fallback_note: str | None = None


class RealtimeTranscriptTurn(BaseModel):
    role: ConversationRole
    text: str = Field(..., min_length=1)
    stage: ConversationStage | None = None


class AudioCaptureMetrics(BaseModel):
    utterance_count: int = Field(default=0, ge=0)
    speech_seconds: float = Field(default=0.0, ge=0.0)
    average_turn_seconds: float | None = Field(default=None, ge=0.0)
    audio_chunk_count: int = Field(default=0, ge=0)


class VisualCaptureMetrics(BaseModel):
    frames_captured: int = Field(default=0, ge=0)
    face_detection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_face_area: float | None = Field(default=None, ge=0.0, le=1.0)
    motion_intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_brightness: float | None = Field(default=None, ge=0.0, le=1.0)


class RealtimeAnalysisRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    language: str = Field(default="en", min_length=1)
    transcript: list[RealtimeTranscriptTurn] = Field(default_factory=list)
    audio_metrics: AudioCaptureMetrics = Field(default_factory=AudioCaptureMetrics)
    visual_metrics: VisualCaptureMetrics = Field(default_factory=VisualCaptureMetrics)


class FeatureSignal(BaseModel):
    label: str
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str


class SimilarityBreakdown(BaseModel):
    dementia_pattern_similarity: float = Field(..., ge=0.0, le=1.0)
    non_dementia_pattern_similarity: float = Field(..., ge=0.0, le=1.0)
    speech_weight: float = Field(..., ge=0.0, le=1.0)
    visual_weight: float = Field(..., ge=0.0, le=1.0)


class TrendPoint(BaseModel):
    label: str
    risk_score: float = Field(..., ge=0.0, le=1.0)


class RealtimeAssessment(BaseModel):
    assessment_id: str
    patient_id: str
    language: str
    created_at: datetime
    risk_label: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_band: RiskBand
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_reasons: list[str]
    recommendation: str
    transcript_summary: str
    speech_features: list[FeatureSignal]
    visual_features: list[FeatureSignal]
    similarity: SimilarityBreakdown
    trend: list[TrendPoint]
    processing_summary: list[str]
    quality_flags: list[str]
    disclaimer: str = DEFAULT_DISCLAIMER
