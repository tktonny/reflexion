"""Longitudinal tracking models for per-patient monitoring over time."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.src.app.models.assessment import RiskTier, ScreeningClassification, SessionUsability
from backend.src.app.models.identity import IdentityTimelineInclusion


LongitudinalDirection = Literal["improving", "stable", "declining"]


class TaskBehaviorFeatures(BaseModel):
    pause_burden: float = Field(..., ge=0.0, le=1.0)
    speech_fluency: float = Field(..., ge=0.0, le=1.0)
    narrative_coherence: float = Field(..., ge=0.0, le=1.0)
    recall_consistency: float = Field(..., ge=0.0, le=1.0)
    support_dependency: float = Field(..., ge=0.0, le=1.0)


class VisualFaceFeatures(BaseModel):
    face_visibility: float = Field(..., ge=0.0, le=1.0)
    face_stability: float = Field(..., ge=0.0, le=1.0)
    motion_regularity: float = Field(..., ge=0.0, le=1.0)
    identity_confidence: float = Field(..., ge=0.0, le=1.0)


class LongitudinalQualityControl(BaseModel):
    qc_score: float = Field(..., ge=0.0, le=1.0)
    reviewer_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    session_usability: SessionUsability
    quality_flags: list[str] = Field(default_factory=list)


class LongitudinalSnapshot(BaseModel):
    snapshot_id: str
    patient_id: str
    assessment_id: str
    session_id: str = ""
    captured_at: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier | None = None
    screening_classification: ScreeningClassification | None = None
    identity_link_id: str | None = None
    timeline_inclusion: IdentityTimelineInclusion = "manual-review"
    identity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    identity_reasons: list[str] = Field(default_factory=list)
    speech_embedding: list[float] = Field(default_factory=list)
    conversation_embedding: list[float] = Field(default_factory=list)
    visual_face_embedding: list[float] = Field(default_factory=list)
    face_recognition_method: str | None = None
    task_behavior_features: TaskBehaviorFeatures
    visual_face_features: VisualFaceFeatures
    qc: LongitudinalQualityControl
    source_summary: str | None = None
    session_record_available: bool = False


class LongitudinalTrendPoint(BaseModel):
    label: str
    recorded_at: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    is_baseline: bool = False
    synthetic: bool = False


class LongitudinalProfile(BaseModel):
    patient_id: str
    updated_at: datetime
    total_sessions_seen: int = Field(default=0, ge=0)
    sessions_included: int = Field(default=0, ge=0)
    sessions_manual_review: int = Field(default=0, ge=0)
    sessions_excluded: int = Field(default=0, ge=0)
    baseline_at: datetime | None = None
    baseline_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    latest_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    direction: LongitudinalDirection = "stable"
    direction_label: str = "Stable"
    baseline_comparison: str = "Compared to your baseline (3 months ago), no longitudinal trend is available yet."
    anomaly_summary: str = "No abnormal deviation from baseline has been flagged yet."
    gating_summary: str = "No session has been held out from longitudinal tracking."
    x_axis_label: str = "Date"
    y_axis_label: str = "Risk Score"
    snapshots: list[LongitudinalSnapshot] = Field(default_factory=list)
    trend_points: list[LongitudinalTrendPoint] = Field(default_factory=list)
