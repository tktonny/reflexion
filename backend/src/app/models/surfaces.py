"""View models for doctor and caregiver product surfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.src.app.models.assessment import (
    ProviderName,
    RiskTier,
    ScreeningClassification,
    SessionUsability,
)
from backend.src.app.models.longitudinal import LongitudinalDirection


CareSignal = Literal["stable", "watch", "urgent"]
CareRiskLabel = Literal["HIGH RISK", "LOW RISK"]


class SurfaceAssessmentSummary(BaseModel):
    assessment_id: str
    patient_id: str
    created_at: datetime
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_tier: RiskTier | None = None
    screening_classification: ScreeningClassification | None = None
    screening_summary: str | None = None
    session_usability: SessionUsability
    reviewer_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    final_provider: ProviderName
    visit_recommendation: str | None = None


class DoctorPatientSummary(BaseModel):
    patient_id: str
    session_count: int = Field(..., ge=0)
    latest_assessment_id: str | None = None
    latest_assessment_at: datetime | None = None
    latest_risk_tier: RiskTier | None = None
    latest_screening_classification: ScreeningClassification | None = None
    latest_summary: str | None = None
    average_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ProviderDistributionItem(BaseModel):
    provider: ProviderName
    assessments: int = Field(..., ge=0)


class DoctorDashboardResponse(BaseModel):
    generated_at: datetime
    total_assessments: int = Field(..., ge=0)
    total_patients: int = Field(..., ge=0)
    high_risk_assessments: int = Field(..., ge=0)
    watchlist_assessments: int = Field(..., ge=0)
    usable_sessions: int = Field(..., ge=0)
    average_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_distribution: list[ProviderDistributionItem] = Field(default_factory=list)
    recent_assessments: list[SurfaceAssessmentSummary] = Field(default_factory=list)
    patient_summaries: list[DoctorPatientSummary] = Field(default_factory=list)


class KnownPatientsResponse(BaseModel):
    patients: list[str] = Field(default_factory=list)


class CaregiverTrendPoint(BaseModel):
    label: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    recorded_at: datetime | None = None
    is_baseline: bool = False
    synthetic: bool = False


class CaregiverTimelinePoint(BaseModel):
    assessment_id: str
    created_at: datetime
    signal: CareSignal
    status_label: str
    shareable_summary: str


class CaregiverDashboardResponse(BaseModel):
    generated_at: datetime
    patient_id: str
    known_patients: list[str] = Field(default_factory=list)
    sessions_completed: int = Field(..., ge=0)
    last_updated_at: datetime | None = None
    risk_label: CareRiskLabel
    risk_score: float = Field(..., ge=0.0, le=1.0)
    top_reasons: list[str] = Field(default_factory=list)
    recommendation: str
    baseline_comparison: str = "Compared to your baseline (3 months ago), no longitudinal trend is available yet."
    longitudinal_direction: LongitudinalDirection = "stable"
    longitudinal_direction_label: str = "Stable"
    anomaly_summary: str = "No abnormal deviation from baseline has been flagged yet."
    identity_gating_summary: str = "No session has been held out from longitudinal tracking."
    chart_x_axis_label: str = "Date"
    chart_y_axis_label: str = "Risk Score"
    longitudinal_points: list[CaregiverTrendPoint] = Field(default_factory=list)
    signal: CareSignal = "stable"
    status_label: str
    care_summary: str
    next_steps: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    latest_assessment: SurfaceAssessmentSummary | None = None
    history: list[CaregiverTimelinePoint] = Field(default_factory=list)
