"""Models for patient identity attribution and cross-session linkage."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.src.app.models.assessment import TargetPatientPresence


IdentityLinkageVerdict = Literal["confirmed", "probable", "uncertain", "mismatch", "no-face"]
IdentityTimelineInclusion = Literal["include", "exclude", "manual-review"]
IdentityPreflightStatus = Literal["verified", "mismatch", "needs-retry", "unenrolled"]


class IdentityLinkRecord(BaseModel):
    link_id: str
    patient_id: str
    assessment_id: str
    session_id: str
    assessed_at: datetime
    enrollment_profile_id: str | None = None
    target_presence: TargetPatientPresence = "uncertain"
    dominant_track_id: str | None = None
    within_session_attribution_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_sample_count: int = Field(default=0, ge=0)
    face_recognition_method: str | None = None
    face_match_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    voice_match_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    final_linkage_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    linkage_verdict: IdentityLinkageVerdict
    timeline_inclusion: IdentityTimelineInclusion
    reasons: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    face_embedding: list[float] = Field(default_factory=list)
    voice_embedding: list[float] = Field(default_factory=list)


class IdentityProfile(BaseModel):
    profile_id: str
    patient_id: str
    created_at: datetime
    updated_at: datetime
    enrollment_assessment_id: str | None = None
    preferred_name: str | None = None
    memory: list[str] = Field(default_factory=list)
    canonical_face_embedding: list[float] = Field(default_factory=list)
    canonical_face_recognition_method: str | None = None
    canonical_voice_embedding: list[float] = Field(default_factory=list)
    sessions_linked: int = Field(default=0, ge=0)
    sessions_included: int = Field(default=0, ge=0)
    sessions_manual_review: int = Field(default=0, ge=0)
    sessions_excluded: int = Field(default=0, ge=0)
    latest_link_id: str | None = None
    latest_linkage_verdict: IdentityLinkageVerdict = "uncertain"
    latest_linkage_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class IdentityPreflightRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    image_base64_samples: list[str] = Field(default_factory=list, min_length=1, max_length=6)


class IdentityPreflightResult(BaseModel):
    patient_id: str
    status: IdentityPreflightStatus
    can_start_session: bool
    requires_patient_reentry: bool = False
    requires_reenrollment: bool = False
    enrolled_face_exists: bool = False
    face_match_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_sample_count: int = Field(default=0, ge=0)
    face_recognition_method: str | None = None
    dominant_track_id: str | None = None
    summary: str
    recommended_action: str
    reasons: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
