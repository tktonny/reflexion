"""Pydantic models and shared type aliases for the clinic assessment flow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ProviderName = Literal["qwen_omni", "gemini", "fusion", "audio_only"]
RiskLabel = Literal["HC", "cognitive_risk"]
RiskTier = Literal["low", "medium", "high"]
ScreeningClassification = Literal["healthy", "needs_observation", "dementia"]
SpeakerStructure = Literal["single_speaker", "multi_speaker", "unclear"]
TargetPatientPresence = Literal["clear", "probable", "uncertain", "absent"]
TargetPatientBasis = Literal["visual_focus", "speaking_time", "interaction_role", "externally_provided", "unclear"]
SessionUsability = Literal["usable", "usable_with_caveats", "unusable"]
TraceStatus = Literal["success", "failed", "skipped"]

DEFAULT_DISCLAIMER = (
    "Research demo only. This output is not a diagnosis and must not be used as a "
    "standalone medical decision."
)


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ProviderAssessmentPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    visual_findings: list[Finding] = Field(default_factory=list)
    body_findings: list[Finding] = Field(default_factory=list)
    voice_findings: list[Finding] = Field(default_factory=list)
    content_findings: list[Finding] = Field(default_factory=list)
    speaker_structure: SpeakerStructure | None = None
    target_patient_presence: TargetPatientPresence | None = None
    target_patient_basis: TargetPatientBasis | None = None
    detected_languages: list[str] = Field(default_factory=list)
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_label: RiskLabel | None = None
    risk_tier: RiskTier | None = None
    screening_classification: ScreeningClassification | None = None
    screening_summary: str | None = None
    evidence_for_risk: list[str] = Field(default_factory=list)
    evidence_against_risk: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    risk_factor_findings: list[str] = Field(default_factory=list)
    subjective_assessment_findings: list[str] = Field(default_factory=list)
    emotional_assessment_findings: list[str] = Field(default_factory=list)
    risk_control_suggestions: list[str] = Field(default_factory=list)
    visit_recommendation: str | None = None
    future_risk_trend_summary: str | None = None
    reviewer_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    context_notes: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    session_usability: SessionUsability = "usable_with_caveats"
    disclaimer: str = DEFAULT_DISCLAIMER

    def should_fallback(self) -> bool:
        """Return True when the payload is too incomplete to serve as a final assessment."""
        if self.session_usability == "unusable":
            return True
        return self.risk_score is None or self.risk_label is None

    def fallback_debug_details(self) -> dict[str, object]:
        missing_required = []
        if self.risk_score is None:
            missing_required.append("risk_score")
        if self.risk_label is None:
            missing_required.append("risk_label")
        return {
            "session_usability": self.session_usability,
            "missing_required_fields": missing_required,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "risk_tier": self.risk_tier,
            "screening_classification": self.screening_classification,
            "quality_flags": list(self.quality_flags),
            "finding_counts": {
                "visual": len(self.visual_findings),
                "body": len(self.body_findings),
                "voice": len(self.voice_findings),
                "content": len(self.content_findings),
            },
        }

    def model_post_init(self, __context) -> None:
        if self.risk_tier is None and self.risk_score is not None:
            if self.risk_score < 0.35:
                self.risk_tier = "low"
            elif self.risk_score < 0.65:
                self.risk_tier = "medium"
            else:
                self.risk_tier = "high"
        if self.screening_classification is None:
            has_language_or_voice = bool(self.voice_findings or self.content_findings)
            target_isolated = self.target_patient_presence in {"clear", "probable", None}
            if self.session_usability == "unusable" or self.risk_score is None:
                self.screening_classification = None
            elif (
                self.risk_tier == "high"
                and self.risk_label == "cognitive_risk"
                and self.session_usability == "usable"
                and target_isolated
            ):
                self.screening_classification = "dementia"
            elif (
                self.risk_tier == "low"
                and self.risk_label == "HC"
                and self.session_usability == "usable"
                and has_language_or_voice
                and "transcript_unavailable" not in self.quality_flags
                and "speech_unintelligible" not in self.quality_flags
            ):
                self.screening_classification = "healthy"
            else:
                self.screening_classification = "needs_observation"
        if self.screening_summary is None and self.risk_tier is not None:
            if self.risk_tier == "low":
                self.screening_summary = (
                    "Low screening risk in this clip; limited evidence of dementia-like impairment."
                )
            elif self.risk_tier == "medium":
                self.screening_summary = (
                    "Medium screening risk in this clip; mixed or limited cues suggest further review."
                )
            elif self.risk_tier == "high":
                self.screening_summary = (
                    "High screening risk in this clip; convergent cues suggest prompt clinical follow-up."
                )
        if self.future_risk_trend_summary is None:
            self.future_risk_trend_summary = (
                "Insufficient longitudinal data to predict future cognitive risk trend from a single clip."
            )
        if self.visit_recommendation is None:
            if self.screening_classification == "healthy":
                self.visit_recommendation = (
                    "Routine follow-up only unless new symptoms or caregiver concerns arise."
                )
            elif self.screening_classification == "needs_observation":
                self.visit_recommendation = (
                    "Consider formal cognitive screening or clinician review if concerns persist."
                )
            elif self.screening_classification == "dementia":
                self.visit_recommendation = (
                    "Recommend prompt clinician review and formal cognitive evaluation."
                )
        if not self.risk_control_suggestions:
            if self.screening_classification == "healthy":
                self.risk_control_suggestions = [
                    "Maintain routine cognitive health monitoring.",
                    "Repeat screening if new symptoms emerge.",
                ]
            elif self.screening_classification == "needs_observation":
                self.risk_control_suggestions = [
                    "Repeat screening with clearer audio/video or a structured task.",
                    "Monitor changes over time rather than relying on one clip.",
                    "Consider formal cognitive screening if symptoms persist.",
                ]
            elif self.screening_classification == "dementia":
                self.risk_control_suggestions = [
                    "Arrange formal clinical cognitive evaluation.",
                    "Consider neurology or memory-clinic referral.",
                    "Review safety, medication, and caregiver support needs with a clinician.",
                ]


class ProviderCapabilities(BaseModel):
    native_video: bool
    native_audio_in_video: bool
    structured_output: bool
    requires_preprocessing: bool
    experimental: bool = False


class ProviderStatus(BaseModel):
    provider: ProviderName
    available: bool
    configured: bool
    mock_mode: bool
    fallback_rank: int
    description: str
    capabilities: ProviderCapabilities


class ProviderTraceEntry(BaseModel):
    provider: ProviderName
    attempt_order: int
    status: TraceStatus
    failure_reason: str | None = None
    latency_ms: int


class ProviderMeta(BaseModel):
    final_provider: ProviderName
    model_name: str
    request_id: str | None = None
    latency_ms: int
    raw_status: str


class ClinicAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    patient_id: str
    language: str
    created_at: datetime
    visual_findings: list[Finding]
    body_findings: list[Finding]
    voice_findings: list[Finding]
    content_findings: list[Finding]
    speaker_structure: SpeakerStructure | None = None
    target_patient_presence: TargetPatientPresence | None = None
    target_patient_basis: TargetPatientBasis | None = None
    detected_languages: list[str]
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_label: RiskLabel | None = None
    risk_tier: RiskTier | None = None
    screening_classification: ScreeningClassification | None = None
    screening_summary: str | None = None
    evidence_for_risk: list[str]
    evidence_against_risk: list[str]
    alternative_explanations: list[str]
    risk_factor_findings: list[str]
    subjective_assessment_findings: list[str]
    emotional_assessment_findings: list[str]
    risk_control_suggestions: list[str]
    visit_recommendation: str | None = None
    future_risk_trend_summary: str | None = None
    reviewer_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    context_notes: list[str]
    quality_flags: list[str]
    session_usability: SessionUsability
    disclaimer: str = DEFAULT_DISCLAIMER
    provider_meta: ProviderMeta
    provider_trace: list[ProviderTraceEntry]
    fallback_message: str | None = None


class AnalyzeRequestMetadata(BaseModel):
    patient_id: str = Field(..., min_length=1)
    language: str = Field(default="en", min_length=1)
    preferred_provider: ProviderName | None = None
    strict_provider: bool = False


class PreparedMedia(BaseModel):
    original_path: str
    standardized_path: str
    mime_type: str
    size_bytes: int
    duration_seconds: float | None = None
    extracted_audio_path: str | None = None
    frame_paths: list[str] = Field(default_factory=list)


class ProviderPrompt(BaseModel):
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, object]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_quality_flags(flags: list[str]) -> list[str]:
    normalized_flags: list[str] = []
    for item in flags:
        clean = item.strip().lower().replace(" ", "_").replace("-", "_")
        if clean and clean not in normalized_flags:
            normalized_flags.append(clean)
    return normalized_flags


class ProviderRawResult(BaseModel):
    payload: ProviderAssessmentPayload
    request_id: str | None = None
    raw_status: str = "ok"
    debug_details: dict[str, object] | None = None


class ProviderContext(BaseModel):
    assessment_id: str
    patient_id: str
    language: str
    preferred_provider: ProviderName | None = None
    strict_provider: bool = False
    media: PreparedMedia


class ProvidersResponse(BaseModel):
    default_provider: ProviderName
    fallback_order: list[ProviderName]
    providers: list[ProviderStatus]
