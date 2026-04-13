"""Core orchestration for clinic assessments and provider fallback routing."""

from __future__ import annotations

import time
import uuid
from typing import Any

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError, RoutingError
from backend.src.app.models import (
    AnalyzeRequestMetadata,
    ClinicAssessment,
    IdentityPreflightResult,
    IdentityLinkRecord,
    IdentityProfile,
    ProviderContext,
    ProviderMeta,
    ProviderName,
    ProviderTraceEntry,
    ProvidersResponse,
    utc_now,
)
from backend.src.app.services.identity_service import IdentityLinkageService
from backend.src.app.services.longitudinal_service import LongitudinalTrackingService
from backend.src.app.services.media_preparer import MediaPreparer
from clinic.database.storage import LocalStorage
from clinic.intelligence.providers.registry import build_provider_registry


class ProviderMeshRouter:
    """Route one assessment request through the provider mesh with ordered fallback."""

    def __init__(self, settings: Settings, media_preparer: MediaPreparer | None = None) -> None:
        self.settings = settings
        self.media_preparer = media_preparer or MediaPreparer(settings)
        self.providers = build_provider_registry(settings)

    def resolve_order(
        self,
        preferred_provider: ProviderName | None,
        strict_provider: bool,
    ) -> list[ProviderName]:
        default_order = list(self.settings.fallback_order)
        if preferred_provider is None:
            return default_order
        if strict_provider:
            return [preferred_provider]
        return [preferred_provider] + [item for item in default_order if item != preferred_provider]

    async def health(self) -> ProvidersResponse:
        providers = []
        for index, provider_name in enumerate(self.settings.fallback_order, start=1):
            provider = self.providers[provider_name]
            providers.append(await provider.healthcheck(index))
        return ProvidersResponse(
            default_provider=self.settings.default_provider,
            fallback_order=list(self.settings.fallback_order),
            providers=providers,
        )

    async def analyze(self, context: ProviderContext) -> ClinicAssessment:
        order = self.resolve_order(context.preferred_provider, context.strict_provider)
        trace: list[ProviderTraceEntry] = []
        error_trace: list[dict[str, object]] = []
        last_error = "provider_unavailable"

        for attempt_order, provider_name in enumerate(order, start=1):
            provider = self.providers[provider_name]
            started = time.perf_counter()
            try:
                provider_media = self.media_preparer.prepare_for_provider(provider_name, context.media)
                provider_context = context.model_copy(update={"media": provider_media})
                provider_input = await provider.prepare(provider_context)
                raw_result = await provider.analyze(provider_input, provider_context)
                payload = provider.normalize(raw_result)
                latency_ms = int((time.perf_counter() - started) * 1000)
                if payload.should_fallback():
                    last_error = "unusable_result"
                    trace_entry = ProviderTraceEntry(
                        provider=provider_name,
                        attempt_order=attempt_order,
                        status="failed",
                        failure_reason="unusable_result",
                        latency_ms=latency_ms,
                    )
                    trace.append(trace_entry)
                    trace_payload = trace_entry.model_dump()
                    trace_payload["debug_details"] = payload.fallback_debug_details()
                    if raw_result.debug_details:
                        trace_payload["debug_details"].update(raw_result.debug_details)
                    trace_payload["raw_status"] = raw_result.raw_status
                    trace_payload["request_id"] = raw_result.request_id
                    error_trace.append(trace_payload)
                    if context.strict_provider:
                        break
                    continue

                trace.append(
                    ProviderTraceEntry(
                        provider=provider_name,
                        attempt_order=attempt_order,
                        status="success",
                        latency_ms=latency_ms,
                    )
                )
                fallback_message = None
                if len(trace) > 1:
                    previous = trace[-2]
                    fallback_message = (
                        f"Processed by {provider_name} after {previous.provider} "
                        f"{previous.failure_reason or previous.status}."
                    )
                return ClinicAssessment(
                    assessment_id=context.assessment_id,
                    patient_id=context.patient_id,
                    language=context.language,
                    created_at=utc_now(),
                    visual_findings=payload.visual_findings,
                    body_findings=payload.body_findings,
                    voice_findings=payload.voice_findings,
                    content_findings=payload.content_findings,
                    speaker_structure=payload.speaker_structure,
                    target_patient_presence=payload.target_patient_presence,
                    target_patient_basis=payload.target_patient_basis,
                    detected_languages=payload.detected_languages,
                    language_confidence=payload.language_confidence,
                    risk_score=payload.risk_score,
                    risk_label=payload.risk_label,
                    risk_tier=payload.risk_tier,
                    screening_classification=payload.screening_classification,
                    screening_summary=payload.screening_summary,
                    evidence_for_risk=payload.evidence_for_risk,
                    evidence_against_risk=payload.evidence_against_risk,
                    alternative_explanations=payload.alternative_explanations,
                    risk_factor_findings=payload.risk_factor_findings,
                    subjective_assessment_findings=payload.subjective_assessment_findings,
                    emotional_assessment_findings=payload.emotional_assessment_findings,
                    risk_control_suggestions=payload.risk_control_suggestions,
                    visit_recommendation=payload.visit_recommendation,
                    future_risk_trend_summary=payload.future_risk_trend_summary,
                    reviewer_confidence=payload.reviewer_confidence,
                    context_notes=payload.context_notes,
                    quality_flags=payload.quality_flags,
                    session_usability=payload.session_usability,
                    disclaimer=payload.disclaimer,
                    provider_meta=ProviderMeta(
                        final_provider=provider_name,
                        model_name=provider.model_name,
                        request_id=raw_result.request_id,
                        latency_ms=latency_ms,
                        raw_status=raw_result.raw_status,
                    ),
                    provider_trace=trace,
                    fallback_message=fallback_message,
                )
            except ProviderError as exc:
                last_error = exc.code
                trace_entry = ProviderTraceEntry(
                    provider=provider_name,
                    attempt_order=attempt_order,
                    status="failed",
                    failure_reason=exc.code,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                trace.append(trace_entry)
                trace_payload = trace_entry.model_dump()
                trace_payload["message"] = exc.message
                if exc.debug_details:
                    trace_payload["debug_details"] = exc.debug_details
                error_trace.append(trace_payload)
                if context.strict_provider:
                    break

        raise RoutingError(
            code=last_error,
            message="All providers failed to produce a usable assessment",
            provider_trace=error_trace or [entry.model_dump() for entry in trace],
        )


class ClinicAssessmentService:
    """Coordinate uploads, media preparation, provider inference, and persistence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = LocalStorage(settings)
        self.media = MediaPreparer(settings)
        self.router = ProviderMeshRouter(settings, media_preparer=self.media)
        self.identity = IdentityLinkageService(self.storage)
        self.longitudinal = LongitudinalTrackingService(self.storage)

    async def analyze_upload(
        self,
        upload: Any,
        metadata: AnalyzeRequestMetadata,
        *,
        session_record: dict[str, Any] | None = None,
    ) -> ClinicAssessment:
        assessment_id = uuid.uuid4().hex
        saved_path = await self.storage.save_upload(assessment_id, upload)
        if session_record:
            self.storage.save_upload_sidecar(
                assessment_id,
                filename="session_record.json",
                payload=session_record,
            )
        prepared_media = self.media.prepare_base(assessment_id, saved_path)
        context = ProviderContext(
            assessment_id=assessment_id,
            patient_id=metadata.patient_id,
            language=metadata.language,
            preferred_provider=metadata.preferred_provider,
            strict_provider=metadata.strict_provider,
            media=prepared_media,
        )
        assessment = await self.router.analyze(context)
        self.storage.save_assessment(assessment)
        identity_link = self.identity.link_assessment(
            assessment,
            session_record=session_record,
            media=prepared_media,
        )
        self.longitudinal.record_assessment(
            assessment,
            session_record=session_record,
            identity_link=identity_link,
        )
        return assessment

    def load_assessment(self, assessment_id: str) -> dict[str, Any]:
        return self.storage.load_assessment(assessment_id)

    def load_identity_profile(self, patient_id: str) -> IdentityProfile:
        return self.identity.load_profile(patient_id)

    def load_identity_link(self, assessment_id: str) -> IdentityLinkRecord:
        return self.identity.load_link(assessment_id)

    def reset_identity_face_profile(self, patient_id: str) -> IdentityProfile:
        return self.identity.reset_face_profile(patient_id)

    def check_realtime_identity(
        self,
        patient_id: str,
        *,
        image_base64_samples: list[str],
    ) -> IdentityPreflightResult:
        return self.identity.check_realtime_identity(
            patient_id,
            image_base64_samples=image_base64_samples,
        )
