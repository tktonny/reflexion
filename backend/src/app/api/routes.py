"""HTTP routes for clinic assessment upload and result retrieval."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from backend.src.app.core.errors import RoutingError
from backend.src.app.models import (
    AnalyzeRequestMetadata,
    IdentityPreflightRequest,
    IdentityPreflightResult,
    IdentityLinkRecord,
    IdentityProfile,
    KnownPatientsResponse,
    ProviderName,
    RealtimeAnalysisRequest,
)
from backend.src.app.services.assessment_service import ClinicAssessmentService
from backend.src.app.services.realtime_service import RealtimeConversationService
from backend.src.app.services.surface_service import SurfaceService


logger = logging.getLogger("clinic.api")


def build_api_router(service: ClinicAssessmentService) -> APIRouter:
    """Build the API router around a ready-to-use assessment service."""
    router = APIRouter()
    realtime = RealtimeConversationService(service.settings)
    surfaces = SurfaceService(service.storage)

    @router.get("/providers")
    async def get_providers():
        return await service.router.health()

    @router.post("/clinic/video/analyze")
    async def analyze_video(
        video: UploadFile = File(...),
        patient_id: str = Form(...),
        language: str = Form("en"),
        preferred_provider: ProviderName | None = Form(default=None),
        strict_provider: bool = Form(default=False),
        session_record_json: str | None = Form(default=None),
    ):
        metadata = AnalyzeRequestMetadata(
            patient_id=patient_id,
            language=language,
            preferred_provider=preferred_provider,
            strict_provider=strict_provider,
        )
        session_record: dict[str, object] | None = None
        if session_record_json:
            try:
                raw_record = json.loads(session_record_json)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail="invalid session_record_json") from exc
            if not isinstance(raw_record, dict):
                raise HTTPException(status_code=422, detail="session_record_json must decode to an object")
            session_record = raw_record
        try:
            assessment = await service.analyze_upload(
                video,
                metadata,
                session_record=session_record,
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except RoutingError as exc:
            logger.warning(
                "Batch clinic analysis failed code=%s patient_id=%s provider_trace=%s",
                exc.code,
                metadata.patient_id,
                exc.provider_trace,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": exc.code,
                    "message": exc.message,
                    "provider_trace": exc.provider_trace,
                },
            )
        return assessment

    @router.get("/clinic/assessments/{assessment_id}")
    async def get_assessment(assessment_id: str):
        try:
            return service.load_assessment(assessment_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="assessment not found") from exc

    @router.get("/identity/profile/{patient_id}", response_model=IdentityProfile)
    async def get_identity_profile(patient_id: str):
        return service.load_identity_profile(patient_id)

    @router.post("/identity/profile/{patient_id}/reset-face", response_model=IdentityProfile)
    async def reset_identity_face_profile(patient_id: str):
        return service.reset_identity_face_profile(patient_id)

    @router.get("/identity/link/{assessment_id}", response_model=IdentityLinkRecord)
    async def get_identity_link(assessment_id: str):
        try:
            return service.load_identity_link(assessment_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="identity link not found") from exc

    @router.get("/doctor/dashboard")
    async def get_doctor_dashboard():
        return surfaces.build_doctor_dashboard()

    @router.get("/care/patients")
    @router.get("/caregiver/patients")
    async def get_caregiver_patients():
        return KnownPatientsResponse(patients=surfaces.list_known_patients())

    @router.get("/care/dashboard/{patient_id}")
    @router.get("/caregiver/dashboard/{patient_id}")
    async def get_caregiver_dashboard(patient_id: str):
        return surfaces.build_caregiver_dashboard(patient_id)

    @router.get("/clinic/realtime/status")
    async def get_realtime_status():
        return realtime.build_session_status()

    @router.post("/clinic/realtime/identity/check", response_model=IdentityPreflightResult)
    async def check_realtime_identity(request: IdentityPreflightRequest):
        result = service.check_realtime_identity(
            request.patient_id,
            image_base64_samples=request.image_base64_samples,
        )
        logging.getLogger("uvicorn.error").info(
            "Realtime identity route patient_id=%s status=%s can_start=%s requires_reentry=%s requires_reenrollment=%s face_match=%s sample_count=%s",
            result.patient_id,
            result.status,
            result.can_start_session,
            result.requires_patient_reentry,
            result.requires_reenrollment,
            f"{result.face_match_confidence:.4f}" if result.face_match_confidence is not None else "none",
            result.face_sample_count,
        )
        return result

    @router.post("/clinic/realtime/analyze")
    async def analyze_realtime_session(request: RealtimeAnalysisRequest):
        return realtime.analyze_session(request)

    @router.websocket("/clinic/realtime/ws")
    async def clinic_realtime_session(
        websocket: WebSocket,
        patient_id: str = "demo-patient",
        language: str = "en",
    ):
        await realtime.run_session(websocket, patient_id=patient_id, language=language)

    return router
