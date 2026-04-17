"""Identity attribution and cross-session linkage for patient tracking."""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.src.app.models.assessment import (
    ClinicAssessment,
    PreparedMedia,
    normalize_quality_flags,
    utc_now,
)
from backend.src.app.models.identity import (
    IdentityLinkRecord,
    IdentityPreflightResult,
    IdentityProfile,
)
from backend.src.app.services.patient_memory import build_patient_memory
from clinic.database.storage import LocalStorage

try:  # pragma: no cover - import availability depends on local env
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - import availability depends on local env
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:  # pragma: no cover - import availability depends on local env
    from skimage.feature import hog, local_binary_pattern
except ImportError:  # pragma: no cover
    hog = None
    local_binary_pattern = None

try:  # pragma: no cover - import availability depends on local env
    from skimage.filters import sobel
except ImportError:  # pragma: no cover
    sobel = None


TOKEN_RE = re.compile(r"[a-z0-9]+")
logger = logging.getLogger("uvicorn.error")


@dataclass
class FaceEvidence:
    descriptor: list[float] = field(default_factory=list)
    sample_count: int = 0
    detection_rate: float | None = None
    average_face_area: float | None = None
    dominant_track_id: str | None = None
    method: str | None = None
    quality_flags: list[str] = field(default_factory=list)


class IdentityLinkageService:
    """Evaluate whether one session belongs to the intended patient timeline."""

    EMBEDDING_DIM = 8
    FACE_ENCODER_METHOD = "opencv_hog_lbp_v2"
    LEGACY_FACE_ENCODER_METHODS = {"opencv_lbp_v1", "opencv-lbp-face-v1"}
    FACE_DESCRIPTOR_POINTS = 8
    FACE_DESCRIPTOR_RADIUS = 1
    FACE_DESCRIPTOR_GRID = 4
    FACE_DESCRIPTOR_BINS = FACE_DESCRIPTOR_POINTS + 2
    FACE_DESCRIPTOR_SIZE = 96
    FACE_DESCRIPTOR_LOW_RES = 24
    FACE_DESCRIPTOR_PROJECTION_BINS = 24
    FACE_HOG_ORIENTATIONS = 9
    FACE_HOG_PIXELS_PER_CELL = 16
    FACE_HOG_CELLS_PER_BLOCK = 2
    MAX_FACE_SAMPLES = 12
    CONFIRMED_THRESHOLD = 0.78
    PROBABLE_THRESHOLD = 0.64
    UNCERTAIN_THRESHOLD = 0.48
    FACE_MATCH_INCLUDE_THRESHOLD = 0.82
    FACE_MATCH_REVIEW_THRESHOLD = 0.70
    FACE_MATCH_EXCLUDE_THRESHOLD = 0.58
    FACE_MIN_SAMPLE_COUNT = 2
    FACE_CANONICAL_REFRESH_THRESHOLD = 0.94
    FACE_CANONICAL_REFRESH_MIN_SAMPLES = 6

    def __init__(self, storage: LocalStorage) -> None:
        self.storage = storage

    def ensure_link(
        self,
        assessment: ClinicAssessment,
        *,
        session_record: dict[str, Any] | None = None,
        media: PreparedMedia | None = None,
    ) -> IdentityLinkRecord:
        try:
            payload = self.storage.load_identity_link(assessment.assessment_id)
        except FileNotFoundError:
            return self.link_assessment(
                assessment,
                session_record=session_record,
                media=media or self._load_prepared_media(assessment.assessment_id),
            )
        return IdentityLinkRecord.model_validate(payload)

    def load_profile(self, patient_id: str) -> IdentityProfile:
        try:
            payload = self.storage.load_identity_profile(patient_id)
        except FileNotFoundError:
            return IdentityProfile(
                profile_id=f"{patient_id}-identity",
                patient_id=patient_id,
                created_at=utc_now(),
                updated_at=utc_now(),
            )
        profile = IdentityProfile.model_validate(payload)
        normalized_profile = self._normalize_profile(profile)
        if normalized_profile != profile:
            self.storage.save_identity_profile(normalized_profile)
        return normalized_profile

    def reset_face_profile(self, patient_id: str) -> IdentityProfile:
        profile = self.load_profile(patient_id)
        reset_profile = self._clear_face_enrollment(
            profile,
            reason="Face enrollment was reset so this patient can be re-enrolled with a fresh face template.",
        )
        self.storage.save_identity_profile(reset_profile)
        logger.info(
            "Identity face profile reset patient_id=%s previous_method=%s previous_dimensions=%s",
            patient_id,
            profile.canonical_face_recognition_method,
            len(profile.canonical_face_embedding),
        )
        return reset_profile

    def load_link(self, assessment_id: str) -> IdentityLinkRecord:
        payload = self.storage.load_identity_link(assessment_id)
        return IdentityLinkRecord.model_validate(payload)

    def check_realtime_identity(
        self,
        patient_id: str,
        *,
        image_base64_samples: list[str],
    ) -> IdentityPreflightResult:
        profile = self.load_profile(patient_id)
        face_evidence = self._extract_face_evidence_from_frames(
            self._decode_base64_frames(image_base64_samples)
        )
        has_enrolled_face = self._has_comparable_enrolled_face(profile, current_method=face_evidence.method)
        has_current_face = face_evidence.sample_count >= self.FACE_MIN_SAMPLE_COUNT and bool(face_evidence.descriptor)
        reasons: list[str] = []

        if not has_current_face:
            logger.info(
                "Identity preflight patient_id=%s status=needs-retry enrolled_face=%s sample_count=%s quality_flags=%s",
                patient_id,
                has_enrolled_face,
                face_evidence.sample_count,
                face_evidence.quality_flags,
            )
            reasons.append("A stable patient face could not be extracted from the opening camera frames.")
            if face_evidence.sample_count > 0:
                reasons.append("Only a limited number of usable face samples were available for matching.")
            if face_evidence.quality_flags:
                reasons.extend(
                    self._quality_flag_reason(flag)
                    for flag in face_evidence.quality_flags
                    if self._quality_flag_reason(flag)
                )
            return IdentityPreflightResult(
                patient_id=patient_id,
                status="needs-retry",
                can_start_session=False,
                enrolled_face_exists=has_enrolled_face,
                face_sample_count=face_evidence.sample_count,
                face_recognition_method=face_evidence.method,
                dominant_track_id=face_evidence.dominant_track_id,
                summary="We could not verify a stable patient face yet.",
                recommended_action="Center only the patient in the camera, improve lighting, and start again.",
                reasons=self._dedupe(reasons),
                quality_flags=face_evidence.quality_flags,
            )

        if not has_enrolled_face:
            had_cleared_face_profile = self._profile_requires_reenrollment(profile)
            logger.info(
                "Identity preflight patient_id=%s status=unenrolled sample_count=%s method=%s canonical_method=%s canonical_dimensions=%s",
                patient_id,
                face_evidence.sample_count,
                face_evidence.method,
                profile.canonical_face_recognition_method,
                len(profile.canonical_face_embedding),
            )
            if had_cleared_face_profile:
                reasons.append("The stored face template is no longer trusted and must be re-enrolled with the newer face encoder.")
            else:
                reasons.append("No canonical face has been enrolled for this patient ID yet.")
            reasons.append("This session can proceed and establish the initial face reference after upload.")
            return IdentityPreflightResult(
                patient_id=patient_id,
                status="unenrolled",
                can_start_session=True,
                requires_reenrollment=had_cleared_face_profile,
                enrolled_face_exists=False,
                face_sample_count=face_evidence.sample_count,
                face_recognition_method=face_evidence.method,
                dominant_track_id=face_evidence.dominant_track_id,
                summary=(
                    "This patient ID needs a fresh face enrollment."
                    if had_cleared_face_profile
                    else "This patient ID has no enrolled face yet."
                ),
                recommended_action="Continue only if this is the intended patient; this session will create the initial face reference.",
                reasons=self._dedupe(reasons),
                quality_flags=face_evidence.quality_flags,
            )

        face_match_confidence = self._face_similarity(
            face_evidence.descriptor,
            profile.canonical_face_embedding,
            left_method=face_evidence.method,
            right_method=profile.canonical_face_recognition_method,
        )
        if face_match_confidence >= self.FACE_MATCH_INCLUDE_THRESHOLD:
            logger.info(
                "Identity preflight patient_id=%s status=verified face_match=%.4f include_threshold=%.2f review_threshold=%.2f exclude_threshold=%.2f sample_count=%s method=%s canonical_method=%s dimensions=%s/%s track_id=%s",
                patient_id,
                face_match_confidence,
                self.FACE_MATCH_INCLUDE_THRESHOLD,
                self.FACE_MATCH_REVIEW_THRESHOLD,
                self.FACE_MATCH_EXCLUDE_THRESHOLD,
                face_evidence.sample_count,
                face_evidence.method,
                profile.canonical_face_recognition_method,
                len(face_evidence.descriptor),
                len(profile.canonical_face_embedding),
                face_evidence.dominant_track_id,
            )
            reasons.append("The opening face sample matches the enrolled patient face profile.")
            return IdentityPreflightResult(
                patient_id=patient_id,
                status="verified",
                can_start_session=True,
                enrolled_face_exists=True,
                face_match_confidence=round(face_match_confidence, 2),
                face_sample_count=face_evidence.sample_count,
                face_recognition_method=face_evidence.method,
                dominant_track_id=face_evidence.dominant_track_id,
                summary="Patient identity verified for realtime intake.",
                recommended_action="Continue with the structured realtime session.",
                reasons=self._dedupe(reasons),
                quality_flags=face_evidence.quality_flags,
            )

        if face_match_confidence < self.FACE_MATCH_EXCLUDE_THRESHOLD:
            logger.info(
                "Identity preflight patient_id=%s status=mismatch face_match=%.4f include_threshold=%.2f review_threshold=%.2f exclude_threshold=%.2f sample_count=%s method=%s canonical_method=%s dimensions=%s/%s track_id=%s",
                patient_id,
                face_match_confidence,
                self.FACE_MATCH_INCLUDE_THRESHOLD,
                self.FACE_MATCH_REVIEW_THRESHOLD,
                self.FACE_MATCH_EXCLUDE_THRESHOLD,
                face_evidence.sample_count,
                face_evidence.method,
                profile.canonical_face_recognition_method,
                len(face_evidence.descriptor),
                len(profile.canonical_face_embedding),
                face_evidence.dominant_track_id,
            )
            reasons.append("The opening face sample does not match the enrolled patient face profile.")
            return IdentityPreflightResult(
                patient_id=patient_id,
                status="mismatch",
                can_start_session=False,
                requires_patient_reentry=True,
                enrolled_face_exists=True,
                face_match_confidence=round(face_match_confidence, 2),
                face_sample_count=face_evidence.sample_count,
                face_recognition_method=face_evidence.method,
                dominant_track_id=face_evidence.dominant_track_id,
                summary="The current face does not appear to match this patient ID.",
                recommended_action="Please confirm the patient and re-enter the correct patient information and patient ID before starting.",
                reasons=self._dedupe(reasons),
                quality_flags=face_evidence.quality_flags,
            )

        logger.info(
            "Identity preflight patient_id=%s status=needs-retry face_match=%.4f include_threshold=%.2f review_threshold=%.2f exclude_threshold=%.2f sample_count=%s method=%s canonical_method=%s dimensions=%s/%s track_id=%s",
            patient_id,
            face_match_confidence,
            self.FACE_MATCH_INCLUDE_THRESHOLD,
            self.FACE_MATCH_REVIEW_THRESHOLD,
            self.FACE_MATCH_EXCLUDE_THRESHOLD,
            face_evidence.sample_count,
            face_evidence.method,
            profile.canonical_face_recognition_method,
            len(face_evidence.descriptor),
            len(profile.canonical_face_embedding),
            face_evidence.dominant_track_id,
        )
        reasons.append("The opening face sample is close to the enrolled patient face but not strong enough to verify automatically.")
        return IdentityPreflightResult(
            patient_id=patient_id,
            status="needs-retry",
            can_start_session=False,
            enrolled_face_exists=True,
            face_match_confidence=round(face_match_confidence, 2),
            face_sample_count=face_evidence.sample_count,
            face_recognition_method=face_evidence.method,
            dominant_track_id=face_evidence.dominant_track_id,
            summary="We need a clearer opening face sample before starting.",
            recommended_action="Ask only the patient to face the camera and try the realtime start again.",
            reasons=self._dedupe(reasons),
            quality_flags=face_evidence.quality_flags,
        )

    def link_assessment(
        self,
        assessment: ClinicAssessment,
        *,
        session_record: dict[str, Any] | None = None,
        media: PreparedMedia | None = None,
    ) -> IdentityLinkRecord:
        profile = self.load_profile(assessment.patient_id)
        resolved_media = media or self._load_prepared_media(assessment.assessment_id)
        face_evidence = self._extract_face_evidence(resolved_media)

        session_id = self._string_value(
            self._read_path(session_record, "sessionId", default=assessment.assessment_id)
        )
        transcript_turns = self._read_path(
            session_record,
            "derivedFeatures",
            "speech",
            "transcriptTurns",
            default=[],
        )
        patient_texts = [
            self._string_value(turn.get("text"))
            for turn in transcript_turns
            if isinstance(turn, dict) and self._string_value(turn.get("role")).lower() == "patient"
        ]
        patient_text = " ".join(text for text in patient_texts if text).strip()
        intro_text = patient_texts[0] if patient_texts else patient_text[:96]

        face_detection_rate = self._optional_float(
            self._read_path(session_record, "derivedFeatures", "facial", "faceDetectionRate")
        )
        average_face_area = self._optional_float(
            self._read_path(session_record, "derivedFeatures", "facial", "averageFaceArea")
        )
        if face_evidence.detection_rate is not None:
            face_detection_rate = max(face_detection_rate or 0.0, face_evidence.detection_rate)
        if face_evidence.average_face_area is not None:
            average_face_area = max(average_face_area or 0.0, face_evidence.average_face_area)

        motion_intensity = self._safe_float(
            self._read_path(session_record, "derivedFeatures", "interactionTiming", "motionIntensity", default=0.28),
            default=0.28,
        )
        mean_brightness = self._safe_float(
            self._read_path(session_record, "derivedFeatures", "interactionTiming", "meanBrightness", default=0.55),
            default=0.55,
        )
        patient_turns = int(
            self._safe_float(
                self._read_path(session_record, "derivedFeatures", "task", "patientTurns", default=0),
                default=0.0,
            )
        )
        utterance_count = int(
            self._safe_float(
                self._read_path(
                    session_record,
                    "derivedFeatures",
                    "speech",
                    "utteranceCount",
                    default=len(patient_texts),
                ),
                default=float(len(patient_texts)),
            )
        )

        presence_score = self._presence_score(assessment.target_patient_presence)
        turn_share = self._clamp(patient_turns / max(1.0, float(max(utterance_count, patient_turns, 1))))
        face_visibility = self._clamp(face_detection_rate if face_detection_rate is not None else presence_score)
        face_stability = self._clamp(
            0.68
            + (min(average_face_area or 0.16, 0.30) * 0.75)
            - (abs(motion_intensity - 0.30) * 0.80)
            + (0.06 if face_evidence.sample_count >= self.FACE_MIN_SAMPLE_COUNT else 0.0)
            + (0.05 if assessment.target_patient_presence == "clear" else 0.0)
        )
        conversation_is_single = 1.0 if assessment.speaker_structure != "multi_speaker" else 0.52
        within_session = self._clamp(
            (presence_score * 0.28)
            + (face_visibility * 0.24)
            + (face_stability * 0.18)
            + (turn_share * 0.16)
            + (conversation_is_single * 0.14)
            - (0.12 if assessment.target_patient_presence == "uncertain" else 0.0)
            - (0.24 if assessment.target_patient_presence == "absent" else 0.0)
        )

        face_embedding = list(face_evidence.descriptor)
        voice_embedding = self._embedding_from_components(
            text_tokens=self._tokens(patient_text)[:18] + [self._string_value(assessment.language).lower()],
            metrics=[
                self._clamp(turn_share),
                self._safe_float(assessment.language_confidence, default=0.65),
                self._safe_float(assessment.reviewer_confidence, default=0.68),
            ],
        )
        dominant_track_id = face_evidence.dominant_track_id

        has_enrolled_face = self._has_comparable_enrolled_face(profile, current_method=face_evidence.method)
        has_current_face = face_evidence.sample_count >= self.FACE_MIN_SAMPLE_COUNT and bool(face_embedding)

        face_match_confidence = None
        voice_match_confidence = None
        reasons: list[str] = []

        if has_enrolled_face and has_current_face:
            face_match_confidence = self._face_similarity(
                face_embedding,
                profile.canonical_face_embedding,
                left_method=face_evidence.method,
                right_method=profile.canonical_face_recognition_method,
            )
        elif not has_enrolled_face and has_current_face:
            face_match_confidence = self._clamp(0.74 + ((within_session - 0.5) * 0.28))
            reasons.append(
                "No enrolled face existed yet, so this session established the initial patient face reference."
            )

        if profile.canonical_voice_embedding and voice_embedding:
            voice_match_confidence = self._cosine_similarity(voice_embedding, profile.canonical_voice_embedding)
        elif voice_embedding:
            voice_match_confidence = self._clamp(0.66 + ((within_session - 0.5) * 0.20))

        if assessment.target_patient_presence == "absent":
            reasons.append("The assessment did not identify the patient as visibly present.")
        elif assessment.target_patient_presence == "uncertain":
            reasons.append("Target-patient presence was uncertain in this session.")
        else:
            reasons.append("A primary patient track was visible for most of the session.")

        if assessment.speaker_structure == "multi_speaker":
            reasons.append("Multiple speakers were detected, which increases attribution risk.")
        if turn_share < 0.45:
            reasons.append("The patient contributed a limited share of the captured turns.")
        if has_current_face:
            reasons.append(
                f"Extracted a face descriptor from {face_evidence.sample_count} sampled frame(s) for identity matching."
            )
        elif face_evidence.sample_count > 0:
            reasons.append("Only a small number of usable face samples were extracted from the video.")
        else:
            reasons.append("No stable patient face could be extracted from the video for automatic matching.")

        if face_match_confidence is not None and face_match_confidence >= self.FACE_MATCH_INCLUDE_THRESHOLD:
            reasons.append("The current face descriptor stays close to the enrolled patient face profile.")
        elif has_enrolled_face and face_match_confidence is not None:
            reasons.append("The current face descriptor diverges from the enrolled patient face profile.")
        if voice_match_confidence is not None and voice_match_confidence >= self.PROBABLE_THRESHOLD:
            reasons.append("Speech cadence and lexical patterns stay close to prior sessions.")

        quality_flags = normalize_quality_flags(
            [
                *assessment.quality_flags,
                *(
                    self._string_value(flag)
                    for flag in self._read_path(session_record, "qualityControl", "flags", default=[])
                    if self._string_value(flag)
                ),
                *face_evidence.quality_flags,
            ]
        )

        if has_current_face:
            final_confidence = self._clamp(
                (within_session * 0.24)
                + ((face_match_confidence or 0.5) * 0.60)
                + ((voice_match_confidence or 0.5) * 0.16)
            )
        else:
            final_confidence = self._clamp(
                (within_session * 0.68)
                + ((voice_match_confidence or 0.5) * 0.32)
            )

        linkage_verdict: str
        timeline_inclusion: str
        if has_enrolled_face and not has_current_face:
            linkage_verdict = "no-face"
            timeline_inclusion = "manual-review"
            reasons.append("An enrolled patient face exists, but this session could not verify it automatically.")
        elif has_enrolled_face and face_match_confidence is not None:
            if face_match_confidence < self.FACE_MATCH_EXCLUDE_THRESHOLD:
                linkage_verdict = "mismatch"
                timeline_inclusion = "exclude"
                reasons.append("Face matching indicates this session belongs to a different person.")
            elif face_match_confidence < self.FACE_MATCH_REVIEW_THRESHOLD:
                linkage_verdict = "uncertain"
                timeline_inclusion = "manual-review"
                reasons.append("Face matching was too weak for automatic patient linkage.")
            elif assessment.target_patient_presence == "absent" or within_session <= 0.24:
                linkage_verdict = "mismatch"
                timeline_inclusion = "exclude"
            elif assessment.target_patient_presence == "uncertain" or (
                assessment.speaker_structure == "multi_speaker" and turn_share < 0.45
            ):
                linkage_verdict = "uncertain"
                timeline_inclusion = "manual-review"
            elif face_match_confidence >= self.FACE_MATCH_INCLUDE_THRESHOLD and within_session >= 0.50:
                linkage_verdict = "confirmed" if final_confidence >= self.CONFIRMED_THRESHOLD else "probable"
                timeline_inclusion = "include"
            else:
                linkage_verdict = "uncertain"
                timeline_inclusion = "manual-review"
        else:
            if assessment.target_patient_presence == "absent" or within_session <= 0.24:
                linkage_verdict = "no-face" if not face_embedding else "mismatch"
                timeline_inclusion = "exclude"
            elif assessment.target_patient_presence == "uncertain" or (
                assessment.speaker_structure == "multi_speaker" and turn_share < 0.45
            ):
                linkage_verdict = "uncertain"
                timeline_inclusion = "manual-review"
            elif has_current_face and final_confidence >= self.CONFIRMED_THRESHOLD and within_session >= 0.58:
                linkage_verdict = "confirmed"
                timeline_inclusion = "include"
            elif has_current_face and final_confidence >= self.PROBABLE_THRESHOLD and within_session >= 0.52:
                linkage_verdict = "probable"
                timeline_inclusion = "include"
            elif final_confidence >= self.UNCERTAIN_THRESHOLD and within_session >= 0.42:
                linkage_verdict = "uncertain"
                timeline_inclusion = "manual-review"
            else:
                linkage_verdict = "mismatch"
                timeline_inclusion = "exclude"

        if timeline_inclusion == "include" and not has_current_face:
            linkage_verdict = "uncertain"
            timeline_inclusion = "manual-review"
            reasons.append("Automatic inclusion requires a verifiable face descriptor from the video.")

        link = IdentityLinkRecord(
            link_id=f"{assessment.assessment_id}-identity-link",
            patient_id=assessment.patient_id,
            assessment_id=assessment.assessment_id,
            session_id=session_id or assessment.assessment_id,
            assessed_at=utc_now(),
            enrollment_profile_id=profile.profile_id,
            target_presence=assessment.target_patient_presence or "uncertain",
            dominant_track_id=dominant_track_id,
            within_session_attribution_confidence=round(within_session, 2),
            face_sample_count=face_evidence.sample_count,
            face_recognition_method=face_evidence.method,
            face_match_confidence=round(face_match_confidence, 2) if face_match_confidence is not None else None,
            voice_match_confidence=round(voice_match_confidence, 2) if voice_match_confidence is not None else None,
            final_linkage_confidence=round(final_confidence, 2),
            linkage_verdict=linkage_verdict,  # type: ignore[arg-type]
            timeline_inclusion=timeline_inclusion,  # type: ignore[arg-type]
            reasons=self._dedupe(reasons),
            quality_flags=quality_flags,
            face_embedding=face_embedding,
            voice_embedding=voice_embedding,
        )

        logger.info(
            "Identity link assessment_id=%s patient_id=%s inclusion=%s verdict=%s face_match=%s final_confidence=%.4f within_session=%.4f sample_count=%s enrolled_face=%s current_face=%s method=%s canonical_method=%s dimensions=%s/%s track_id=%s",
            assessment.assessment_id,
            assessment.patient_id,
            timeline_inclusion,
            linkage_verdict,
            f"{face_match_confidence:.4f}" if face_match_confidence is not None else "none",
            final_confidence,
            within_session,
            face_evidence.sample_count,
            has_enrolled_face,
            has_current_face,
            face_evidence.method,
            profile.canonical_face_recognition_method,
            len(face_embedding),
            len(profile.canonical_face_embedding),
            dominant_track_id,
        )

        updated_profile = self._update_profile(profile, link)
        updated_profile = self._update_profile_memory(updated_profile, session_record=session_record)
        self.storage.save_identity_profile(updated_profile)
        self.storage.save_identity_link(link)
        return link

    def _update_profile(self, profile: IdentityProfile, link: IdentityLinkRecord) -> IdentityProfile:
        sessions_linked = profile.sessions_linked + 1
        sessions_included = profile.sessions_included + (1 if link.timeline_inclusion == "include" else 0)
        sessions_manual_review = profile.sessions_manual_review + (
            1 if link.timeline_inclusion == "manual-review" else 0
        )
        sessions_excluded = profile.sessions_excluded + (1 if link.timeline_inclusion == "exclude" else 0)

        canonical_face_embedding = profile.canonical_face_embedding
        canonical_face_recognition_method = profile.canonical_face_recognition_method
        canonical_voice_embedding = profile.canonical_voice_embedding
        enrollment_assessment_id = profile.enrollment_assessment_id
        notes = list(profile.notes)

        if link.timeline_inclusion == "include":
            if not enrollment_assessment_id:
                enrollment_assessment_id = link.assessment_id
                notes.append("Initial identity enrollment was created from the first includable session.")
            if link.face_sample_count > 0 and link.face_embedding:
                if not profile.canonical_face_embedding or (
                    profile.canonical_face_recognition_method
                    and link.face_recognition_method
                    and profile.canonical_face_recognition_method != link.face_recognition_method
                ):
                    canonical_face_embedding = list(link.face_embedding)
                    canonical_face_recognition_method = link.face_recognition_method
                elif self._should_refresh_canonical_face(link):
                    canonical_face_embedding = self._blend_embeddings(
                        profile.canonical_face_embedding,
                        link.face_embedding,
                    )
                    canonical_face_recognition_method = link.face_recognition_method or canonical_face_recognition_method
            if link.voice_embedding:
                canonical_voice_embedding = self._blend_embeddings(
                    profile.canonical_voice_embedding,
                    link.voice_embedding,
                )
        elif link.timeline_inclusion == "manual-review":
            notes.append("One or more sessions were held for identity review before longitudinal aggregation.")
        else:
            notes.append("At least one session was excluded from the patient timeline due to identity mismatch risk.")

        return profile.model_copy(
            update={
                "updated_at": utc_now(),
                "enrollment_assessment_id": enrollment_assessment_id,
                "preferred_name": profile.preferred_name,
                "memory": list(profile.memory),
                "canonical_face_embedding": canonical_face_embedding,
                "canonical_face_recognition_method": canonical_face_recognition_method,
                "canonical_voice_embedding": canonical_voice_embedding,
                "sessions_linked": sessions_linked,
                "sessions_included": sessions_included,
                "sessions_manual_review": sessions_manual_review,
                "sessions_excluded": sessions_excluded,
                "latest_link_id": link.link_id,
                "latest_linkage_verdict": link.linkage_verdict,
                "latest_linkage_confidence": link.final_linkage_confidence,
                "notes": self._dedupe(notes),
            }
        )

    def _update_profile_memory(
        self,
        profile: IdentityProfile,
        *,
        session_record: dict[str, Any] | None,
    ) -> IdentityProfile:
        preferred_name, memory = build_patient_memory(session_record)
        updates: dict[str, Any] = {}
        if preferred_name and preferred_name != profile.preferred_name:
            updates["preferred_name"] = preferred_name
        if memory:
            updates["memory"] = memory
        if not updates:
            return profile
        updates["updated_at"] = utc_now()
        return profile.model_copy(update=updates)

    def _load_prepared_media(self, assessment_id: str) -> PreparedMedia | None:
        prepared_dir = self.storage.settings.prepared_dir / assessment_id
        standardized_path = prepared_dir / "standardized.mp4"
        if not standardized_path.exists():
            return None
        frame_paths = [str(path) for path in sorted(prepared_dir.glob("frame-*.jpg"))]
        audio_path = prepared_dir / "audio.wav"
        return PreparedMedia(
            original_path=str(standardized_path),
            standardized_path=str(standardized_path),
            mime_type="video/mp4",
            size_bytes=standardized_path.stat().st_size,
            extracted_audio_path=str(audio_path) if audio_path.exists() else None,
            frame_paths=frame_paths,
        )

    def _extract_face_evidence(self, media: PreparedMedia | None) -> FaceEvidence:
        if cv2 is None or np is None or local_binary_pattern is None or hog is None or sobel is None:
            return FaceEvidence(quality_flags=["face_identity_runtime_unavailable"])
        if media is None:
            return FaceEvidence(quality_flags=["face_media_unavailable"])

        frames = self._load_frames(media)
        return self._extract_face_evidence_from_frames(frames)

    def _extract_face_evidence_from_frames(self, frames: list[Any]) -> FaceEvidence:
        if cv2 is None or np is None or local_binary_pattern is None or hog is None or sobel is None:
            return FaceEvidence(quality_flags=["face_identity_runtime_unavailable"])
        if not frames:
            return FaceEvidence(quality_flags=["face_frames_unavailable"])

        face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        profile_cascade = self._load_cascade("haarcascade_profileface.xml")
        if face_cascade is None:
            return FaceEvidence(quality_flags=["face_detector_unavailable"])

        descriptors: list[Any] = []
        area_samples: list[float] = []
        previous_box: tuple[int, int, int, int] | None = None
        for frame in frames:
            selection = self._select_primary_face(
                frame,
                face_cascade=face_cascade,
                profile_cascade=profile_cascade,
                previous_box=previous_box,
            )
            if selection is None:
                continue
            crop, box, area_ratio = selection
            descriptor = self._describe_face_crop(crop)
            if descriptor is None:
                continue
            descriptors.append(descriptor)
            area_samples.append(area_ratio)
            previous_box = box

        sample_count = len(descriptors)
        detection_rate = sample_count / max(1, len(frames))
        quality_flags: list[str] = []
        if sample_count == 0:
            quality_flags.append("face_identity_no_detected_face")
            return FaceEvidence(
                sample_count=0,
                detection_rate=detection_rate,
                quality_flags=quality_flags,
            )
        if sample_count < self.FACE_MIN_SAMPLE_COUNT:
            quality_flags.append("face_identity_samples_low")
        if detection_rate < 0.25:
            quality_flags.append("face_identity_unstable")

        averaged = np.mean(np.stack(descriptors, axis=0), axis=0)
        normalized = self._normalize_vector(averaged)
        if normalized is None:
            quality_flags.append("face_identity_descriptor_unavailable")
            return FaceEvidence(
                sample_count=sample_count,
                detection_rate=detection_rate,
                average_face_area=(sum(area_samples) / len(area_samples)) if area_samples else None,
                quality_flags=quality_flags,
            )
        dominant_track_id = "primary-face-" + hashlib.sha1(
            ",".join(f"{value:.5f}" for value in normalized[:24]).encode("utf-8")
        ).hexdigest()[:8]
        return FaceEvidence(
            descriptor=[round(float(value), 4) for value in normalized],
            sample_count=sample_count,
            detection_rate=self._clamp(detection_rate),
            average_face_area=self._clamp(sum(area_samples) / len(area_samples)) if area_samples else None,
            dominant_track_id=dominant_track_id,
            method=self.FACE_ENCODER_METHOD,
            quality_flags=quality_flags,
        )

    def _decode_base64_frames(self, image_base64_samples: list[str]) -> list[Any]:
        if cv2 is None or np is None:
            return []

        frames: list[Any] = []
        for sample in image_base64_samples[: self.MAX_FACE_SAMPLES]:
            encoded = str(sample or "").strip()
            if not encoded:
                continue
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                raw = base64.b64decode(encoded, validate=False)
            except (ValueError, TypeError):
                continue
            if not raw:
                continue
            buffer = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        return frames

    def _load_frames(self, media: PreparedMedia) -> list[Any]:
        if cv2 is None or np is None:
            return []

        frames: list[Any] = []
        for frame_path in media.frame_paths:
            path = Path(frame_path)
            if not path.exists():
                continue
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
        if frames:
            return frames[: self.MAX_FACE_SAMPLES]

        video_path = Path(media.standardized_path)
        if not video_path.exists():
            return []

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return []
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0:
                positions = sorted(
                    {
                        int(round(value))
                        for value in np.linspace(
                            0,
                            max(total_frames - 1, 0),
                            num=min(self.MAX_FACE_SAMPLES, total_frames),
                        )
                    }
                )
                for position in positions:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        frames.append(frame)
            else:
                read_count = 0
                while len(frames) < self.MAX_FACE_SAMPLES:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    if read_count % 15 == 0:
                        frames.append(frame)
                    read_count += 1
        finally:
            capture.release()
        return frames

    def _load_cascade(self, filename: str):
        if cv2 is None:
            return None
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / filename))
        return None if cascade.empty() else cascade

    def _select_primary_face(
        self,
        frame: Any,
        *,
        face_cascade,
        profile_cascade,
        previous_box: tuple[int, int, int, int] | None,
    ) -> tuple[Any, tuple[int, int, int, int], float] | None:
        if cv2 is None:
            return None
        gray = frame
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frontal = list(
            face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )
        )
        profile = (
            list(
                profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(48, 48),
                )
            )
            if profile_cascade is not None
            else []
        )
        candidates = frontal + profile
        if not candidates:
            return None

        frame_height, frame_width = gray.shape[:2]
        center_x = frame_width / 2
        center_y = frame_height / 2

        def score(rect: tuple[int, int, int, int]) -> float:
            x, y, width, height = rect
            area_score = (width * height) / max(1.0, float(frame_width * frame_height))
            cx = x + (width / 2)
            cy = y + (height / 2)
            center_penalty = (((cx - center_x) / frame_width) ** 2) + (((cy - center_y) / frame_height) ** 2)
            continuity_penalty = 0.0
            if previous_box is not None:
                px, py, pwidth, pheight = previous_box
                pcx = px + (pwidth / 2)
                pcy = py + (pheight / 2)
                continuity_penalty = (
                    (((cx - pcx) / frame_width) ** 2) + (((cy - pcy) / frame_height) ** 2)
                ) * 0.8
            return (area_score * 2.8) - (center_penalty * 0.7) - continuity_penalty

        best = max(candidates, key=score)
        x, y, width, height = [int(value) for value in best]
        margin_x = int(width * 0.18)
        margin_y = int(height * 0.18)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(frame.shape[1], x + width + margin_x)
        bottom = min(frame.shape[0], y + height + margin_y)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        area_ratio = (width * height) / max(1.0, float(frame.shape[0] * frame.shape[1]))
        return crop, (x, y, width, height), float(area_ratio)

    def _describe_face_crop(self, crop: Any):
        if cv2 is None or np is None or local_binary_pattern is None or hog is None or sobel is None:
            return None
        gray = crop
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.FACE_DESCRIPTOR_SIZE, self.FACE_DESCRIPTOR_SIZE))
        gray = cv2.equalizeHist(gray)
        gray_float = gray.astype("float32") / 255.0

        hog_descriptor = hog(
            gray_float,
            orientations=self.FACE_HOG_ORIENTATIONS,
            pixels_per_cell=(self.FACE_HOG_PIXELS_PER_CELL, self.FACE_HOG_PIXELS_PER_CELL),
            cells_per_block=(self.FACE_HOG_CELLS_PER_BLOCK, self.FACE_HOG_CELLS_PER_BLOCK),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype("float32")

        lbp = local_binary_pattern(
            gray,
            self.FACE_DESCRIPTOR_POINTS,
            self.FACE_DESCRIPTOR_RADIUS,
            method="uniform",
        )
        cell_size = self.FACE_DESCRIPTOR_SIZE // self.FACE_DESCRIPTOR_GRID
        descriptor_parts: list[Any] = []
        for row in range(self.FACE_DESCRIPTOR_GRID):
            for col in range(self.FACE_DESCRIPTOR_GRID):
                region = lbp[
                    row * cell_size : (row + 1) * cell_size,
                    col * cell_size : (col + 1) * cell_size,
                ]
                histogram, _ = np.histogram(
                    region.ravel(),
                    bins=np.arange(0, self.FACE_DESCRIPTOR_BINS + 1),
                    range=(0, self.FACE_DESCRIPTOR_BINS),
                )
                histogram = histogram.astype("float32")
                histogram /= max(histogram.sum(), 1.0)
                descriptor_parts.append(histogram)

        low_res = cv2.resize(
            gray_float,
            (self.FACE_DESCRIPTOR_LOW_RES, self.FACE_DESCRIPTOR_LOW_RES),
            interpolation=cv2.INTER_AREA,
        ).astype("float32")
        low_res = low_res - float(low_res.mean())
        low_res_std = float(low_res.std())
        if low_res_std > 1e-6:
            low_res = low_res / low_res_std

        edge_map = sobel(gray_float).astype("float32")
        edge_low_res = cv2.resize(
            edge_map,
            (self.FACE_DESCRIPTOR_PROJECTION_BINS, self.FACE_DESCRIPTOR_PROJECTION_BINS),
            interpolation=cv2.INTER_AREA,
        ).astype("float32")
        row_projection = edge_low_res.mean(axis=1)
        col_projection = edge_low_res.mean(axis=0)
        projections = np.concatenate([row_projection, col_projection], axis=0)
        projections = projections - float(projections.mean())
        projection_std = float(projections.std())
        if projection_std > 1e-6:
            projections = projections / projection_std

        normalized_hog = self._normalize_vector(hog_descriptor)
        normalized_lbp = self._normalize_vector(np.concatenate(descriptor_parts, axis=0))
        normalized_low_res = self._normalize_vector(low_res.ravel())
        normalized_projection = self._normalize_vector(projections)
        if (
            normalized_hog is None
            or normalized_lbp is None
            or normalized_low_res is None
            or normalized_projection is None
        ):
            return None

        descriptor = np.concatenate(
            [
                normalized_hog,
                normalized_lbp,
                normalized_low_res,
                normalized_projection,
            ],
            axis=0,
        )
        return self._normalize_vector(descriptor)

    def _normalize_vector(self, values: Any):
        if np is None:
            return None
        array = np.asarray(values, dtype="float32")
        norm = float(np.linalg.norm(array))
        if norm <= 1e-8:
            return None
        return array / norm

    def _normalize_profile(self, profile: IdentityProfile) -> IdentityProfile:
        if not profile.canonical_face_embedding:
            return profile
        if self._is_legacy_face_profile(profile):
            return self._clear_face_enrollment(
                profile,
                reason=(
                    "Legacy face profile was cleared because the previous embedding format was too permissive "
                    "and must be re-enrolled with the current face encoder."
                ),
            )
        return profile

    def _is_legacy_face_profile(self, profile: IdentityProfile) -> bool:
        method = (profile.canonical_face_recognition_method or "").strip().lower()
        dimensions = len(profile.canonical_face_embedding)
        if method in {item.lower() for item in self.LEGACY_FACE_ENCODER_METHODS}:
            return True
        if not method and dimensions == 176:
            return True
        return False

    def _profile_requires_reenrollment(self, profile: IdentityProfile) -> bool:
        return any("re-enrolled" in note.lower() or "re-enroll" in note.lower() for note in profile.notes)

    def _clear_face_enrollment(self, profile: IdentityProfile, *, reason: str) -> IdentityProfile:
        notes = list(profile.notes)
        notes.append(reason)
        return profile.model_copy(
            update={
                "updated_at": utc_now(),
                "enrollment_assessment_id": None,
                "canonical_face_embedding": [],
                "canonical_face_recognition_method": None,
                "notes": self._dedupe(notes),
            }
        )

    def _has_comparable_enrolled_face(self, profile: IdentityProfile, *, current_method: str | None) -> bool:
        if not profile.canonical_face_embedding:
            return False
        if self._is_legacy_face_profile(profile):
            return False
        if not profile.canonical_face_recognition_method:
            return False
        if not current_method:
            return True
        return profile.canonical_face_recognition_method == current_method

    def _face_descriptor_section_lengths(self) -> tuple[int, int, int, int]:
        cells_per_axis = self.FACE_DESCRIPTOR_SIZE // self.FACE_HOG_PIXELS_PER_CELL
        blocks_per_axis = cells_per_axis - self.FACE_HOG_CELLS_PER_BLOCK + 1
        hog_length = (
            blocks_per_axis
            * blocks_per_axis
            * self.FACE_HOG_CELLS_PER_BLOCK
            * self.FACE_HOG_CELLS_PER_BLOCK
            * self.FACE_HOG_ORIENTATIONS
        )
        lbp_length = (
            self.FACE_DESCRIPTOR_GRID
            * self.FACE_DESCRIPTOR_GRID
            * self.FACE_DESCRIPTOR_BINS
        )
        low_res_length = self.FACE_DESCRIPTOR_LOW_RES * self.FACE_DESCRIPTOR_LOW_RES
        projection_length = self.FACE_DESCRIPTOR_PROJECTION_BINS * 2
        return hog_length, lbp_length, low_res_length, projection_length

    def _split_face_descriptor(self, values: list[float]) -> tuple[Any, Any, Any, Any] | None:
        if np is None:
            return None
        hog_length, lbp_length, low_res_length, projection_length = self._face_descriptor_section_lengths()
        expected = hog_length + lbp_length + low_res_length + projection_length
        if len(values) != expected:
            return None
        array = np.asarray(values, dtype="float32")
        hog_end = hog_length
        lbp_end = hog_end + lbp_length
        low_res_end = lbp_end + low_res_length
        return (
            array[:hog_end],
            array[hog_end:lbp_end],
            array[lbp_end:low_res_end],
            array[low_res_end:expected],
        )

    def _array_cosine_similarity(self, left: Any, right: Any) -> float:
        if np is None:
            return 0.0
        left_array = np.asarray(left, dtype="float32")
        right_array = np.asarray(right, dtype="float32")
        left_norm = float(np.linalg.norm(left_array))
        right_norm = float(np.linalg.norm(right_array))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return 0.0
        raw_cosine = float(np.dot(left_array, right_array) / (left_norm * right_norm))
        return self._clamp((raw_cosine + 1.0) / 2.0)

    def _histogram_similarity(self, left: Any, right: Any) -> float:
        if np is None:
            return 0.0
        left_array = np.asarray(left, dtype="float32")
        right_array = np.asarray(right, dtype="float32")
        left_array = np.clip(left_array, 0.0, None)
        right_array = np.clip(right_array, 0.0, None)
        left_sum = float(left_array.sum())
        right_sum = float(right_array.sum())
        if left_sum <= 1e-8 or right_sum <= 1e-8:
            return 0.0
        left_hist = left_array / left_sum
        right_hist = right_array / right_sum
        bhattacharyya = float(np.sum(np.sqrt(left_hist * right_hist)))
        hellinger_similarity = 1.0 - math.sqrt(max(0.0, 1.0 - bhattacharyya))
        cosine = self._array_cosine_similarity(left_hist, right_hist)
        return self._clamp((hellinger_similarity * 0.72) + (cosine * 0.28))

    def _generic_face_similarity(self, left: list[float], right: list[float]) -> float:
        if np is None or not left or not right:
            return 0.0
        length = min(len(left), len(right))
        if length <= 0:
            return 0.0
        return self._array_cosine_similarity(left[:length], right[:length])

    def _blend_embeddings(self, current: list[float], incoming: list[float]) -> list[float]:
        if not incoming:
            return current
        if not current:
            return incoming
        length = min(len(current), len(incoming))
        return [
            round((current[index] * 0.72) + (incoming[index] * 0.28), 4)
            for index in range(length)
        ]

    def _should_refresh_canonical_face(self, link: IdentityLinkRecord) -> bool:
        return bool(
            link.face_embedding
            and link.face_sample_count >= self.FACE_CANONICAL_REFRESH_MIN_SAMPLES
            and (link.face_match_confidence or 0.0) >= self.FACE_CANONICAL_REFRESH_THRESHOLD
        )

    def _embedding_from_components(self, *, text_tokens: list[str], metrics: list[float]) -> list[float]:
        if not text_tokens and not metrics:
            return []
        vector = [0.0] * self.EMBEDDING_DIM
        for token in text_tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.EMBEDDING_DIM):
                sign = 1.0 if digest[index] % 2 == 0 else -1.0
                weight = 0.35 + (digest[index + self.EMBEDDING_DIM] / 255.0) * 0.65
                vector[index] += sign * weight
        for index, metric in enumerate(metrics):
            vector[index % self.EMBEDDING_DIM] += (self._clamp(metric) - 0.5) * 2.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 4) for value in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        dot = sum(left[index] * right[index] for index in range(length))
        left_norm = math.sqrt(sum(left[index] * left[index] for index in range(length))) or 1.0
        right_norm = math.sqrt(sum(right[index] * right[index] for index in range(length))) or 1.0
        return self._clamp((dot / (left_norm * right_norm) + 1.0) / 2.0)

    def _face_similarity(
        self,
        left: list[float],
        right: list[float],
        *,
        left_method: str | None = None,
        right_method: str | None = None,
    ) -> float:
        if not left or not right or np is None:
            return 0.0
        if left_method and right_method and left_method != right_method:
            return 0.0
        if left_method != self.FACE_ENCODER_METHOD or right_method != self.FACE_ENCODER_METHOD:
            return self._generic_face_similarity(left, right)

        split_left = self._split_face_descriptor(left)
        split_right = self._split_face_descriptor(right)
        if split_left is None or split_right is None:
            return self._generic_face_similarity(left, right)

        left_hog, left_lbp, left_low_res, left_projection = split_left
        right_hog, right_lbp, right_low_res, right_projection = split_right

        hog_score = self._array_cosine_similarity(left_hog, right_hog)
        low_res_score = self._array_cosine_similarity(left_low_res, right_low_res)
        lbp_score = self._histogram_similarity(left_lbp, right_lbp)
        projection_score = self._array_cosine_similarity(left_projection, right_projection)

        blended = (
            (hog_score * 0.42)
            + (low_res_score * 0.28)
            + (lbp_score * 0.18)
            + (projection_score * 0.12)
        )
        guard_score = min(hog_score, low_res_score)
        if guard_score < 0.72:
            blended -= (0.72 - guard_score) * 0.55
        if lbp_score < 0.68:
            blended -= (0.68 - lbp_score) * 0.20
        return self._clamp(blended)

    def _extract_name_tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        if "name is" in lowered:
            lowered = lowered.split("name is", 1)[1]
        elif "i am" in lowered:
            lowered = lowered.split("i am", 1)[1]
        elif "i'm" in lowered:
            lowered = lowered.split("i'm", 1)[1]
        tokens = self._tokens(lowered)
        return tokens[:4]

    def _tokens(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def _presence_score(self, value: str | None) -> float:
        mapping = {
            "clear": 0.88,
            "probable": 0.72,
            "uncertain": 0.42,
            "absent": 0.12,
        }
        return mapping.get(value or "", 0.64)

    def _quality_flag_reason(self, flag: str) -> str | None:
        mapping = {
            "face_frames_unavailable": "No usable opening camera frames reached the identity checker.",
            "face_identity_no_detected_face": "No face was detected in the opening identity frames.",
            "face_identity_samples_low": "Too few face samples were available for stable identity matching.",
            "face_identity_unstable": "The detected face moved too much across the sampled opening frames.",
            "face_identity_runtime_unavailable": "The local face-recognition runtime is unavailable on this deployment.",
        }
        return mapping.get(flag)

    def _read_path(
        self,
        payload: dict[str, Any] | None,
        *path: str,
        default: Any = None,
    ) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
        return default if current is None else current

    def _safe_float(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _optional_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _string_value(self, value: Any) -> str:
        return str(value).strip() if value is not None else ""

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    def _dedupe(self, items: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in items:
            stripped = item.strip()
            if stripped and stripped not in seen:
                cleaned.append(stripped)
                seen.add(stripped)
        return cleaned
