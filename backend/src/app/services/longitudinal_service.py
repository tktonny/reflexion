"""Longitudinal tracking service for per-patient trend and embedding state."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from typing import Any

from backend.src.app.models.assessment import ClinicAssessment, normalize_quality_flags, utc_now
from backend.src.app.models.identity import IdentityLinkRecord
from backend.src.app.models.longitudinal import (
    LongitudinalDirection,
    LongitudinalProfile,
    LongitudinalQualityControl,
    LongitudinalSnapshot,
    LongitudinalTrendPoint,
    TaskBehaviorFeatures,
    VisualFaceFeatures,
)
from backend.src.app.services.identity_service import IdentityLinkageService
from clinic.database.storage import LocalStorage


class LongitudinalTrackingService:
    """Persist and derive lightweight longitudinal state for demo and product surfaces."""

    MIN_TREND_POINTS = 4
    MAX_TREND_POINTS = 6
    BASELINE_WINDOW_DAYS = 90
    DECLINING_DELTA = 0.08
    IMPROVING_DELTA = -0.08
    EMBEDDING_DIM = 8

    def __init__(self, storage: LocalStorage) -> None:
        self.storage = storage
        self.identity = IdentityLinkageService(storage)

    def record_assessment(
        self,
        assessment: ClinicAssessment,
        *,
        session_record: dict[str, Any] | None = None,
        identity_link: IdentityLinkRecord | None = None,
    ) -> LongitudinalProfile:
        """Upsert one formal assessment into the patient's longitudinal profile."""
        profile = self.load_profile(assessment.patient_id)
        snapshots = [snapshot for snapshot in profile.snapshots if snapshot.assessment_id != assessment.assessment_id]
        resolved_identity = identity_link or self.identity.ensure_link(assessment, session_record=session_record)
        snapshot = self._build_snapshot(
            assessment,
            session_record=session_record,
            identity_link=resolved_identity,
        )
        snapshots.append(snapshot)
        self.storage.save_feature_snapshot(
            assessment.assessment_id,
            self._feature_snapshot_payload(snapshot, assessment),
        )
        updated = self._finalize_profile(
            profile.model_copy(
                update={
                    "snapshots": snapshots,
                }
            )
        )
        self.storage.save_longitudinal_profile(updated)
        return updated

    def build_profile(
        self,
        patient_id: str,
        assessments: list[ClinicAssessment],
    ) -> LongitudinalProfile:
        """Load the stored profile and backfill any missing assessments from history."""
        profile = self.load_profile(patient_id)
        snapshots_by_assessment = {
            snapshot.assessment_id: snapshot
            for snapshot in profile.snapshots
        }

        for assessment in sorted(assessments, key=lambda item: item.created_at):
            if assessment.assessment_id in snapshots_by_assessment:
                continue
            identity_link = self.identity.ensure_link(assessment, session_record=None)
            snapshot = self._build_snapshot(
                assessment,
                session_record=None,
                identity_link=identity_link,
            )
            snapshots_by_assessment[assessment.assessment_id] = snapshot
            self.storage.save_feature_snapshot(
                assessment.assessment_id,
                self._feature_snapshot_payload(snapshot, assessment),
            )

        if not snapshots_by_assessment:
            return self._empty_profile(patient_id)

        updated = self._finalize_profile(
            profile.model_copy(update={"snapshots": list(snapshots_by_assessment.values())})
        )
        self.storage.save_longitudinal_profile(updated)
        return updated

    def load_profile(self, patient_id: str) -> LongitudinalProfile:
        try:
            payload = self.storage.load_longitudinal_profile(patient_id)
        except FileNotFoundError:
            return self._empty_profile(patient_id)
        return self._finalize_profile(LongitudinalProfile.model_validate(payload))

    def _empty_profile(self, patient_id: str) -> LongitudinalProfile:
        return LongitudinalProfile(patient_id=patient_id, updated_at=utc_now())

    def _finalize_profile(self, profile: LongitudinalProfile) -> LongitudinalProfile:
        snapshots = sorted(profile.snapshots, key=lambda item: item.captured_at)
        if not snapshots:
            return profile.model_copy(update={"updated_at": utc_now()})

        included_snapshots = [snapshot for snapshot in snapshots if snapshot.timeline_inclusion == "include"]
        counts = self._snapshot_counts(snapshots)
        latest_snapshot = snapshots[-1]
        if not included_snapshots:
            return profile.model_copy(
                update={
                    "updated_at": utc_now(),
                    "total_sessions_seen": counts["total"],
                    "sessions_included": counts["include"],
                    "sessions_manual_review": counts["manual-review"],
                    "sessions_excluded": counts["exclude"],
                    "baseline_at": None,
                    "baseline_risk_score": None,
                    "latest_risk_score": round(latest_snapshot.risk_score, 2),
                    "direction": "stable",
                    "direction_label": "Stable",
                    "baseline_comparison": (
                        "Compared to your baseline (3 months ago), no longitudinal trend is shown yet because "
                        "recent sessions are still pending identity or quality review."
                    ),
                    "anomaly_summary": (
                        "Longitudinal tracking has not started because no session has passed identity and QC gating yet."
                    ),
                    "gating_summary": self._gating_summary(counts),
                    "x_axis_label": "Date",
                    "y_axis_label": "Risk Score",
                    "snapshots": snapshots,
                    "trend_points": [],
                }
            )

        trend_points = self._build_trend_points(included_snapshots)
        baseline_target_at = trend_points[-1].recorded_at - timedelta(days=self.BASELINE_WINDOW_DAYS)
        baseline_index = min(
            range(len(trend_points)),
            key=lambda index: abs((trend_points[index].recorded_at - baseline_target_at).total_seconds()),
        )
        trend_points = [
            point.model_copy(update={"is_baseline": index == baseline_index})
            for index, point in enumerate(trend_points)
        ]

        baseline_point = trend_points[baseline_index]
        latest_point = trend_points[-1]
        direction = self._direction_for_delta(latest_point.risk_score - baseline_point.risk_score)
        latest_included_snapshot = included_snapshots[-1]

        return profile.model_copy(
            update={
                "updated_at": utc_now(),
                "total_sessions_seen": counts["total"],
                "sessions_included": counts["include"],
                "sessions_manual_review": counts["manual-review"],
                "sessions_excluded": counts["exclude"],
                "baseline_at": baseline_point.recorded_at,
                "baseline_risk_score": round(baseline_point.risk_score, 2),
                "latest_risk_score": round(latest_included_snapshot.risk_score, 2),
                "direction": direction,
                "direction_label": direction.title(),
                "baseline_comparison": self._baseline_comparison_text(
                    baseline_score=baseline_point.risk_score,
                    latest_score=latest_point.risk_score,
                    direction=direction,
                ),
                "anomaly_summary": self._anomaly_summary(direction, latest_included_snapshot),
                "gating_summary": self._gating_summary(counts),
                "x_axis_label": "Date",
                "y_axis_label": "Risk Score",
                "snapshots": snapshots,
                "trend_points": trend_points,
            }
        )

    def _build_snapshot(
        self,
        assessment: ClinicAssessment,
        *,
        session_record: dict[str, Any] | None,
        identity_link: IdentityLinkRecord,
    ) -> LongitudinalSnapshot:
        transcript_turns = self._read_path(
            session_record,
            "derivedFeatures",
            "speech",
            "transcriptTurns",
            default=[],
        )
        transcript_text = " ".join(
            self._string_value(turn.get("text"))
            for turn in transcript_turns
            if isinstance(turn, dict)
        ).strip()
        completed_stages = self._read_path(
            session_record,
            "derivedFeatures",
            "task",
            "completedStages",
            default=[],
        )
        patient_turns = int(
            self._safe_float(
                self._read_path(
                    session_record,
                    "derivedFeatures",
                    "task",
                    "patientTurns",
                    default=0,
                ),
                default=0.0,
            )
        )
        speech_seconds = self._safe_float(
            self._read_path(
                session_record,
                "derivedFeatures",
                "speech",
                "speechSeconds",
                default=0.0,
            ),
            default=0.0,
        )
        average_turn_seconds = self._safe_float(
            self._read_path(
                session_record,
                "derivedFeatures",
                "speech",
                "averageTurnSeconds",
                default=0.0,
            ),
            default=0.0,
        )
        face_detection_rate = self._optional_float(
            self._read_path(
                session_record,
                "derivedFeatures",
                "facial",
                "faceDetectionRate",
            )
        )
        average_face_area = self._optional_float(
            self._read_path(
                session_record,
                "derivedFeatures",
                "facial",
                "averageFaceArea",
            )
        )
        motion_intensity = self._safe_float(
            self._read_path(
                session_record,
                "derivedFeatures",
                "interactionTiming",
                "motionIntensity",
                default=0.28,
            ),
            default=0.28,
        )
        mean_brightness = self._safe_float(
            self._read_path(
                session_record,
                "derivedFeatures",
                "interactionTiming",
                "meanBrightness",
                default=0.55,
            ),
            default=0.55,
        )
        audio_quality_score = self._safe_float(
            self._read_path(
                session_record,
                "qualityControl",
                "audioQualityScore",
                default=0.74 if assessment.session_usability == "usable" else 0.58,
            ),
            default=0.74 if assessment.session_usability == "usable" else 0.58,
        )
        video_quality_score = self._safe_float(
            self._read_path(
                session_record,
                "qualityControl",
                "videoQualityScore",
                default=0.72 if assessment.session_usability == "usable" else 0.56,
            ),
            default=0.72 if assessment.session_usability == "usable" else 0.56,
        )
        session_quality_flags = self._read_path(
            session_record,
            "qualityControl",
            "flags",
            default=[],
        )
        source_text = self._build_source_text(assessment, transcript_text)
        risk_score = float(assessment.risk_score or 0.0)
        keyword_text = source_text.lower()
        stage_completion = self._clamp(len(completed_stages) / 4 if completed_stages else 0.0)
        pause_signal = self._keyword_signal(keyword_text, ("pause", "hesitat", "halting"))
        fluency_signal = self._keyword_signal(keyword_text, ("fluency", "word-finding", "word finding", "language"))
        coherence_signal = self._keyword_signal(keyword_text, ("coherent", "narrative", "organized", "clear"))
        recall_signal = self._keyword_signal(keyword_text, ("memory", "recall", "forget"))

        high_risk = assessment.screening_classification == "dementia" or risk_score >= 0.7
        low_risk = assessment.screening_classification == "healthy" or risk_score <= 0.35

        pause_burden = self._clamp(
            0.14
            + (risk_score * 0.40)
            + (pause_signal * 0.16)
            + (max(0.0, average_turn_seconds - 4.0) * 0.05)
            - (stage_completion * 0.04)
            + (0.10 if high_risk else -0.08 if low_risk else 0.0)
        )
        speech_fluency = self._clamp(
            0.86
            - (pause_burden * 0.45)
            - (fluency_signal * 0.14)
            + min(speech_seconds, 60.0) / 750.0
            + (0.08 if low_risk else -0.10 if high_risk else 0.0)
        )
        narrative_coherence = self._clamp(
            0.74
            - (risk_score * 0.28)
            + (coherence_signal * 0.14)
            + (stage_completion * 0.08)
            - (fluency_signal * 0.08)
            + (0.08 if low_risk else -0.12 if high_risk else 0.0)
        )
        recall_consistency = self._clamp(
            0.72
            - (risk_score * 0.24)
            - (recall_signal * 0.14)
            + (stage_completion * 0.06)
            + (0.06 if low_risk else -0.08 if high_risk else 0.0)
        )
        support_dependency = self._clamp(
            0.12
            + (0.20 if assessment.speaker_structure == "multi_speaker" else 0.04)
            + (0.16 if assessment.target_patient_presence in {"uncertain", "absent"} else 0.0)
            + (max(0.0, 1.0 - (patient_turns / 4.0)) * 0.18)
            + (0.08 if high_risk else -0.06 if low_risk else 0.0)
        )

        presence_score = self._presence_score(assessment.target_patient_presence)
        face_visibility = self._clamp(face_detection_rate if face_detection_rate is not None else presence_score)
        face_stability = self._clamp(
            0.70
            + (min(average_face_area or 0.14, 0.30) * 0.60)
            - (motion_intensity * 0.35)
            + (0.08 if assessment.target_patient_presence == "clear" else 0.0)
        )
        motion_regularity = self._clamp(0.80 - (abs(motion_intensity - 0.32) * 0.90))
        identity_confidence = self._clamp(
            (face_visibility * 0.42)
            + (face_stability * 0.25)
            + (motion_regularity * 0.10)
            + (presence_score * 0.18)
            + (0.12 if assessment.speaker_structure != "multi_speaker" else -0.05)
        )

        qc_flags = normalize_quality_flags(
            [
                *assessment.quality_flags,
                *[self._string_value(flag) for flag in session_quality_flags if self._string_value(flag)],
            ]
        )
        qc_score = self._clamp(
            (
                audio_quality_score
                + video_quality_score
                + self._safe_float(assessment.reviewer_confidence, default=0.68)
            )
            / 3
        )
        session_usability = self._normalize_session_usability(
            self._string_value(
                self._read_path(
                    session_record,
                    "qualityControl",
                    "usability",
                    default=assessment.session_usability,
                )
            ),
            fallback=assessment.session_usability,
        )

        speech_seed = "|".join(
            [
                assessment.patient_id,
                assessment.assessment_id,
                "speech",
                transcript_text,
                str(round(speech_seconds, 2)),
                str(round(pause_burden, 3)),
            ]
        )
        conversation_seed = "|".join(
            [
                assessment.patient_id,
                assessment.assessment_id,
                "conversation",
                source_text,
                ",".join(str(stage) for stage in completed_stages),
                str(patient_turns),
                str(round(risk_score, 3)),
            ]
        )
        return LongitudinalSnapshot(
            snapshot_id=f"{assessment.assessment_id}-snapshot",
            patient_id=assessment.patient_id,
            assessment_id=assessment.assessment_id,
            session_id=self._string_value(
                self._read_path(session_record, "sessionId", default=identity_link.session_id or assessment.assessment_id)
            ),
            captured_at=assessment.created_at,
            risk_score=risk_score,
            risk_tier=assessment.risk_tier,
            screening_classification=assessment.screening_classification,
            identity_link_id=identity_link.link_id,
            timeline_inclusion=identity_link.timeline_inclusion,
            identity_confidence=identity_link.final_linkage_confidence,
            identity_reasons=list(identity_link.reasons),
            speech_embedding=self._embedding_from_seed(speech_seed),
            conversation_embedding=self._embedding_from_seed(conversation_seed),
            visual_face_embedding=list(identity_link.face_embedding),
            face_recognition_method=identity_link.face_recognition_method,
            task_behavior_features=TaskBehaviorFeatures(
                pause_burden=pause_burden,
                speech_fluency=speech_fluency,
                narrative_coherence=narrative_coherence,
                recall_consistency=recall_consistency,
                support_dependency=support_dependency,
            ),
            visual_face_features=VisualFaceFeatures(
                face_visibility=face_visibility,
                face_stability=face_stability,
                motion_regularity=motion_regularity,
                identity_confidence=identity_confidence,
            ),
            qc=LongitudinalQualityControl(
                qc_score=qc_score,
                reviewer_confidence=assessment.reviewer_confidence,
                session_usability=session_usability,
                quality_flags=qc_flags,
            ),
            source_summary=(transcript_text[:240] if transcript_text else assessment.screening_summary),
            session_record_available=bool(session_record),
        )

    def _build_trend_points(self, snapshots: list[LongitudinalSnapshot]) -> list[LongitudinalTrendPoint]:
        if not snapshots:
            return []

        selected_actual = snapshots[-self.MAX_TREND_POINTS :]
        latest_snapshot = selected_actual[-1]
        baseline_anchor_at = latest_snapshot.captured_at - timedelta(days=self.BASELINE_WINDOW_DAYS)
        gap_days = max(0, (selected_actual[0].captured_at - baseline_anchor_at).days)
        prefix_needed = 0
        if gap_days > 15:
            prefix_needed = math.ceil(gap_days / 30)
        prefix_needed = max(prefix_needed, self.MIN_TREND_POINTS - len(selected_actual))
        prefix_needed = min(prefix_needed, self.MAX_TREND_POINTS - len(selected_actual))

        points: list[LongitudinalTrendPoint] = []
        if prefix_needed > 0:
            start_score = self._baseline_seed_score(selected_actual)
            end_score = selected_actual[0].risk_score
            prefix_dates = [
                selected_actual[0].captured_at - timedelta(days=30 * step)
                for step in range(prefix_needed, 0, -1)
            ]
            for index, recorded_at in enumerate(prefix_dates):
                interpolation = index / max(1, prefix_needed)
                score = start_score + ((end_score - start_score) * interpolation)
                points.append(
                    LongitudinalTrendPoint(
                        label=self._format_date_label(recorded_at),
                        recorded_at=recorded_at,
                        risk_score=round(self._clamp(score), 2),
                        synthetic=True,
                    )
                )

        points.extend(
            LongitudinalTrendPoint(
                label=self._format_date_label(snapshot.captured_at),
                recorded_at=snapshot.captured_at,
                risk_score=round(snapshot.risk_score, 2),
                synthetic=False,
            )
            for snapshot in selected_actual
        )
        return points[-self.MAX_TREND_POINTS :]

    def _feature_snapshot_payload(
        self,
        snapshot: LongitudinalSnapshot,
        assessment: ClinicAssessment,
    ) -> dict[str, Any]:
        included_count = 1 if snapshot.timeline_inclusion == "include" else 0
        baseline_state = "complete" if included_count >= self.MIN_TREND_POINTS else "building"
        if snapshot.timeline_inclusion != "include":
            baseline_state = "not-started"
        return {
            "snapshotId": snapshot.snapshot_id,
            "patientId": snapshot.patient_id,
            "sessionId": snapshot.session_id,
            "productMode": "clinic",
            "capturedAt": snapshot.captured_at.isoformat(),
            "language": assessment.language,
            "featurePipelineVersion": "longitudinal-engine-v1",
            "quality": {
                "usability": snapshot.qc.session_usability.replace("_", "-"),
                "coverageScore": snapshot.qc.qc_score,
                "flags": snapshot.qc.quality_flags,
            },
            "embeddings": {
                "speech": {
                    "vectorUri": f"inline://feature_snapshots/{assessment.assessment_id}#speech",
                    "dimensions": len(snapshot.speech_embedding),
                    "modelVersion": "mock-speech-v1",
                },
                "conversation": {
                    "vectorUri": f"inline://feature_snapshots/{assessment.assessment_id}#conversation",
                    "dimensions": len(snapshot.conversation_embedding),
                    "modelVersion": "mock-conversation-v1",
                },
                "face": {
                    "vectorUri": f"inline://feature_snapshots/{assessment.assessment_id}#face",
                    "dimensions": len(snapshot.visual_face_embedding),
                    "modelVersion": snapshot.face_recognition_method or "face-unavailable",
                },
            },
            "features": {
                "speech": {
                    "embeddingReady": bool(snapshot.speech_embedding),
                    "sessionRecordAvailable": snapshot.session_record_available,
                },
                "conversation": {
                    "riskScore": snapshot.risk_score,
                    "screeningClassification": snapshot.screening_classification,
                },
                "taskBehavior": snapshot.task_behavior_features.model_dump(mode="json"),
                "visualFace": {
                    **snapshot.visual_face_features.model_dump(mode="json"),
                    "identityLinkId": snapshot.identity_link_id,
                    "timelineInclusion": snapshot.timeline_inclusion,
                    "identityConfidence": snapshot.identity_confidence,
                },
            },
            "trendContext": {
                "baselineState": baseline_state,
                "windowLabel": "3 months",
                "daysSinceEnrollment": 0,
            },
        }

    def _baseline_seed_score(self, snapshots: list[LongitudinalSnapshot]) -> float:
        latest = snapshots[-1]
        if len(snapshots) == 1:
            return self._default_baseline_score(latest.risk_score, latest.screening_classification)

        first_actual = snapshots[0].risk_score
        latest_delta = latest.risk_score - first_actual
        if latest_delta >= self.DECLINING_DELTA / 2:
            return self._clamp(first_actual - max(0.04, latest_delta * 0.60))
        if latest_delta <= self.IMPROVING_DELTA / 2:
            return self._clamp(first_actual + max(0.04, abs(latest_delta) * 0.60))
        return self._clamp(first_actual + (0.02 if latest.risk_score < 0.35 else 0.0))

    def _default_baseline_score(
        self,
        risk_score: float,
        screening_classification: str | None,
    ) -> float:
        if screening_classification == "dementia" or risk_score >= 0.7:
            return self._clamp(risk_score - 0.22)
        if screening_classification == "healthy" or risk_score <= 0.35:
            return self._clamp(risk_score + 0.02)
        if risk_score >= 0.55:
            return self._clamp(risk_score - 0.10)
        return self._clamp(risk_score + 0.03)

    def _direction_for_delta(self, delta: float) -> LongitudinalDirection:
        if delta >= self.DECLINING_DELTA:
            return "declining"
        if delta <= self.IMPROVING_DELTA:
            return "improving"
        return "stable"

    def _baseline_comparison_text(
        self,
        *,
        baseline_score: float,
        latest_score: float,
        direction: LongitudinalDirection,
    ) -> str:
        delta = latest_score - baseline_score
        absolute_delta = abs(delta)
        if direction == "declining":
            return (
                "Compared to your baseline (3 months ago), your risk score is "
                f"{absolute_delta:.2f} higher and the trend is Declining."
            )
        if direction == "improving":
            return (
                "Compared to your baseline (3 months ago), your risk score is "
                f"{absolute_delta:.2f} lower and the trend is Improving."
            )
        return (
            "Compared to your baseline (3 months ago), your risk score remains within "
            f"{absolute_delta:.2f} of baseline and the trend is Stable."
        )

    def _anomaly_summary(
        self,
        direction: LongitudinalDirection,
        latest_snapshot: LongitudinalSnapshot,
    ) -> str:
        features = latest_snapshot.task_behavior_features
        cues: list[str] = []
        if features.pause_burden >= 0.55:
            cues.append("higher pause burden")
        if features.speech_fluency <= 0.45:
            cues.append("reduced fluency")
        if features.recall_consistency <= 0.48:
            cues.append("less consistent recall")
        if features.narrative_coherence <= 0.48:
            cues.append("weaker narrative coherence")
        if not cues:
            cues = ["speech and task behavior"]

        joined_cues = " and ".join(cues[:2])
        if direction == "declining":
            return f"Longitudinal tracking flagged a declining pattern driven by {joined_cues}."
        if direction == "improving":
            return "Longitudinal tracking shows lower risk than baseline with more stable speech and task behavior."
        return "Longitudinal tracking remains within the expected range of the patient's baseline."

    def _gating_summary(self, counts: dict[str, int]) -> str:
        if counts["manual-review"] == 0 and counts["exclude"] == 0:
            return "No session has been held out from longitudinal tracking."
        fragments: list[str] = []
        if counts["manual-review"] > 0:
            fragments.append(f"{counts['manual-review']} session(s) pending identity review")
        if counts["exclude"] > 0:
            fragments.append(f"{counts['exclude']} session(s) excluded from the patient timeline")
        return "Longitudinal gating: " + "; ".join(fragments) + "."

    def _snapshot_counts(self, snapshots: list[LongitudinalSnapshot]) -> dict[str, int]:
        counts = {
            "total": len(snapshots),
            "include": 0,
            "manual-review": 0,
            "exclude": 0,
        }
        for snapshot in snapshots:
            counts[snapshot.timeline_inclusion] += 1
        return counts

    def _build_source_text(
        self,
        assessment: ClinicAssessment,
        transcript_text: str,
    ) -> str:
        items = [
            assessment.screening_summary or "",
            *assessment.evidence_for_risk,
            *assessment.evidence_against_risk,
            *assessment.risk_factor_findings,
            *assessment.subjective_assessment_findings,
            *assessment.emotional_assessment_findings,
            *(finding.label for finding in assessment.voice_findings),
            *(finding.summary for finding in assessment.voice_findings),
            *(finding.label for finding in assessment.content_findings),
            *(finding.summary for finding in assessment.content_findings),
            transcript_text,
        ]
        return " ".join(item.strip() for item in items if item and item.strip())

    def _embedding_from_seed(self, seed: str) -> list[float]:
        values: list[float] = []
        for index in range(self.EMBEDDING_DIM):
            digest = hashlib.sha256(f"{seed}|{index}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:4], "big") / 4294967295
            values.append(round(value, 4))
        return values

    def _keyword_signal(self, text: str, keywords: tuple[str, ...]) -> float:
        return 1.0 if any(keyword in text for keyword in keywords) else 0.0

    def _presence_score(self, value: str | None) -> float:
        mapping = {
            "clear": 0.88,
            "probable": 0.72,
            "uncertain": 0.42,
            "absent": 0.16,
        }
        return mapping.get(value or "", 0.68)

    def _normalize_session_usability(self, value: str, *, fallback: str) -> str:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"usable", "usable_with_caveats", "unusable"}:
            return normalized
        return fallback

    def _format_date_label(self, value: datetime) -> str:
        return value.strftime("%b %d")

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
