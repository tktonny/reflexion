"""Surface-oriented aggregation for doctor and caregiver product views."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta

from backend.src.app.models import (
    CareScoreMetric,
    CaregiverDashboardResponse,
    CaregiverTrendPoint,
    CaregiverTimelinePoint,
    ClinicAssessment,
    DoctorDashboardResponse,
    DoctorPatientSummary,
    ProviderDistributionItem,
    SurfaceAssessmentSummary,
    utc_now,
)
from backend.src.app.services.longitudinal_service import LongitudinalTrackingService
from clinic.database.storage import LocalStorage


class SurfaceService:
    """Build lightweight doctor and caregiver dashboards from stored assessments."""

    DEMO_PATIENTS: tuple[str, str] = ("demo-dementia-case", "demo-healthy-case")

    def __init__(self, storage: LocalStorage) -> None:
        self.storage = storage
        self.longitudinal = LongitudinalTrackingService(storage)

    def list_known_patients(self) -> list[str]:
        actual_patients = sorted({assessment.patient_id for assessment in self._load_assessments()})
        longitudinal_patients = sorted(
            {
                str(payload.get("patient_id", "")).strip()
                for payload in self.storage.list_longitudinal_profiles()
                if str(payload.get("patient_id", "")).strip()
            }
        )
        known = list(self.DEMO_PATIENTS)
        for patient_id in actual_patients:
            if patient_id not in known:
                known.append(patient_id)
        for patient_id in longitudinal_patients:
            if patient_id not in known:
                known.append(patient_id)
        return known

    def build_doctor_dashboard(
        self,
        *,
        recent_limit: int = 10,
        patient_limit: int = 12,
    ) -> DoctorDashboardResponse:
        assessments = self._load_assessments()
        grouped = self._group_by_patient(assessments)
        risk_scores = [item.risk_score for item in assessments if item.risk_score is not None]
        provider_counts = Counter(item.provider_meta.final_provider for item in assessments)

        patient_summaries = []
        for patient_id, items in grouped.items():
            latest = items[0]
            patient_risk_scores = [item.risk_score for item in items if item.risk_score is not None]
            patient_summaries.append(
                DoctorPatientSummary(
                    patient_id=patient_id,
                    session_count=len(items),
                    latest_assessment_id=latest.assessment_id,
                    latest_assessment_at=latest.created_at,
                    latest_risk_tier=latest.risk_tier,
                    latest_screening_classification=latest.screening_classification,
                    latest_summary=latest.screening_summary,
                    average_risk_score=self._average(patient_risk_scores),
                )
            )

        patient_summaries.sort(
            key=lambda item: (
                item.latest_assessment_at is not None,
                item.latest_assessment_at,
                item.average_risk_score or 0.0,
            ),
            reverse=True,
        )

        return DoctorDashboardResponse(
            generated_at=utc_now(),
            total_assessments=len(assessments),
            total_patients=len(grouped),
            high_risk_assessments=sum(1 for item in assessments if item.risk_tier == "high"),
            watchlist_assessments=sum(
                1
                for item in assessments
                if item.risk_tier == "medium" or item.screening_classification == "needs_observation"
            ),
            usable_sessions=sum(1 for item in assessments if item.session_usability == "usable"),
            average_risk_score=self._average(risk_scores),
            provider_distribution=[
                ProviderDistributionItem(provider=provider, assessments=count)
                for provider, count in sorted(provider_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            recent_assessments=[self._to_summary(item) for item in assessments[:recent_limit]],
            patient_summaries=patient_summaries[:patient_limit],
        )

    def build_caregiver_dashboard(self, patient_id: str, *, history_limit: int = 6) -> CaregiverDashboardResponse:
        assessments = self._load_assessments()
        known_patients = self.list_known_patients()

        if patient_id == "demo-dementia-case":
            return self._build_demo_dashboard(patient_id, known_patients, scenario="dementia")
        if patient_id == "demo-healthy-case":
            return self._build_demo_dashboard(patient_id, known_patients, scenario="healthy")

        patient_history = [item for item in assessments if item.patient_id == patient_id]

        if not patient_history:
            return CaregiverDashboardResponse(
                generated_at=utc_now(),
                patient_id=patient_id,
                known_patients=known_patients,
                sessions_completed=0,
                last_updated_at=None,
                risk_label="LOW RISK",
                risk_score=0.0,
                score_breakdown=self._default_score_breakdown(),
                top_reasons=["No completed assessment is available yet."],
                recommendation="Complete a mirror or clinic session to generate the first formal assessment.",
                baseline_comparison=(
                    "Compared to your baseline (3 months ago), no longitudinal trend is available yet."
                ),
                longitudinal_direction="stable",
                longitudinal_direction_label="Stable",
                anomaly_summary="No abnormal deviation from baseline has been flagged yet.",
                identity_gating_summary="No session has been held out from longitudinal tracking.",
                chart_x_axis_label="Date",
                chart_y_axis_label="Risk Score",
                longitudinal_points=[
                    CaregiverTrendPoint(label="Jan 01", risk_score=0.0, is_baseline=True, synthetic=True),
                    CaregiverTrendPoint(label="Feb 01", risk_score=0.0, synthetic=True),
                    CaregiverTrendPoint(label="Mar 01", risk_score=0.0, synthetic=True),
                    CaregiverTrendPoint(label="Today", risk_score=0.0, synthetic=True),
                ],
                signal="stable",
                status_label="No shared updates yet",
                care_summary="No completed formal assessment has been shared for this patient yet.",
                next_steps=[
                    "Complete a mirror or clinic session to generate the first formal assessment.",
                    "Ask the care team to share the latest approved summary here.",
                ],
                alerts=[],
                latest_assessment=None,
                history=[],
            )

        latest = patient_history[0]
        longitudinal = self.longitudinal.build_profile(patient_id, patient_history)
        signal = self._care_signal(latest)
        risk_label = self._care_risk_label(latest)
        risk_score = round(latest.risk_score if latest.risk_score is not None else 0.0, 2)
        score_breakdown = self._care_score_breakdown(latest, longitudinal.snapshots)
        top_reasons = self._care_top_reasons(latest)
        recommendation = self._care_recommendation(latest, risk_label)
        status_label = self._care_status_label(signal)
        next_steps = self._dedupe(
            [recommendation]
            + list(latest.risk_control_suggestions[:2])
        )
        alerts = self._care_alerts(latest)
        if longitudinal.direction != "stable":
            alerts = self._dedupe([longitudinal.anomaly_summary] + alerts)
        if longitudinal.gating_summary != "No session has been held out from longitudinal tracking.":
            alerts = self._dedupe([longitudinal.gating_summary] + alerts)

        return CaregiverDashboardResponse(
            generated_at=utc_now(),
            patient_id=patient_id,
            known_patients=known_patients,
            sessions_completed=max(len(patient_history), longitudinal.total_sessions_seen),
            last_updated_at=latest.created_at,
            risk_label=risk_label,
            risk_score=risk_score,
            score_breakdown=score_breakdown,
            top_reasons=top_reasons,
            recommendation=recommendation,
            baseline_comparison=longitudinal.baseline_comparison,
            longitudinal_direction=longitudinal.direction,
            longitudinal_direction_label=longitudinal.direction_label,
            anomaly_summary=longitudinal.anomaly_summary,
            identity_gating_summary=longitudinal.gating_summary,
            chart_x_axis_label=longitudinal.x_axis_label,
            chart_y_axis_label=longitudinal.y_axis_label,
            longitudinal_points=self._to_caregiver_trend_points(longitudinal.trend_points),
            signal=signal,
            status_label=status_label,
            care_summary=self._caregiver_summary(latest, risk_label, top_reasons),
            next_steps=next_steps,
            alerts=alerts,
            latest_assessment=self._to_summary(latest),
            history=[
                CaregiverTimelinePoint(
                    assessment_id=item.assessment_id,
                    created_at=item.created_at,
                    signal=self._care_signal(item),
                    status_label=self._care_status_label(self._care_signal(item)),
                    shareable_summary=self._caregiver_summary(
                        item,
                        self._care_risk_label(item),
                        self._care_top_reasons(item),
                    ),
                )
                for item in patient_history[:history_limit]
            ],
        )

    def _build_demo_dashboard(
        self,
        patient_id: str,
        known_patients: list[str],
        *,
        scenario: str,
    ) -> CaregiverDashboardResponse:
        generated_at = utc_now()
        if scenario == "dementia":
            return CaregiverDashboardResponse(
                generated_at=generated_at,
                patient_id=patient_id,
                known_patients=known_patients,
                sessions_completed=4,
                last_updated_at=generated_at,
                risk_label="HIGH RISK",
                risk_score=0.78,
                score_breakdown=[
                    CareScoreMetric(key="speak", label="Speak", value=0.82),
                    CareScoreMetric(key="content", label="Content", value=0.75),
                    CareScoreMetric(key="pose", label="Pose", value=0.44),
                    CareScoreMetric(key="face", label="Face", value=0.28),
                ],
                top_reasons=[
                    "Frequent pauses",
                    "Reduced fluency",
                    "Word-finding difficulty",
                ],
                recommendation="Recommend further cognitive assessment.",
                baseline_comparison=(
                    "Compared to your baseline (3 months ago), your risk score is 0.50 higher and the trend is Declining."
                ),
                longitudinal_direction="declining",
                longitudinal_direction_label="Declining",
                anomaly_summary=(
                    "Longitudinal tracking flagged a declining pattern driven by higher pause burden and reduced fluency."
                ),
                identity_gating_summary="No session has been held out from longitudinal tracking.",
                chart_x_axis_label="Date",
                chart_y_axis_label="Risk Score",
                longitudinal_points=self._demo_trend_points(
                    generated_at,
                    [0.28, 0.46, 0.63, 0.78],
                ),
                signal="urgent",
                status_label="Higher concern today",
                care_summary=(
                    "This demo case shows a higher-risk pattern than the patient's baseline, "
                    "with progressive language difficulty across recent sessions."
                ),
                next_steps=[
                    "Recommend further cognitive assessment.",
                    "Share the result with the clinician or care team today.",
                ],
                alerts=[
                    "This is the hard-coded high-risk demo case for reliable live presentations.",
                ],
                latest_assessment=None,
                history=[
                    CaregiverTimelinePoint(
                        assessment_id="demo-dementia-1",
                        created_at=generated_at - timedelta(days=42),
                        signal="watch",
                        status_label="Early change",
                        shareable_summary="Language pauses were longer than baseline during the first check-in.",
                    ),
                    CaregiverTimelinePoint(
                        assessment_id="demo-dementia-2",
                        created_at=generated_at - timedelta(days=14),
                        signal="urgent",
                        status_label="Rising concern",
                        shareable_summary="Reduced fluency and word-finding difficulty became more consistent.",
                    ),
                ],
            )

        return CaregiverDashboardResponse(
            generated_at=generated_at,
            patient_id=patient_id,
            known_patients=known_patients,
            sessions_completed=4,
            last_updated_at=generated_at,
            risk_label="LOW RISK",
            risk_score=0.24,
            score_breakdown=[
                CareScoreMetric(key="speak", label="Speak", value=0.18),
                CareScoreMetric(key="content", label="Content", value=0.16),
                CareScoreMetric(key="pose", label="Pose", value=0.11),
                CareScoreMetric(key="face", label="Face", value=0.09),
            ],
            top_reasons=[
                "Normal pause duration",
                "Coherent narrative",
                "Typical vocabulary",
            ],
            recommendation="Continue routine monitoring.",
            baseline_comparison=(
                "Compared to your baseline (3 months ago), your risk score remains within 0.02 of baseline and the trend is Stable."
            ),
            longitudinal_direction="stable",
            longitudinal_direction_label="Stable",
            anomaly_summary="Longitudinal tracking remains within the expected range of the patient's baseline.",
            identity_gating_summary="No session has been held out from longitudinal tracking.",
            chart_x_axis_label="Date",
            chart_y_axis_label="Risk Score",
            longitudinal_points=self._demo_trend_points(
                generated_at,
                [0.26, 0.24, 0.25, 0.24],
            ),
            signal="stable",
            status_label="Stable today",
            care_summary=(
                "This demo case stays close to baseline and shows the lower-risk pattern expected "
                "for a healthy control example."
            ),
            next_steps=[
                "Continue routine monitoring.",
                "Repeat the check only if new concerns appear.",
            ],
            alerts=[
                "This is the hard-coded low-risk demo case for reliable live presentations.",
            ],
            latest_assessment=None,
            history=[
                CaregiverTimelinePoint(
                    assessment_id="demo-healthy-1",
                    created_at=generated_at - timedelta(days=42),
                    signal="stable",
                    status_label="Stable update",
                    shareable_summary="Narrative quality and pause timing stayed close to baseline.",
                ),
                CaregiverTimelinePoint(
                    assessment_id="demo-healthy-2",
                    created_at=generated_at - timedelta(days=14),
                    signal="stable",
                    status_label="Stable update",
                    shareable_summary="Typical vocabulary and coherent storytelling remained intact.",
                ),
            ],
        )

    def _load_assessments(self) -> list[ClinicAssessment]:
        assessments = [
            ClinicAssessment.model_validate(payload)
            for payload in self.storage.list_assessments()
        ]
        assessments.sort(key=lambda item: item.created_at, reverse=True)
        return assessments

    def _group_by_patient(self, assessments: list[ClinicAssessment]) -> dict[str, list[ClinicAssessment]]:
        grouped: dict[str, list[ClinicAssessment]] = defaultdict(list)
        for assessment in assessments:
            grouped[assessment.patient_id].append(assessment)
        return grouped

    def _to_summary(self, assessment: ClinicAssessment) -> SurfaceAssessmentSummary:
        return SurfaceAssessmentSummary(
            assessment_id=assessment.assessment_id,
            patient_id=assessment.patient_id,
            created_at=assessment.created_at,
            risk_score=assessment.risk_score,
            risk_tier=assessment.risk_tier,
            screening_classification=assessment.screening_classification,
            screening_summary=assessment.screening_summary,
            session_usability=assessment.session_usability,
            reviewer_confidence=assessment.reviewer_confidence,
            final_provider=assessment.provider_meta.final_provider,
            visit_recommendation=assessment.visit_recommendation,
        )

    def _average(self, values: list[float | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return round(sum(filtered) / len(filtered), 2)

    def _bounded_score(self, value: float) -> float:
        return round(max(0.0, min(1.0, value)), 2)

    def _default_score_breakdown(self) -> list[CareScoreMetric]:
        return [
            CareScoreMetric(key="speak", label="Speak", value=0.0),
            CareScoreMetric(key="content", label="Content", value=0.0),
            CareScoreMetric(key="pose", label="Pose", value=0.0),
            CareScoreMetric(key="face", label="Face", value=0.0),
        ]

    def _care_score_breakdown(
        self,
        assessment: ClinicAssessment,
        snapshots,
    ) -> list[CareScoreMetric]:
        latest_snapshot = next(
            (item for item in reversed(list(snapshots)) if item.assessment_id == assessment.assessment_id),
            None,
        )
        if latest_snapshot is None:
            risk_score = float(assessment.risk_score or 0.0)
            return [
                CareScoreMetric(key="speak", label="Speak", value=self._bounded_score((risk_score * 0.82) + 0.08)),
                CareScoreMetric(key="content", label="Content", value=self._bounded_score((risk_score * 0.76) + 0.08)),
                CareScoreMetric(key="pose", label="Pose", value=self._bounded_score((risk_score * 0.48) + 0.05)),
                CareScoreMetric(key="face", label="Face", value=self._bounded_score((risk_score * 0.32) + 0.05)),
            ]

        task = latest_snapshot.task_behavior_features
        visual = latest_snapshot.visual_face_features
        speak = (task.pause_burden + (1.0 - task.speech_fluency)) / 2
        content = ((1.0 - task.narrative_coherence) + (1.0 - task.recall_consistency)) / 2
        pose = 1.0 - visual.motion_regularity
        face = ((1.0 - visual.face_visibility) + (1.0 - visual.face_stability)) / 2
        return [
            CareScoreMetric(key="speak", label="Speak", value=self._bounded_score(speak)),
            CareScoreMetric(key="content", label="Content", value=self._bounded_score(content)),
            CareScoreMetric(key="pose", label="Pose", value=self._bounded_score(pose)),
            CareScoreMetric(key="face", label="Face", value=self._bounded_score(face)),
        ]

    def _care_signal(self, assessment: ClinicAssessment) -> str:
        if assessment.screening_classification == "dementia" or assessment.risk_tier == "high":
            return "urgent"
        if assessment.screening_classification == "needs_observation" or assessment.risk_tier == "medium":
            return "watch"
        return "stable"

    def _care_risk_label(self, assessment: ClinicAssessment) -> str:
        score = assessment.risk_score if assessment.risk_score is not None else 0.0
        if assessment.screening_classification == "dementia" or score >= 0.7:
            return "HIGH RISK"
        if assessment.screening_classification == "healthy" or score <= 0.35:
            return "LOW RISK"
        if assessment.screening_classification == "needs_observation":
            return "HIGH RISK" if score >= 0.55 else "LOW RISK"
        return "HIGH RISK" if score >= 0.55 else "LOW RISK"

    def _care_status_label(self, signal: str) -> str:
        if signal == "urgent":
            return "Higher concern today"
        if signal == "watch":
            return "Watch for change"
        return "Stable today"

    def _caregiver_summary(
        self,
        assessment: ClinicAssessment,
        risk_label: str,
        top_reasons: list[str],
    ) -> str:
        reasons = ", ".join(top_reasons[:2]).lower()
        if risk_label == "LOW RISK":
            return (
                "This result looks lower risk overall and stays close to baseline, with "
                f"{reasons or 'reassuring speech and narrative signals'}."
            )
        if assessment.screening_classification == "needs_observation":
            return (
                "This result sits above baseline and should be watched closely, especially for "
                f"{reasons or 'language and recall changes'}."
            )
        return (
            "This result shows a higher-risk pattern than baseline, especially for "
            f"{reasons or 'language and fluency changes'}."
        )

    def _care_top_reasons(self, assessment: ClinicAssessment) -> list[str]:
        text = " ".join(
            item
            for item in (
                assessment.screening_summary or "",
                *assessment.evidence_for_risk,
                *assessment.evidence_against_risk,
                *assessment.risk_factor_findings,
                *(finding.label for finding in assessment.voice_findings),
                *(finding.summary for finding in assessment.voice_findings),
                *(finding.label for finding in assessment.content_findings),
                *(finding.summary for finding in assessment.content_findings),
            )
        ).lower()

        if self._care_risk_label(assessment) == "HIGH RISK":
            candidates = [
                ("pause", "Frequent pauses"),
                ("hesitat", "Frequent pauses"),
                ("fluency", "Reduced fluency"),
                ("disorgan", "Reduced fluency"),
                ("word", "Word-finding difficulty"),
                ("language", "Word-finding difficulty"),
                ("memory", "Memory retrieval difficulty"),
                ("recall", "Memory retrieval difficulty"),
            ]
            fallback = [
                "Frequent pauses",
                "Reduced fluency",
                "Word-finding difficulty",
            ]
        else:
            candidates = [
                ("pause", "Normal pause duration"),
                ("coherent", "Coherent narrative"),
                ("narrative", "Coherent narrative"),
                ("vocabulary", "Typical vocabulary"),
                ("fluent", "Typical vocabulary"),
                ("clear", "Typical vocabulary"),
            ]
            fallback = [
                "Normal pause duration",
                "Coherent narrative",
                "Typical vocabulary",
            ]

        reasons: list[str] = []
        for keyword, label in candidates:
            if keyword in text and label not in reasons:
                reasons.append(label)
        for label in fallback:
            if label not in reasons:
                reasons.append(label)
        return reasons[:3]

    def _care_recommendation(self, assessment: ClinicAssessment, risk_label: str) -> str:
        if risk_label == "HIGH RISK":
            return assessment.visit_recommendation or "Recommend further cognitive assessment."
        return assessment.visit_recommendation or "Continue routine monitoring."

    def _care_trend_points(self, patient_history: list[ClinicAssessment]) -> list[CaregiverTrendPoint]:
        scored_history = [item for item in reversed(patient_history[:5]) if item.risk_score is not None]
        if len(scored_history) >= 2:
            points: list[CaregiverTrendPoint] = []
            for index, item in enumerate(scored_history):
                label = "Baseline" if index == 0 else item.created_at.strftime("%b %d")
                points.append(
                    CaregiverTrendPoint(
                        label=label,
                        risk_score=round(item.risk_score or 0.0, 2),
                        is_baseline=index == 0,
                    )
                )
            return points

        latest = patient_history[0]
        latest_score = round(latest.risk_score or 0.0, 2)
        if self._care_risk_label(latest) == "HIGH RISK":
            series = [max(0.12, latest_score - 0.32), max(0.18, latest_score - 0.18), latest_score]
        else:
            series = [min(0.32, latest_score + 0.03), min(0.32, latest_score + 0.01), latest_score]
        labels = ["Baseline", "Recent", "Today"]
        return [
            CaregiverTrendPoint(label=label, risk_score=round(score, 2), is_baseline=index == 0)
            for index, (label, score) in enumerate(zip(labels, series, strict=False))
        ]

    def _to_caregiver_trend_points(self, points) -> list[CaregiverTrendPoint]:
        return [
            CaregiverTrendPoint(
                label=point.label,
                risk_score=round(point.risk_score, 2),
                recorded_at=point.recorded_at,
                is_baseline=point.is_baseline,
                synthetic=point.synthetic,
            )
            for point in points
        ]

    def _demo_trend_points(
        self,
        generated_at,
        scores: list[float],
    ) -> list[CaregiverTrendPoint]:
        last_index = len(scores) - 1
        return [
            CaregiverTrendPoint(
                label=(generated_at - timedelta(days=30 * (last_index - index))).strftime("%b %d"),
                recorded_at=generated_at - timedelta(days=30 * (last_index - index)),
                risk_score=round(score, 2),
                is_baseline=index == 0,
                synthetic=True,
            )
            for index, score in enumerate(scores)
        ]

    def _care_alerts(self, assessment: ClinicAssessment) -> list[str]:
        alerts: list[str] = []
        if assessment.screening_classification == "dementia" or assessment.risk_tier == "high":
            alerts.append("The latest formal review suggests prompt clinician follow-up.")
        if assessment.session_usability != "usable":
            alerts.append("This recording had quality caveats, so repeating the session may help.")
        if assessment.target_patient_presence not in {"clear", "probable", None}:
            alerts.append("The capture may not have focused clearly enough on the patient.")
        if assessment.quality_flags:
            alerts.append(
                "Quality notes: " + ", ".join(flag.replace("_", " ") for flag in assessment.quality_flags[:3]) + "."
            )
        return self._dedupe(alerts)

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in items:
            stripped = item.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                cleaned.append(stripped)
        return cleaned
