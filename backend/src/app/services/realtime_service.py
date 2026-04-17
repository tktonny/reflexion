"""Realtime conversation relay and demo risk scoring for the clinic UI."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from clinic.configs.settings import Settings
from backend.src.app.models import (
    FeatureSignal,
    RealtimeAnalysisRequest,
    RealtimeAssessment,
    RealtimeSessionStatus,
    SimilarityBreakdown,
    TrendPoint,
    utc_now,
)
from backend.src.app.services.realtime_orchestrator import (
    RealtimeConversationOrchestrator,
)
from backend.src.app.services.identity_service import IdentityLinkageService
from backend.src.app.services.patient_memory import normalize_patient_name
from clinic.database.storage import LocalStorage

try:
    import websockets
except ImportError:  # pragma: no cover - covered by runtime fallback mode
    websockets = None


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class RealtimeVoiceProfile:
    language_key: str
    language_label: str
    voice: str
    source: str


@dataclass(frozen=True)
class RealtimeLanguageSignal:
    language_key: str
    confidence: float
    source: str


class RealtimeConversationService:
    """Power the live Qwen relay and the post-session demo scoring flow."""

    HESITATION_MARKERS: tuple[str, ...] = (
        " um ",
        " uh ",
        " er ",
        " ah ",
        " hmm ",
        " mm ",
        " 呃 ",
        " 额 ",
        " 嗯 ",
    )
    MEMORY_MARKERS: tuple[str, ...] = (
        "don't remember",
        "do not remember",
        "can't remember",
        "cannot remember",
        "forgot",
        "forgotten",
        "not sure",
        "hard to remember",
        "记不清",
        "不记得",
        "想不起来",
        "忘了",
    )
    DETAIL_MARKERS: tuple[str, ...] = (
        "first",
        "then",
        "after",
        "before",
        "later",
        "because",
        "today",
        "this morning",
        "after that",
        "earlier",
        "先",
        "然后",
        "之后",
        "因为",
        "早上",
        "今天",
        "后来",
    )
    ORIENTATION_MARKERS: tuple[str, ...] = (
        "home",
        "clinic",
        "hospital",
        "april",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "在家",
        "诊所",
        "医院",
        "今天",
        "四月",
    )
    SUPPORT_MARKERS: tuple[str, ...] = (
        "my daughter helps",
        "my son helps",
        "my wife helps",
        "my husband helps",
        "caregiver",
        "someone helps",
        "need help",
        "helps me with medication",
        "helps me with appointments",
        "女儿帮我",
        "儿子帮我",
        "需要帮助",
        "提醒我吃药",
        "帮我预约",
    )
    LOW_SIGNAL_TOKENS: frozenset[str] = frozenset(
        {
            "about",
            "after",
            "again",
            "also",
            "and",
            "before",
            "earlier",
            "from",
            "have",
            "just",
            "later",
            "mentioned",
            "morning",
            "really",
            "said",
            "that",
            "then",
            "there",
            "they",
            "thing",
            "this",
            "today",
            "what",
            "with",
            "would",
            "your",
        }
    )
    LANGUAGE_HINT_ALIASES: dict[str, tuple[str, ...]] = {
        "english": ("en", "en-us", "en-gb", "english"),
        "mandarin": (
            "zh",
            "zh-cn",
            "zh-hans",
            "cmn",
            "chinese",
            "mandarin",
            "mandarin chinese",
            "chinese mandarin",
            "putonghua",
            "普通话",
            "国语",
            "中文",
            "汉语",
            "漢語",
        ),
        "minnan": (
            "nan",
            "minnan",
            "hokkien",
            "taiwanese",
            "taiyu",
            "min nan",
            "minnan chinese",
            "闽南",
            "闽南话",
            "闽南语",
            "閩南",
            "閩南話",
            "閩南語",
            "台语",
            "台語",
            "臺語",
        ),
        "cantonese": (
            "yue",
            "yue-cn",
            "cantonese",
            "cantonese chinese",
            "guangdonghua",
            "广东话",
            "廣東話",
            "粤语",
            "粵語",
            "白话",
            "白話",
        ),
        "malay": (
            "ms",
            "ms-my",
            "malay",
            "bahasa",
            "bahasa melayu",
            "melayu",
        ),
        "tamil": (
            "ta",
            "ta-in",
            "tamil",
            "தமிழ்",
        ),
    }
    MINNAN_MARKERS: tuple[str, ...] = (
        "按怎",
        "啥物",
        "歹势",
        "歹勢",
        "有影",
        "毋知",
        "欲",
        "今仔日",
        "昨昏",
        "恁",
        "阮",
        "咱",
        "這馬",
        "这马",
        "家己",
        "逐家",
        "伊",
        "彼个",
        "啥人",
        "無啥",
        "无啥",
        "有夠",
        "真濟",
        "遐",
        "食饱未",
        "食飽未",
    )
    CANTONESE_MARKERS: tuple[str, ...] = (
        "佢",
        "冇",
        "唔",
        "喺",
        "而家",
        "依家",
        "有冇",
        "咩",
        "乜",
        "嘅",
        "咗",
        "嚟",
        "係",
        "呢",
        "啦",
        "喇",
        "喎",
        "啱啱",
        "頭先",
        "返屋企",
        "佢哋",
        "唔係",
        "咁樣",
    )
    ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset(
        {
            "a",
            "am",
            "and",
            "are",
            "at",
            "did",
            "for",
            "had",
            "have",
            "hello",
            "home",
            "i",
            "i'm",
            "im",
            "in",
            "is",
            "it",
            "me",
            "my",
            "name",
            "now",
            "right",
            "the",
            "this",
            "today",
            "was",
            "went",
            "where",
            "you",
        }
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = LocalStorage(settings)
        self.identity = IdentityLinkageService(self.storage)
        self.orchestrator = RealtimeConversationOrchestrator(settings)

    def build_session_status(
        self,
        *,
        force_guided_demo: bool = False,
        preferred_language: str | None = None,
        patient_id: str | None = None,
    ) -> RealtimeSessionStatus:
        known_name, known_memory = self._known_patient_context(patient_id)
        live_relay_available = bool(not force_guided_demo and self._live_qwen_ready())
        if live_relay_available:
            voice_profile = self._voice_profile_for_session(language_hint=preferred_language)
            return RealtimeSessionStatus(
                session_mode="live_qwen",
                conversation_provider="qwen_omni_realtime",
                model_name=self.settings.qwen_omni_realtime_model,
                live_relay_available=True,
                selected_voice=voice_profile.voice,
                selected_language=voice_profile.language_label,
                voice_selection_source=voice_profile.source,
                max_session_seconds=self.settings.realtime_max_session_seconds,
                max_reply_seconds=self.settings.realtime_max_assistant_response_seconds,
                max_reply_chars=self.settings.realtime_max_assistant_response_chars,
                flow_id=self.orchestrator.flow_id,
                flow_title=self.orchestrator.flow_title_for_session(
                    patient_name=known_name,
                    memory=known_memory,
                ),
                conversation_goal=self.orchestrator.conversation_goal_for_session(
                    patient_name=known_name,
                    memory=known_memory,
                ),
                completion_rule=self.orchestrator.completion_rule_for_session(
                    patient_name=known_name,
                    memory=known_memory,
                ),
                greeting=self.orchestrator.opening_message_for_language(
                    preferred_language,
                    patient_name=known_name,
                    returning_patient=self.orchestrator._is_returning_patient(
                        patient_name=known_name,
                        memory=known_memory,
                    ),
                ),
                prompt_steps=self.orchestrator.prompt_steps_for_session(
                    language=preferred_language,
                    patient_name=known_name,
                    memory=known_memory,
                ),
                processing_steps=self.orchestrator.processing_steps,
                fallback_note=None,
            )
        return RealtimeSessionStatus(
            session_mode="guided_demo",
            conversation_provider="guided_demo",
            model_name="scripted-fallback",
            live_relay_available=False,
            selected_voice=None,
            selected_language=None,
            voice_selection_source=None,
            max_session_seconds=self.settings.realtime_max_session_seconds,
            max_reply_seconds=self.settings.realtime_max_assistant_response_seconds,
            max_reply_chars=self.settings.realtime_max_assistant_response_chars,
            flow_id=self.orchestrator.flow_id,
            flow_title=self.orchestrator.flow_title_for_session(
                patient_name=known_name,
                memory=known_memory,
            ),
            conversation_goal=self.orchestrator.conversation_goal_for_session(
                patient_name=known_name,
                memory=known_memory,
            ),
            completion_rule=self.orchestrator.completion_rule_for_session(
                patient_name=known_name,
                memory=known_memory,
            ),
            greeting=self.orchestrator.opening_message_for_language(
                preferred_language,
                patient_name=known_name,
                returning_patient=self.orchestrator._is_returning_patient(
                    patient_name=known_name,
                    memory=known_memory,
                ),
            ),
            prompt_steps=self.orchestrator.prompt_steps_for_session(
                language=preferred_language,
                patient_name=known_name,
                memory=known_memory,
            ),
            processing_steps=self.orchestrator.processing_steps,
            fallback_note=(
                "Live Qwen relay is unavailable, so the interface keeps the guided conversation and local risk narrative active."
            ),
        )

    async def run_session(
        self,
        websocket: WebSocket,
        patient_id: str,
        language: str,
    ) -> None:
        """Accept one browser session and serve either live or guided mode."""

        await websocket.accept()
        status = self.build_session_status(preferred_language=language, patient_id=patient_id)
        logger.info(
            "Realtime browser session accepted patient_id=%s language=%s mode=%s provider=%s live=%s",
            patient_id,
            language,
            status.session_mode,
            status.conversation_provider,
            status.live_relay_available,
        )
        await websocket.send_json({"type": "reflexion.session.ready", "session": status.model_dump()})

        if status.session_mode == "live_qwen":
            try:
                await self._run_live_qwen(
                    websocket,
                    patient_id=patient_id,
                    language=language,
                )
                return
            except WebSocketDisconnect:
                logger.info("Realtime browser websocket disconnected during live session")
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Realtime live relay degraded to guided demo: %s", exc)
                degraded = self.build_session_status(
                    force_guided_demo=True,
                    preferred_language=language,
                    patient_id=patient_id,
                )
                await websocket.send_json(
                    {
                        "type": "reflexion.session.degraded",
                        "reason": str(exc),
                        "session": degraded.model_dump(),
                    }
                )

        logger.warning("Realtime session running in guided demo mode")
        with suppress(WebSocketDisconnect):
            await self._run_guided_demo(
                websocket,
                language=language,
                patient_id=patient_id,
            )

    def analyze_session(self, request: RealtimeAnalysisRequest) -> RealtimeAssessment:
        """Convert one captured conversation into a deterministic demo assessment."""

        patient_turns = [turn for turn in request.transcript if turn.role == "patient" and turn.text.strip()]
        patient_text = " ".join(turn.text for turn in patient_turns).strip()
        lower_text = f" {patient_text.lower()} " if patient_text else " "
        tokens = self._tokenize(patient_text)
        word_count = len(tokens)
        patient_turn_count = len(patient_turns)
        average_turn_length = word_count / max(1, patient_turn_count)

        hesitation_rate = min(1.0, self._count_markers(lower_text, self.HESITATION_MARKERS) / max(1, patient_turn_count * 2))
        memory_difficulty = min(1.0, self._count_markers(lower_text, self.MEMORY_MARKERS) / max(1, patient_turn_count))
        support_dependency = min(1.0, self._count_markers(lower_text, self.SUPPORT_MARKERS) / max(1, patient_turn_count))
        detail_density = min(
            1.0,
            (
                self._count_markers(lower_text, self.DETAIL_MARKERS)
                + self._count_markers(lower_text, self.ORIENTATION_MARKERS)
            )
            / max(1, patient_turn_count * 3),
        )
        lexical_richness = min(1.0, len(set(tokens)) / max(1, word_count))
        turn_elaboration = min(1.0, average_turn_length / 18.0)

        recent_story_text = " ".join(
            turn.text for turn in patient_turns if turn.stage == "recent_story"
        )
        delayed_recall_text = " ".join(
            turn.text for turn in patient_turns if turn.stage == "delayed_recall"
        )
        delayed_recall_score = self._score_delayed_recall_response(
            delayed_recall_text,
            reference_text=recent_story_text,
        )

        frames_captured = request.visual_metrics.frames_captured
        face_detection_rate = request.visual_metrics.face_detection_rate
        average_face_area = request.visual_metrics.average_face_area
        motion_intensity = request.visual_metrics.motion_intensity
        mean_brightness = request.visual_metrics.mean_brightness

        speech_completeness = self._clip(((word_count / 70.0) + (patient_turn_count / 4.0)) / 2.0)
        visual_completeness = self._clip(
            min(1.0, frames_captured / 8.0) * (1.0 if face_detection_rate is not None else 0.6)
        )

        hesitation_rate = self._blend_with_neutral(hesitation_rate, speech_completeness)
        memory_difficulty = self._blend_with_neutral(memory_difficulty, speech_completeness)
        support_dependency = self._blend_with_neutral(support_dependency, speech_completeness)
        detail_density = self._blend_with_neutral(detail_density, speech_completeness)
        lexical_richness = self._blend_with_neutral(lexical_richness, speech_completeness)
        turn_elaboration = self._blend_with_neutral(turn_elaboration, speech_completeness)
        delayed_recall_score = self._blend_with_neutral(delayed_recall_score, speech_completeness)

        face_presence = self._blend_with_neutral(
            face_detection_rate if face_detection_rate is not None else 0.55,
            visual_completeness,
        )
        face_coverage = self._blend_with_neutral(
            average_face_area if average_face_area is not None else 0.24,
            visual_completeness,
        )
        movement_proxy = self._blend_with_neutral(
            motion_intensity if motion_intensity is not None else 0.42,
            visual_completeness,
        )
        brightness_balance = self._blend_with_neutral(
            1.0 - min(1.0, abs((mean_brightness if mean_brightness is not None else 0.52) - 0.52) * 2.2),
            visual_completeness,
        )

        speech_embedding = [
            memory_difficulty,
            hesitation_rate,
            1.0 - detail_density,
            1.0 - delayed_recall_score,
            support_dependency,
            1.0 - turn_elaboration,
            1.0 - lexical_richness,
        ]
        visual_embedding = [
            1.0 - face_presence,
            1.0 - movement_proxy,
            1.0 - face_coverage,
            1.0 - brightness_balance,
        ]

        speech_dementia_centroid = [0.84, 0.62, 0.58, 0.88, 0.56, 0.53, 0.46]
        speech_non_dementia_centroid = [0.18, 0.24, 0.22, 0.16, 0.18, 0.24, 0.18]
        visual_dementia_centroid = [0.42, 0.48, 0.38, 0.34]
        visual_non_dementia_centroid = [0.18, 0.22, 0.16, 0.20]

        speech_similarity_dementia = self._cosine_similarity(speech_embedding, speech_dementia_centroid)
        speech_similarity_non_dementia = self._cosine_similarity(
            speech_embedding,
            speech_non_dementia_centroid,
        )
        visual_similarity_dementia = self._cosine_similarity(visual_embedding, visual_dementia_centroid)
        visual_similarity_non_dementia = self._cosine_similarity(
            visual_embedding,
            visual_non_dementia_centroid,
        )

        speech_weight = 0.78
        visual_weight = 0.22
        dementia_similarity = self._clip(
            speech_similarity_dementia * speech_weight + visual_similarity_dementia * visual_weight
        )
        non_dementia_similarity = self._clip(
            speech_similarity_non_dementia * speech_weight
            + visual_similarity_non_dementia * visual_weight
        )
        risk_score = self._clip(0.5 + ((dementia_similarity - non_dementia_similarity) * 2.0))

        if risk_score < 0.4:
            risk_band = "low"
            risk_label = "Lower Dementia Pattern Match"
        elif risk_score < 0.6:
            risk_band = "medium"
            risk_label = "Moderate Dementia Pattern Match"
        else:
            risk_band = "high"
            risk_label = "Elevated Dementia Pattern Match"

        confidence = self._clip(0.25 + speech_completeness * 0.5 + visual_completeness * 0.25)
        quality_flags: list[str] = []
        if word_count < 35:
            quality_flags.append("limited_transcript")
        if frames_captured < 4:
            quality_flags.append("limited_visual_sampling")
        if face_detection_rate is None:
            quality_flags.append("face_detection_unavailable")
        elif face_detection_rate < 0.35:
            quality_flags.append("face_visibility_low")

        top_reasons = self._build_top_reasons(
            risk_score=risk_score,
            memory_difficulty=memory_difficulty,
            hesitation_rate=hesitation_rate,
            support_dependency=support_dependency,
            detail_density=detail_density,
            delayed_recall_score=delayed_recall_score,
            lexical_richness=lexical_richness,
            turn_elaboration=turn_elaboration,
        )
        recommendation = self._build_recommendation(risk_band, confidence)
        transcript_summary = (
            f"{patient_turn_count} patient turns, {word_count} words, "
            f"{request.audio_metrics.speech_seconds:.1f}s of speech, {frames_captured} visual frames."
        )

        speech_features = [
            FeatureSignal(
                label="Memory Retrieval Strain",
                value=memory_difficulty,
                summary="Higher values mean the patient used more uncertainty or memory-retrieval language.",
            ),
            FeatureSignal(
                label="Hesitation Rate",
                value=hesitation_rate,
                summary="Tracks restarts, filler markers, and brief verbal stalls across patient turns.",
            ),
            FeatureSignal(
                label="Narrative Detail",
                value=detail_density,
                summary="Higher values mean the patient gave more sequenced detail during recent-event prompts.",
            ),
            FeatureSignal(
                label="Conversation Recall",
                value=delayed_recall_score,
                summary="Measures whether the patient could briefly return to an earlier conversation detail near the end.",
            ),
        ]
        visual_features = [
            FeatureSignal(
                label="Face Continuity",
                value=face_presence,
                summary="How consistently a face remained visible across sampled frames.",
            ),
            FeatureSignal(
                label="Frame Coverage",
                value=face_coverage,
                summary="Approximate share of the frame occupied by the detected face when available.",
            ),
            FeatureSignal(
                label="Motion Proxy",
                value=movement_proxy,
                summary="A simple visual-activity proxy computed from frame-to-frame change.",
            ),
        ]

        trend = self._build_trend(patient_turns, overall_risk=risk_score)
        processing_summary = [
            f"Captured {request.audio_metrics.speech_seconds:.1f}s of speech and {frames_captured} sampled visual frames.",
            "Extracted speech cues from hesitation rate, wrap-up recall, daily-function language, and narrative detail.",
            "Compressed speech and face-stream proxies into embeddings and compared them with dementia and non-dementia centroids.",
            "Produced a single demo risk label, score, reasons, recommendation, and within-session trend.",
        ]

        return RealtimeAssessment(
            assessment_id=uuid.uuid4().hex,
            patient_id=request.patient_id,
            language=request.language,
            created_at=utc_now(),
            risk_label=risk_label,
            risk_score=round(risk_score, 2),
            risk_band=risk_band,
            confidence=round(confidence, 2),
            top_reasons=top_reasons,
            recommendation=recommendation,
            transcript_summary=transcript_summary,
            speech_features=speech_features,
            visual_features=visual_features,
            similarity=SimilarityBreakdown(
                dementia_pattern_similarity=round(dementia_similarity, 2),
                non_dementia_pattern_similarity=round(non_dementia_similarity, 2),
                speech_weight=speech_weight,
                visual_weight=visual_weight,
            ),
            trend=trend,
            processing_summary=processing_summary,
            quality_flags=quality_flags,
        )

    async def _run_live_qwen(
        self,
        websocket: WebSocket,
        patient_id: str,
        language: str,
    ) -> None:
        assert websockets is not None

        headers = {"Authorization": f"Bearer {self.settings.qwen_omni_api_key}"}
        selected_voice_profile = self._voice_profile_for_session(language_hint=language)
        realtime_urls = self._realtime_upstream_urls()
        last_error: Exception | None = None

        for attempt_index, realtime_url in enumerate(realtime_urls):
            url = f"{realtime_url}?model={self.settings.qwen_omni_realtime_model}"
            logger.info(
                "Opening Qwen realtime upstream connection patient_id=%s language=%s url=%s model=%s attempt=%s/%s",
                patient_id,
                language,
                realtime_url,
                self.settings.qwen_omni_realtime_model,
                attempt_index + 1,
                len(realtime_urls),
            )
            try:
                async with websockets.connect(
                    url,
                    additional_headers=headers,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                    proxy=None,
                ) as upstream:
                    logger.info("Qwen realtime upstream connected url=%s", realtime_url)
                    await self._relay_live_qwen_session(
                        websocket,
                        upstream,
                        patient_id=patient_id,
                        language=language,
                        selected_voice_profile=selected_voice_profile,
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if (
                    attempt_index + 1 < len(realtime_urls)
                    and self._should_retry_live_qwen_on_china_backup(exc)
                ):
                    logger.warning(
                        "Qwen realtime upstream handshake failed url=%s status=%s, retrying on China backup url=%s",
                        realtime_url,
                        self._realtime_handshake_status_code(exc),
                        realtime_urls[attempt_index + 1],
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error

    async def _relay_live_qwen_session(
        self,
        websocket: WebSocket,
        upstream: Any,
        *,
        patient_id: str,
        language: str,
        selected_voice_profile: RealtimeVoiceProfile,
    ) -> None:
            upstream_send_lock = asyncio.Lock()

            async def send_upstream_event(event: dict[str, Any]) -> None:
                if "event_id" not in event:
                    event["event_id"] = f"event_{uuid.uuid4().hex[:12]}"
                self._log_client_event(event)
                async with upstream_send_lock:
                    await upstream.send(json.dumps(event))

            async def send_session_update(profile: RealtimeVoiceProfile, *, reason: str) -> None:
                logger.info(
                    "Sending Qwen realtime session.update flow_id=%s steps=%s voice=%s language=%s reason=%s",
                    self.orchestrator.flow_id,
                    len(self.orchestrator.prompt_steps),
                    profile.voice,
                    profile.language_label,
                    reason,
                )
                await send_upstream_event(
                    self._build_live_session_update(
                        patient_id,
                        profile.language_label,
                        voice=profile.voice,
                    )
                )

            async def send_wrap_up_response() -> None:
                logger.info(
                    "Requesting realtime wrap-up patient_id=%s voice=%s language=%s",
                    patient_id,
                    selected_voice_profile.voice,
                    selected_voice_profile.language_label,
                )
                await send_upstream_event(
                    self._build_live_session_update(
                        patient_id,
                        selected_voice_profile.language_label,
                        voice=selected_voice_profile.voice,
                        wrap_up=True,
                    )
                )
                await send_upstream_event({"type": "response.create"})

            async def send_opening_response() -> None:
                logger.info(
                    "Requesting realtime opening patient_id=%s voice=%s language=%s",
                    patient_id,
                    selected_voice_profile.voice,
                    selected_voice_profile.language_label,
                )
                await send_upstream_event({"type": "response.create"})

            async def apply_voice_profile_to_session(
                profile: RealtimeVoiceProfile,
                *,
                reason: str,
            ) -> None:
                nonlocal selected_voice_profile
                await send_session_update(profile, reason=reason)
                selected_voice_profile = profile
                await websocket.send_json(
                    {
                        "type": "reflexion.voice.selected",
                        "voice": selected_voice_profile.voice,
                        "language": selected_voice_profile.language_label,
                        "language_key": selected_voice_profile.language_key,
                        "language_input": self._language_input_value(
                            selected_voice_profile.language_key,
                            selected_voice_profile.language_label,
                        ),
                        "source": selected_voice_profile.source,
                    }
                )

            await send_session_update(selected_voice_profile, reason=selected_voice_profile.source)
            deferred_voice_profile: RealtimeVoiceProfile | None = None
            recent_language_signals: list[RealtimeLanguageSignal] = []
            session_ready = False
            assistant_response_active = False
            assistant_response_done_count = 0
            transcript_turn_count = 0
            pending_first_response_restart = False
            opening_response_requested = False

            async def pump_upstream_to_client() -> None:
                nonlocal assistant_response_active
                nonlocal assistant_response_done_count
                nonlocal deferred_voice_profile
                nonlocal opening_response_requested
                nonlocal pending_first_response_restart
                nonlocal recent_language_signals
                nonlocal session_ready
                nonlocal transcript_turn_count
                async for message in upstream:
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("Qwen realtime upstream sent non-JSON payload preview=%r", message[:240])
                        continue
                    self._log_upstream_event(payload)
                    event_type = str(payload.get("type", ""))

                    if event_type == "conversation.item.input_audio_transcription.completed":
                        transcript_turn_count += 1
                        transcript_text = str(payload.get("transcript", ""))
                        language_signal = self._detect_language_signal_from_transcript(transcript_text)
                        if language_signal is not None:
                            recent_language_signals.append(language_signal)
                            recent_language_signals = recent_language_signals[-3:]

                        current_voice_profile = selected_voice_profile
                        detected_voice_profile = self._voice_profile_from_recent_signals(
                            language_hint=language,
                            recent_signals=recent_language_signals,
                            current_profile=current_voice_profile,
                        )
                        if detected_voice_profile is not None and (
                            detected_voice_profile.voice != current_voice_profile.voice
                            or detected_voice_profile.language_label != current_voice_profile.language_label
                        ):
                            should_restart_first_response = self._should_restart_response_for_language_switch(
                                transcript_turn_index=transcript_turn_count,
                                current_profile=current_voice_profile,
                                detected_profile=detected_voice_profile,
                                assistant_response_done_count=assistant_response_done_count,
                            )
                            logger.info(
                                "Realtime voice reassessment patient_id=%s from_voice=%s to_voice=%s from_language=%s to_language=%s transcript_preview=%r",
                                patient_id,
                                current_voice_profile.voice,
                                detected_voice_profile.voice,
                                current_voice_profile.language_label,
                                detected_voice_profile.language_label,
                                transcript_text[:120],
                            )
                            if session_ready:
                                try:
                                    await apply_voice_profile_to_session(
                                        detected_voice_profile,
                                        reason="transcript_reassessment",
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning(
                                        "Failed to update realtime voice from transcript reassessment: %s",
                                        exc,
                                    )
                            else:
                                deferred_voice_profile = detected_voice_profile
                            if should_restart_first_response:
                                if assistant_response_active and session_ready:
                                    logger.info(
                                        "Restarting first realtime reply patient_id=%s from_language=%s to_language=%s",
                                        patient_id,
                                        current_voice_profile.language_label,
                                        detected_voice_profile.language_label,
                                    )
                                    pending_first_response_restart = False
                                    assistant_response_active = False
                                    await send_upstream_event({"type": "response.cancel"})
                                    await send_upstream_event({"type": "response.create"})
                                else:
                                    pending_first_response_restart = True

                    if event_type == "session.updated":
                        if not session_ready:
                            session_ready = True
                        if deferred_voice_profile is not None:
                            try:
                                await apply_voice_profile_to_session(
                                    deferred_voice_profile,
                                    reason="transcript_reassessment",
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "Failed to apply deferred realtime voice update: %s",
                                    exc,
                                )
                            finally:
                                deferred_voice_profile = None
                        elif not opening_response_requested:
                            opening_response_requested = True
                            await send_opening_response()
                    if event_type == "response.created":
                        assistant_response_active = True
                        if pending_first_response_restart and assistant_response_done_count == 0:
                            logger.info(
                                "Restarting queued first realtime reply patient_id=%s voice=%s language=%s",
                                patient_id,
                                selected_voice_profile.voice,
                                selected_voice_profile.language_label,
                            )
                            pending_first_response_restart = False
                            assistant_response_active = False
                            await send_upstream_event({"type": "response.cancel"})
                            await send_upstream_event({"type": "response.create"})
                    elif event_type == "response.done":
                        assistant_response_active = False
                        assistant_response_done_count += 1

                    await websocket.send_json(payload)

            async def pump_client_to_upstream() -> None:
                nonlocal deferred_voice_profile, recent_language_signals, session_ready
                audio_append_started = False
                audio_window_chunks = 0
                audio_window_rms_total = 0.0
                audio_window_peak = 0.0
                audio_total_chunks = 0
                audio_last_rms = 0.0
                audio_window_started_at = asyncio.get_running_loop().time()
                while True:
                    event = await websocket.receive_json()
                    prepared_event, audio_append_started = self._prepare_live_client_event(
                        event,
                        audio_append_started=audio_append_started,
                    )
                    if prepared_event is None:
                        continue
                    event = prepared_event
                    event_type = str(event.get("type", ""))
                    if event_type == "reflexion.language_hint":
                        transcript_text = str(event.get("text", ""))
                        language_signal = self._detect_language_signal_from_transcript(transcript_text)
                        if language_signal is None:
                            continue
                        recent_language_signals.append(language_signal)
                        recent_language_signals = recent_language_signals[-3:]
                        detected_voice_profile = self._voice_profile_from_recent_signals(
                            language_hint=language,
                            recent_signals=recent_language_signals,
                            current_profile=selected_voice_profile,
                        )
                        if detected_voice_profile is None or (
                            detected_voice_profile.voice == selected_voice_profile.voice
                            and detected_voice_profile.language_label == selected_voice_profile.language_label
                        ):
                            continue
                        logger.info(
                            "Realtime voice hint patient_id=%s from_voice=%s to_voice=%s from_language=%s to_language=%s transcript_preview=%r",
                            patient_id,
                            selected_voice_profile.voice,
                            detected_voice_profile.voice,
                            selected_voice_profile.language_label,
                            detected_voice_profile.language_label,
                            transcript_text[:120],
                        )
                        if session_ready:
                            try:
                                await apply_voice_profile_to_session(
                                    detected_voice_profile,
                                    reason="browser_hint",
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "Failed to update realtime voice from browser hint: %s",
                                    exc,
                                )
                        else:
                            deferred_voice_profile = detected_voice_profile
                        continue
                    if event_type == "reflexion.wrap_up":
                        await send_wrap_up_response()
                        continue
                    if event_type == "reflexion.close":
                        logger.info("Browser requested realtime upstream close")
                        await upstream.close()
                        return
                    if event_type == "input_audio_buffer.append":
                        chunk_stats = self._audio_chunk_stats(str(event.get("audio", "")))
                        if chunk_stats is not None:
                            audio_total_chunks += 1
                            audio_window_chunks += 1
                            audio_window_rms_total += chunk_stats["rms"]
                            audio_window_peak = max(audio_window_peak, chunk_stats["peak"])
                            audio_last_rms = chunk_stats["rms"]
                            now = asyncio.get_running_loop().time()
                            if audio_window_chunks >= 12 or (now - audio_window_started_at) >= 1.0:
                                average_rms = audio_window_rms_total / max(1, audio_window_chunks)
                                logger.info(
                                    "Browser->Qwen audio append stats total_chunks=%s window_chunks=%s avg_rms=%.4f last_rms=%.4f peak=%.4f samples_per_chunk=%s",
                                    audio_total_chunks,
                                    audio_window_chunks,
                                    average_rms,
                                    audio_last_rms,
                                    audio_window_peak,
                                    chunk_stats["samples"],
                                )
                                audio_window_chunks = 0
                                audio_window_rms_total = 0.0
                                audio_window_peak = 0.0
                                audio_window_started_at = now
                    if "event_id" not in event:
                        event["event_id"] = f"event_{uuid.uuid4().hex[:12]}"
                    await send_upstream_event(event)

            upstream_task = asyncio.create_task(pump_upstream_to_client())
            client_task = asyncio.create_task(pump_client_to_upstream())
            done, pending = await asyncio.wait(
                {upstream_task, client_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()
            logger.info("Qwen realtime upstream relay finished cleanly")

    def _realtime_upstream_urls(self) -> list[str]:
        candidates = [
            str(self.settings.qwen_omni_realtime_url).strip(),
            str(self.settings.qwen_omni_realtime_url_china).strip(),
        ]
        urls: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in urls:
                urls.append(candidate)
        return urls

    def _realtime_handshake_status_code(self, exc: Exception) -> int | None:
        response = getattr(exc, "response", None)
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
        value = getattr(exc, "status_code", None)
        if isinstance(value, int):
            return value
        match = re.search(r"HTTP\s+(\d{3})", str(exc))
        if match:
            return int(match.group(1))
        return None

    def _should_retry_live_qwen_on_china_backup(self, exc: Exception) -> bool:
        return self._realtime_handshake_status_code(exc) in {401, 403}

    async def _run_guided_demo(
        self,
        websocket: WebSocket,
        *,
        language: str,
        patient_id: str | None = None,
    ) -> None:
        committed_turns = 0
        last_patient_text = ""
        patient_name, patient_memory = self._known_patient_context(patient_id)
        returning_patient = self.orchestrator._is_returning_patient(
            patient_name=patient_name,
            memory=patient_memory,
        )

        while True:
            event = await websocket.receive_json()
            event_type = str(event.get("type", ""))
            if event_type == "reflexion.close":
                return
            if event_type == "reflexion.patient_turn":
                last_patient_text = str(event.get("text", "")).strip()
                patient_name = patient_name or self.orchestrator.extract_patient_name(last_patient_text)
                continue
            if event_type == "input_audio_buffer.commit":
                committed_turns += 1
                await websocket.send_json(
                    {
                        "type": "input_audio_buffer.committed",
                        "turn_index": committed_turns,
                    }
                )
                continue
            if event_type != "response.create":
                continue

            response_id = f"guided-{committed_turns}"
            message = self._mock_reply(
                committed_turns,
                language=language,
                patient_text=last_patient_text,
                patient_name=patient_name,
                returning_patient=returning_patient,
            )
            await websocket.send_json({"type": "response.created", "response": {"id": response_id}})
            assembled = []
            for chunk in self._chunk_text(message):
                assembled.append(chunk)
                await websocket.send_json({"type": "response.text.delta", "delta": chunk})
                await asyncio.sleep(0.03)
            full_text = "".join(assembled)
            await websocket.send_json({"type": "response.text.done", "text": full_text})
            await websocket.send_json({"type": "response.done", "response": {"id": response_id}})
            last_patient_text = ""

    def _build_live_session_update(
        self,
        patient_id: str,
        language: str,
        *,
        voice: str | None = None,
        wrap_up: bool = False,
    ) -> dict[str, Any]:
        known_name, known_memory = self._known_patient_context(patient_id)
        instructions = self.orchestrator.build_live_instructions(
            patient_id,
            language,
            patient_name=known_name,
            memory=known_memory,
        )
        if wrap_up:
            instructions += (
                "\nThe live capture is ending now. In your next reply, briefly thank the patient, "
                "say the conversation is ending, and do not ask another question."
            )
        return {
            "event_id": f"event_{uuid.uuid4().hex[:12]}",
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": voice or self._voice_profile_for_session(language_hint=language).voice,
                "instructions": instructions,
                "max_tokens": self.settings.qwen_omni_realtime_max_tokens,
                "temperature": self.settings.qwen_omni_realtime_temperature,
                "top_p": self.settings.qwen_omni_realtime_top_p,
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.settings.qwen_omni_realtime_vad_threshold,
                    "prefix_padding_ms": self.settings.qwen_omni_realtime_vad_prefix_padding_ms,
                    "silence_duration_ms": self.settings.qwen_omni_realtime_vad_silence_duration_ms,
                    "create_response": True,
                    "interrupt_response": False,
                },
                "input_audio_transcription": {
                    "model": self.settings.qwen_omni_realtime_transcription_model,
                },
            },
        }

    def _mock_reply(
        self,
        committed_turns: int,
        *,
        language: str | None = None,
        patient_text: str | None = None,
        patient_name: str | None = None,
        returning_patient: bool = False,
    ) -> str:
        return self.orchestrator.guided_reply(
            committed_turns,
            language=language,
            patient_text=patient_text,
            patient_name=patient_name,
            returning_patient=returning_patient,
        )

    def _known_patient_context(self, patient_id: str | None) -> tuple[str | None, list[str]]:
        normalized_patient_id = str(patient_id or "").strip()
        if not normalized_patient_id:
            return None, []
        profile = self.identity.load_profile(normalized_patient_id)
        return normalize_patient_name(profile.preferred_name), [item for item in profile.memory if item]

    def _build_top_reasons(
        self,
        *,
        risk_score: float,
        memory_difficulty: float,
        hesitation_rate: float,
        support_dependency: float,
        detail_density: float,
        delayed_recall_score: float,
        lexical_richness: float,
        turn_elaboration: float,
    ) -> list[str]:
        if risk_score >= 0.5:
            candidates = [
                (memory_difficulty, "The transcript contained repeated uncertainty or memory-retrieval language."),
                (
                    1.0 - delayed_recall_score,
                    "The patient struggled to recall an earlier conversation detail near the end of the session.",
                ),
                (support_dependency, "Daily-function answers suggested reliance on reminders or help from others."),
                (hesitation_rate, "Speech contained multiple hesitations or verbal restarts."),
                (1.0 - detail_density, "Recent-event answers were sparse or weakly sequenced."),
            ]
        else:
            candidates = [
                (detail_density, "The patient maintained a sequenced narrative when describing recent events."),
                (
                    delayed_recall_score,
                    "The patient could return to an earlier conversation detail during the wrap-up recall.",
                ),
                (turn_elaboration, "Patient turns stayed reasonably elaborated for a brief screening conversation."),
                (lexical_richness, "Language variety remained intact across the short interview."),
                (1.0 - hesitation_rate, "Speech showed limited hesitation relative to the conversation length."),
            ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [reason for _, reason in candidates[:3]]

    def _build_recommendation(self, risk_band: str, confidence: float) -> str:
        if risk_band == "low":
            if confidence < 0.45:
                return "Repeat the capture with clearer audio/video before treating this as reassuring."
            return "Low demo risk. Keep routine monitoring and repeat if new concerns emerge."
        if risk_band == "medium":
            return "Mixed demo signal. Repeat the session or escalate to formal cognitive screening if concerns persist."
        return "Elevated demo risk. Recommend clinician review and formal cognitive evaluation rather than relying on this demo alone."

    def _score_delayed_recall_response(self, recall_text: str | None, *, reference_text: str | None = None) -> float:
        normalized_recall = str(recall_text or "").strip()
        if not normalized_recall:
            return 0.5

        lowered_recall = f" {normalized_recall.lower()} "
        memory_penalty = min(0.7, self._count_markers(lowered_recall, self.MEMORY_MARKERS) * 0.32)
        detail_bonus = min(0.2, self._count_markers(lowered_recall, self.DETAIL_MARKERS) * 0.04)
        recall_tokens = self._content_tokens(normalized_recall)
        token_bonus = min(0.3, len(recall_tokens) * 0.06)

        overlap_bonus = 0.0
        reference_tokens = self._content_tokens(reference_text or "")
        if recall_tokens and reference_tokens:
            overlap_bonus = min(0.4, len(recall_tokens & reference_tokens) * 0.16)

        answered_bonus = 0.18 if len(self._tokenize(normalized_recall)) >= 3 else 0.08
        return self._clip(answered_bonus + token_bonus + detail_bonus + overlap_bonus - memory_penalty)

    def _content_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in self._tokenize(text)
            if len(token) >= 4 and token not in self.LOW_SIGNAL_TOKENS
        }

    def _build_trend(self, patient_turns: list[Any], *, overall_risk: float) -> list[TrendPoint]:
        scores_by_stage: dict[str, float] = {}
        for step in self.orchestrator.prompt_steps:
            stage_text = " ".join(
                turn.text for turn in patient_turns if getattr(turn, "stage", None) == step.key and turn.text
            )
            if not stage_text:
                scores_by_stage[step.key] = overall_risk
                continue
            lowered = f" {stage_text.lower()} "
            if step.key == "orientation":
                score = 0.34
                score += self._count_markers(lowered, self.MEMORY_MARKERS) * 0.18
                score += self._count_markers(lowered, self.HESITATION_MARKERS) * 0.08
                score += max(0.0, 0.22 - self._count_markers(lowered, self.ORIENTATION_MARKERS) * 0.08)
            elif step.key == "recent_story":
                score = 0.32
                score += self._count_markers(lowered, self.HESITATION_MARKERS) * 0.08
                score += self._count_markers(lowered, self.MEMORY_MARKERS) * 0.15
                score += max(0.0, 0.35 - self._count_markers(lowered, self.DETAIL_MARKERS) * 0.06)
            elif step.key == "daily_function":
                score = 0.36 + self._count_markers(lowered, self.SUPPORT_MARKERS) * 0.18
            else:
                recent_story_text = " ".join(
                    turn.text for turn in patient_turns if getattr(turn, "stage", None) == "recent_story" and turn.text
                )
                recall_score = self._score_delayed_recall_response(
                    stage_text,
                    reference_text=recent_story_text,
                )
                score = 0.25 + (1.0 - recall_score) * 0.55
            scores_by_stage[step.key] = self._clip(score)

        return [
            TrendPoint(label=step.title, risk_score=round(scores_by_stage[step.key], 2))
            for step in self.orchestrator.prompt_steps
        ]

    def _live_qwen_ready(self) -> bool:
        return bool(self.settings.qwen_omni_api_key) and websockets is not None

    def _prepare_live_client_event(
        self,
        event: dict[str, Any],
        *,
        audio_append_started: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        event_type = str(event.get("type", ""))
        if event_type == "reflexion.close":
            return event, audio_append_started
        if event_type not in {
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
            "input_image_buffer.append",
            "response.create",
            "response.cancel",
            "reflexion.language_hint",
            "reflexion.wrap_up",
        }:
            logger.debug("Skipping unsupported browser realtime event type=%s", event_type)
            return None, audio_append_started
        if event_type == "input_audio_buffer.append":
            return event, True
        if event_type == "input_image_buffer.append" and not audio_append_started:
            logger.info("Dropping browser image frame until first audio append reaches realtime upstream")
            return None, audio_append_started
        return event, audio_append_started

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)

    def _count_markers(self, text: str, markers: tuple[str, ...]) -> int:
        return sum(text.count(marker) for marker in markers)

    def _count_unique_markers(self, text: str, markers: tuple[str, ...]) -> int:
        return sum(1 for marker in markers if marker and marker in text)

    def _normalize_language_key(self, language_hint: str | None) -> str | None:
        if not language_hint:
            return None
        normalized = re.sub(r"[\s_]+", " ", language_hint).strip().lower()
        for language_key, aliases in self.LANGUAGE_HINT_ALIASES.items():
            if normalized in aliases:
                return language_key
        return None

    def _detect_language_signal_from_transcript(self, transcript: str | None) -> RealtimeLanguageSignal | None:
        normalized = str(transcript or "").strip().lower()
        if not normalized:
            return None
        minnan_hits = self._count_unique_markers(normalized, self.MINNAN_MARKERS)
        if minnan_hits >= 1:
            return RealtimeLanguageSignal(
                language_key="minnan",
                confidence=0.95 if minnan_hits >= 2 else 0.82,
                source="transcript_reassessment",
            )
        cantonese_hits = self._count_unique_markers(normalized, self.CANTONESE_MARKERS)
        if cantonese_hits >= 1:
            return RealtimeLanguageSignal(
                language_key="cantonese",
                confidence=0.95 if cantonese_hits >= 2 else 0.82,
                source="transcript_reassessment",
            )
        english_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", normalized)
        english_words = len(english_tokens)
        english_function_hits = len(set(english_tokens) & self.ENGLISH_FUNCTION_WORDS)
        contains_cjk = any("\u4e00" <= char <= "\u9fff" for char in normalized)
        if english_words >= 3 and not contains_cjk:
            if english_words >= 4 and english_function_hits >= 2:
                confidence = 0.84
            elif english_words >= 3 and english_function_hits >= 2:
                confidence = 0.79
            elif english_words >= 6:
                confidence = 0.8
            elif english_function_hits >= 1:
                confidence = 0.74
            else:
                confidence = 0.65
            return RealtimeLanguageSignal(
                language_key="english",
                confidence=confidence,
                source="transcript_reassessment",
            )
        if contains_cjk:
            return RealtimeLanguageSignal(
                language_key="mandarin",
                confidence=0.72,
                source="transcript_reassessment",
            )
        return None

    def _detect_language_key_from_transcript(self, transcript: str | None) -> str | None:
        signal = self._detect_language_signal_from_transcript(transcript)
        return signal.language_key if signal is not None else None

    def _voice_profile_from_recent_signals(
        self,
        *,
        language_hint: str | None,
        recent_signals: list[RealtimeLanguageSignal],
        current_profile: RealtimeVoiceProfile,
    ) -> RealtimeVoiceProfile | None:
        if not recent_signals:
            return None

        latest_signal = recent_signals[-1]
        if latest_signal.language_key == current_profile.language_key:
            return None

        if latest_signal.language_key in {"minnan", "cantonese"} and latest_signal.confidence >= 0.8:
            return self._voice_profile_for_language_key(
                latest_signal.language_key,
                source=latest_signal.source,
            )

        if latest_signal.language_key == "english" and latest_signal.confidence >= 0.75:
            return self._voice_profile_for_language_key(
                latest_signal.language_key,
                source=latest_signal.source,
            )

        if latest_signal.confidence >= 0.9:
            return self._voice_profile_for_language_key(
                latest_signal.language_key,
                source=latest_signal.source,
            )

        if len(recent_signals) >= 2:
            last_two = recent_signals[-2:]
            if all(signal.language_key == latest_signal.language_key for signal in last_two):
                return self._voice_profile_for_language_key(
                    latest_signal.language_key,
                    source=latest_signal.source,
                )

        hinted_key = self._normalize_language_key(language_hint)
        if (
            hinted_key == current_profile.language_key
            and latest_signal.confidence >= 0.6
        ):
            return self._voice_profile_for_language_key(
                latest_signal.language_key,
                source=latest_signal.source,
            )

        return None

    def _should_restart_response_for_language_switch(
        self,
        *,
        transcript_turn_index: int,
        current_profile: RealtimeVoiceProfile,
        detected_profile: RealtimeVoiceProfile,
        assistant_response_done_count: int,
    ) -> bool:
        if transcript_turn_index != 1:
            return False
        if assistant_response_done_count > 0:
            return False
        return (
            detected_profile.voice != current_profile.voice
            or detected_profile.language_label != current_profile.language_label
        )

    def _language_input_value(self, language_key: str, language_label: str) -> str:
        if language_key == "english":
            return "en"
        if language_key == "mandarin":
            return "zh"
        if language_key == "minnan":
            return "nan"
        if language_key == "cantonese":
            return "yue"
        if language_key == "malay":
            return "ms"
        if language_key == "tamil":
            return "ta"
        return language_label

    def _default_voice_profile(self, *, source: str = "default") -> RealtimeVoiceProfile:
        return RealtimeVoiceProfile(
            language_key="mandarin",
            language_label="Mandarin Chinese",
            voice=self.settings.qwen_omni_realtime_default_voice,
            source=source,
        )

    def _voice_profile_for_language_key(self, language_key: str, *, source: str) -> RealtimeVoiceProfile:
        if language_key == "english":
            return RealtimeVoiceProfile(
                language_key="english",
                language_label="English",
                voice=self.settings.qwen_omni_realtime_english_voice,
                source=source,
            )
        if language_key == "mandarin":
            return RealtimeVoiceProfile(
                language_key="mandarin",
                language_label="Mandarin Chinese",
                voice=self.settings.qwen_omni_realtime_default_voice,
                source=source,
            )
        if language_key == "minnan":
            return RealtimeVoiceProfile(
                language_key="minnan",
                language_label="Minnan Chinese",
                voice=self.settings.qwen_omni_realtime_minnan_voice,
                source=source,
            )
        if language_key == "cantonese":
            return RealtimeVoiceProfile(
                language_key="cantonese",
                language_label="Cantonese",
                voice=self.settings.qwen_omni_realtime_cantonese_voice,
                source=source,
            )
        if language_key == "malay":
            return RealtimeVoiceProfile(
                language_key="malay",
                language_label="Malay",
                voice=self.settings.qwen_omni_realtime_default_voice,
                source=source,
            )
        if language_key == "tamil":
            return RealtimeVoiceProfile(
                language_key="tamil",
                language_label="Tamil",
                voice=self.settings.qwen_omni_realtime_default_voice,
                source=source,
            )
        return self._default_voice_profile(source=source)

    def _voice_profile_for_session(
        self,
        *,
        language_hint: str | None,
        transcript: str | None = None,
    ) -> RealtimeVoiceProfile:
        detected_key = self._detect_language_key_from_transcript(transcript)
        if detected_key is not None:
            return self._voice_profile_for_language_key(
                detected_key,
                source="transcript_reassessment",
            )

        hinted_key = self._normalize_language_key(language_hint)
        if hinted_key is not None:
            return self._voice_profile_for_language_key(
                hinted_key,
                source="language_hint",
            )

        clean_hint = str(language_hint or "").strip()
        if clean_hint:
            return RealtimeVoiceProfile(
                language_key="custom",
                language_label=clean_hint,
                voice=self.settings.qwen_omni_realtime_default_voice,
                source="language_hint",
            )

        return self._default_voice_profile()

    def _blend_with_neutral(self, value: float, completeness: float) -> float:
        return self._clip((0.5 * (1.0 - completeness)) + (value * completeness))

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(item * item for item in left))
        right_norm = math.sqrt(sum(item * item for item in right))
        if left_norm == 0 or right_norm == 0:
            return 0.5
        return self._clip(numerator / (left_norm * right_norm))

    def _clip(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _chunk_text(self, text: str, size: int = 14) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current = []
        current_len = 0
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= size:
                chunks.append(" ".join(current) + " ")
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def _audio_chunk_stats(self, audio_b64: str) -> dict[str, float | int] | None:
        if not audio_b64:
            return None
        try:
            raw = base64.b64decode(audio_b64)
        except (binascii.Error, ValueError):
            logger.info("Browser->Qwen audio append stats unavailable: invalid base64 audio payload")
            return None

        sample_count = len(raw) // 2
        if sample_count <= 0:
            return None

        sum_squares = 0.0
        peak = 0.0
        for index in range(0, sample_count * 2, 2):
            sample = int.from_bytes(raw[index : index + 2], byteorder="little", signed=True) / 32768.0
            absolute = abs(sample)
            sum_squares += sample * sample
            if absolute > peak:
                peak = absolute

        rms = math.sqrt(sum_squares / sample_count)
        return {
            "samples": sample_count,
            "rms": rms,
            "peak": peak,
        }

    def _log_client_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "input_audio_buffer.append":
            logger.debug(
                "Browser->Qwen event=%s event_id=%s audio_b64_len=%s",
                event_type,
                event.get("event_id"),
                len(str(event.get("audio", ""))),
            )
            return
        if event_type == "input_image_buffer.append":
            logger.info(
                "Browser->Qwen event=%s event_id=%s image_b64_len=%s",
                event_type,
                event.get("event_id"),
                len(str(event.get("image", ""))),
            )
            return
        logger.info(
            "Browser->Qwen event=%s event_id=%s",
            event_type,
            event.get("event_id"),
        )

    def _log_upstream_event(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type", ""))
        if event_type in {"response.audio.delta", "response.audio_transcript.delta", "response.text.delta"}:
            logger.debug("Qwen->Browser event=%s", event_type)
            return

        if event_type == "error":
            error = payload.get("error") or {}
            logger.warning(
                "Qwen->Browser event=%s code=%s message=%s",
                event_type,
                error.get("code"),
                error.get("message") or error,
            )
            return

        transcript = payload.get("transcript")
        text = payload.get("text")
        response = payload.get("response") or {}
        conversation_item = payload.get("item") or {}
        logger.info(
            "Qwen->Browser event=%s response_id=%s item_id=%s transcript_preview=%r text_preview=%r",
            event_type,
            response.get("id"),
            conversation_item.get("id"),
            str(transcript)[:120] if transcript is not None else None,
            str(text)[:120] if text is not None else None,
        )
