"""Unit tests for the realtime conversation and demo risk-scoring service."""

from __future__ import annotations

import json
from pathlib import Path

from clinic.configs.settings import Settings
from backend.src.app.models import RealtimeAnalysisRequest
from backend.src.app.services.realtime_service import RealtimeConversationService


def write_flow_config(tmp_path: Path) -> Path:
    flow_path = tmp_path / "realtime_flow.json"
    flow_path.write_text(
        json.dumps(
            {
                "flow_id": "test-flow",
                "title": "Test Intake Flow",
                "opening_message": "Hello. Please tell me your full name and where you are right now.",
                "conversation_goal": "Collect a short structured intake across the four required stages.",
                "completion_rule": "Finish all stages unless the patient cannot continue.",
                "completion_message": "Thanks. The structured intake is complete.",
                "assistant_response_rules": [
                    "Acknowledge briefly before moving on.",
                    "Keep each response short.",
                ],
                "processing_steps": [
                    "Capture audio and frames.",
                    "Follow a configurable staged conversation plan.",
                ],
                "steps": [
                    {
                        "key": "orientation",
                        "title": "Orientation",
                        "goal": "Confirm self and place.",
                        "prompt": "Please tell me your full name and where you are right now.",
                        "rationale": "Checks orientation.",
                        "exit_when": [
                            "The patient gives a name or self-reference.",
                            "The patient gives a place or states uncertainty.",
                        ],
                        "max_follow_ups": 1,
                    },
                    {
                        "key": "recent_story",
                        "title": "Recent Story",
                        "goal": "Collect a sequenced narrative from earlier today.",
                        "prompt": "Walk me through what you did earlier today.",
                        "rationale": "Checks narrative continuity.",
                        "exit_when": [
                            "The patient gives at least two events.",
                        ],
                        "max_follow_ups": 1,
                    },
                    {
                        "key": "daily_function",
                        "title": "Daily Function",
                        "goal": "Understand how routines are managed.",
                        "prompt": "How are you managing meals, medicines, or appointments at home?",
                        "rationale": "Checks daily-function support.",
                        "exit_when": [
                            "The patient describes at least one routine.",
                        ],
                        "max_follow_ups": 1,
                    },
                    {
                        "key": "delayed_recall",
                        "title": "Wrap-up Recall",
                        "goal": "Check short recall near the end of the interview.",
                        "prompt": "Before we finish, can you tell me one thing you mentioned earlier about your day?",
                        "rationale": "Checks short recall.",
                        "exit_when": [
                            "The patient attempts recall.",
                        ],
                        "max_follow_ups": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return flow_path


def make_settings(
    tmp_path: Path,
    *,
    flow_path: Path | None = None,
    qwen_api_key: str | None = None,
) -> Settings:
    return Settings(
        app_name="test",
        storage_dir=tmp_path,
        uploads_dir=tmp_path / "uploads",
        prepared_dir=tmp_path / "prepared",
        assessments_dir=tmp_path / "assessments",
        server_host="127.0.0.1",
        server_port=8000,
        server_reload=False,
        max_upload_mb=100,
        max_inline_video_mb=5,
        allow_mock_providers=False,
        default_provider="qwen_omni",
        fallback_order=("qwen_omni", "gemini", "fusion", "audio_only"),
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        qwen_omni_api_key=qwen_api_key,
        qwen_omni_base_url="https://example.com/qwen",
        qwen_omni_model="qwen3.5-omni-plus",
        gemini_api_key=None,
        gemini_base_url="https://example.com/gemini",
        gemini_model="gemini-3.1-pro-preview",
        openai_api_key=None,
        openai_base_url="https://example.com/openai",
        openai_fusion_model="gpt-4.1",
        openai_text_model="gpt-4.1",
        openai_transcription_model="gpt-4o-transcribe",
        realtime_flow_path=flow_path,
    )


def test_status_defaults_to_guided_demo_when_live_qwen_is_unavailable(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    status = service.build_session_status()

    assert status.session_mode == "guided_demo"
    assert status.live_relay_available is False
    assert status.conversation_provider == "guided_demo"
    assert len(status.prompt_steps) == 4
    assert status.conversation_goal == "Collect a short structured intake across the four required stages."
    assert status.prompt_steps[0].goal == "Confirm self and place."
    assert status.prompt_steps[0].exit_when[0] == "The patient gives a name or self-reference."


def test_realtime_upstream_urls_include_china_backup_after_primary(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, flow_path=write_flow_config(tmp_path))
    settings = Settings(
        **{
            **settings.__dict__,
            "qwen_omni_realtime_url": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
            "qwen_omni_realtime_url_china": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        }
    )
    service = RealtimeConversationService(settings)

    assert service._realtime_upstream_urls() == [
        "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    ]


def test_realtime_retry_switches_on_handshake_401_or_403(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    class DummyError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)

    assert service._should_retry_live_qwen_on_china_backup(DummyError("server rejected WebSocket connection: HTTP 401"))
    assert service._should_retry_live_qwen_on_china_backup(DummyError("server rejected WebSocket connection: HTTP 403"))
    assert not service._should_retry_live_qwen_on_china_backup(DummyError("server rejected WebSocket connection: HTTP 500"))


def test_live_session_update_uses_server_vad(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    payload = service._build_live_session_update("patient-001", "en")
    instructions = payload["session"]["instructions"]

    assert payload["type"] == "session.update"
    assert payload["session"]["modalities"] == ["text", "audio"]
    assert payload["session"]["voice"] == "Cherry"
    assert payload["session"]["max_tokens"] == 48
    assert payload["session"]["temperature"] == 0.25
    assert payload["session"]["top_p"] == 0.7
    assert payload["session"]["output_audio_format"] == "pcm"
    assert payload["session"]["turn_detection"]["type"] == "server_vad"
    assert payload["session"]["turn_detection"]["threshold"] == 0.1
    assert payload["session"]["turn_detection"]["prefix_padding_ms"] == 500
    assert payload["session"]["turn_detection"]["silence_duration_ms"] == 900
    assert payload["session"]["turn_detection"]["create_response"] is True
    assert payload["session"]["turn_detection"]["interrupt_response"] is False
    assert payload["session"]["input_audio_transcription"]["model"] == "gummy-realtime-v1"
    assert "Conversation goal: Collect a short structured intake" in instructions
    assert "hidden guidance" in instructions
    assert "Exit when:" in instructions
    assert "Confirm self and place." in instructions
    assert 'For your first turn only, say exactly this opening in en: "Hi, nice to meet you. What should I call you? And where are you right now?"' in instructions
    assert "The local interface has already delivered the opening greeting" not in instructions


def test_live_status_reports_selected_voice_from_language_hint(tmp_path: Path) -> None:
    service = RealtimeConversationService(
        make_settings(
            tmp_path,
            flow_path=write_flow_config(tmp_path),
            qwen_api_key="test-key",
        )
    )

    status = service.build_session_status(preferred_language="粤语")

    assert status.session_mode == "live_qwen"
    assert status.selected_voice == "Kiki"
    assert status.selected_language == "Cantonese"
    assert status.voice_selection_source == "language_hint"
    assert status.max_session_seconds == 90
    assert status.max_reply_seconds == 7
    assert status.max_reply_chars == 140


def test_live_status_localizes_opening_greeting_for_cantonese(tmp_path: Path) -> None:
    service = RealtimeConversationService(
        make_settings(
            tmp_path,
            flow_path=write_flow_config(tmp_path),
            qwen_api_key="test-key",
        )
    )

    status = service.build_session_status(preferred_language="粤语")

    assert status.greeting == "你好，好高兴见到你。我应该点称呼你？你而家喺边度？"


def test_live_status_reports_malay_language_selection(tmp_path: Path) -> None:
    service = RealtimeConversationService(
        make_settings(
            tmp_path,
            flow_path=write_flow_config(tmp_path),
            qwen_api_key="test-key",
        )
    )

    status = service.build_session_status(preferred_language="ms")

    assert status.selected_voice == "Cherry"
    assert status.selected_language == "Malay"
    assert status.greeting == "Hai, gembira bertemu dengan anda. Saya patut panggil anda apa? Dan sekarang anda berada di mana?"


def test_live_status_reports_tamil_language_selection(tmp_path: Path) -> None:
    service = RealtimeConversationService(
        make_settings(
            tmp_path,
            flow_path=write_flow_config(tmp_path),
            qwen_api_key="test-key",
        )
    )

    status = service.build_session_status(preferred_language="ta")

    assert status.selected_voice == "Cherry"
    assert status.selected_language == "Tamil"
    assert status.greeting == "வணக்கம், உங்களை சந்தித்ததில் மகிழ்ச்சி. நான் உங்களை எப்படி அழைக்கலாம்? நீங்கள் இப்போது எங்கே இருக்கிறீர்கள்?"


def test_voice_profile_uses_english_voice_for_language_hint(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(language_hint="English")

    assert profile.voice == "Cherry"
    assert profile.language_label == "English"
    assert profile.source == "language_hint"


def test_voice_profile_uses_configured_english_voice_override(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, flow_path=write_flow_config(tmp_path))
    settings = Settings(
        **{
            **settings.__dict__,
            "qwen_omni_realtime_english_voice": "Jennifer",
        }
    )
    service = RealtimeConversationService(settings)

    profile = service._voice_profile_for_session(language_hint="English")

    assert profile.voice == "Jennifer"
    assert profile.language_label == "English"
    assert profile.source == "language_hint"


def test_voice_profile_uses_minnan_voice_for_language_hint(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(language_hint="闽南语")

    assert profile.voice == "Roy"
    assert profile.language_label == "Minnan Chinese"
    assert profile.source == "language_hint"


def test_voice_profile_uses_malay_default_voice_for_language_hint(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(language_hint="Malay")

    assert profile.voice == "Cherry"
    assert profile.language_label == "Malay"
    assert profile.source == "language_hint"


def test_voice_profile_uses_tamil_default_voice_for_language_hint(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(language_hint="Tamil")

    assert profile.voice == "Cherry"
    assert profile.language_label == "Tamil"
    assert profile.source == "language_hint"


def test_voice_profile_detects_minnan_voice_from_transcript(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(
        language_hint="zh",
        transcript="我今仔日毋知欲按怎讲，歹势。",
    )

    assert profile.voice == "Roy"
    assert profile.language_label == "Minnan Chinese"
    assert profile.source == "transcript_reassessment"


def test_voice_profile_detects_cantonese_voice_from_transcript(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(
        language_hint="zh",
        transcript="我而家喺屋企，佢啱啱食咗飯。",
    )

    assert profile.voice == "Kiki"
    assert profile.language_label == "Cantonese"
    assert profile.source == "transcript_reassessment"


def test_voice_profile_detects_english_voice_from_transcript(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(
        language_hint="zh",
        transcript="I had breakfast at home and then I went for a walk outside.",
    )

    assert profile.voice == "Cherry"
    assert profile.language_label == "English"
    assert profile.source == "transcript_reassessment"


def test_detect_language_signal_scores_short_clear_english_as_switchable(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    signal = service._detect_language_signal_from_transcript("I am Tony and I am home now.")

    assert signal is not None
    assert signal.language_key == "english"
    assert signal.confidence >= 0.75


def test_voice_profile_detects_mandarin_voice_from_transcript(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_for_session(
        language_hint="en",
        transcript="我今天在家里吃了早饭，然后出去散步。",
    )

    assert profile.voice == "Cherry"
    assert profile.language_label == "Mandarin Chinese"
    assert profile.source == "transcript_reassessment"


def test_recent_language_signal_switches_to_mandarin_after_single_cjk_turn(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_from_recent_signals(
        language_hint="en",
        recent_signals=[
            service._detect_language_signal_from_transcript("你好，我叫夏一川，我现在在我家里。")
        ],
        current_profile=service._voice_profile_for_session(language_hint="en"),
    )

    assert profile is not None
    assert profile.voice == "Cherry"
    assert profile.language_label == "Mandarin Chinese"


def test_recent_language_signal_switches_to_cantonese_after_single_strong_turn(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_from_recent_signals(
        language_hint="en",
        recent_signals=[
            service._detect_language_signal_from_transcript("我依家返屋企啦，頭先啱啱食咗飯。")
        ],
        current_profile=service._voice_profile_for_session(language_hint="en"),
    )

    assert profile is not None
    assert profile.voice == "Kiki"
    assert profile.language_label == "Cantonese"


def test_recent_language_signal_switches_to_minnan_after_single_strong_turn(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_from_recent_signals(
        language_hint="zh",
        recent_signals=[
            service._detect_language_signal_from_transcript("阮這馬欲轉去，今仔日有夠熱。")
        ],
        current_profile=service._voice_profile_for_session(language_hint="zh"),
    )

    assert profile is not None
    assert profile.voice == "Roy"
    assert profile.language_label == "Minnan Chinese"


def test_recent_language_signal_switches_to_english_after_single_clear_turn(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_from_recent_signals(
        language_hint="zh",
        recent_signals=[
            service._detect_language_signal_from_transcript(
                "I am at home right now and I had breakfast a little earlier today."
            )
        ],
        current_profile=service._voice_profile_for_session(language_hint="zh"),
    )

    assert profile is not None
    assert profile.voice == "Cherry"
    assert profile.language_label == "English"


def test_recent_language_signal_switches_to_english_after_short_clear_turn(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    profile = service._voice_profile_from_recent_signals(
        language_hint="zh",
        recent_signals=[
            service._detect_language_signal_from_transcript("I am Tony and I am home now.")
        ],
        current_profile=service._voice_profile_for_session(language_hint="zh"),
    )

    assert profile is not None
    assert profile.voice == "Cherry"
    assert profile.language_label == "English"


def test_first_turn_language_switch_requests_response_restart(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    should_restart = service._should_restart_response_for_language_switch(
        transcript_turn_index=1,
        current_profile=service._voice_profile_for_session(language_hint="zh"),
        detected_profile=service._voice_profile_for_session(
            language_hint="zh",
            transcript="I am Tony and I am home now.",
        ),
        assistant_response_done_count=0,
    )

    assert should_restart is True


def test_completed_first_reply_does_not_restart_for_language_switch(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    should_restart = service._should_restart_response_for_language_switch(
        transcript_turn_index=1,
        current_profile=service._voice_profile_for_session(language_hint="zh"),
        detected_profile=service._voice_profile_for_session(
            language_hint="zh",
            transcript="I am Tony and I am home now.",
        ),
        assistant_response_done_count=1,
    )

    assert should_restart is False


def test_live_session_update_accepts_voice_override(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    payload = service._build_live_session_update("patient-001", "Cantonese", voice="Kiki")

    assert payload["session"]["voice"] == "Kiki"
    assert "Respond in Cantonese" in payload["session"]["instructions"]


def test_live_session_update_wrap_up_instruction_is_brief(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    payload = service._build_live_session_update("patient-001", "Mandarin Chinese", wrap_up=True)

    assert "The live capture is ending now." in payload["session"]["instructions"]
    assert "do not ask another question" in payload["session"]["instructions"]


def test_live_relay_drops_image_until_first_audio_append(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    image_event, audio_started = service._prepare_live_client_event(
        {"type": "input_image_buffer.append", "image": "abc"},
        audio_append_started=False,
    )
    assert image_event is None
    assert audio_started is False

    audio_event, audio_started = service._prepare_live_client_event(
        {"type": "input_audio_buffer.append", "audio": "pcm"},
        audio_append_started=False,
    )
    assert audio_event is not None
    assert audio_started is True

    image_event, audio_started = service._prepare_live_client_event(
        {"type": "input_image_buffer.append", "image": "abc"},
        audio_append_started=audio_started,
    )
    assert image_event is not None
    assert audio_started is True


def test_guided_demo_reply_uses_configured_flow_order(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    assert service._mock_reply(
        1,
        patient_text="My name is Tony and I'm at home.",
        patient_name="Tony",
    ) == "Walk me through what you did earlier today."
    assert service._mock_reply(4) == "Thanks. The structured intake is complete."


def test_guided_demo_opening_reply_localizes_for_minnan(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    assert service._mock_reply(0, language="nan") == "你好，很欢喜见着你。我欲按怎称呼你？你这马佇佗位？"


def test_guided_demo_reply_uses_name_aware_transition_when_configured(tmp_path: Path) -> None:
    flow_path = tmp_path / "realtime_flow.json"
    flow_path.write_text(
        json.dumps(
            {
                "flow_id": "test-flow",
                "title": "Test Intake Flow",
                "opening_message": "Hi there. What should I call you, and where are you right now?",
                "conversation_goal": "Collect a short structured intake across the four required stages.",
                "completion_rule": "Finish all stages unless the patient cannot continue.",
                "completion_message": "Thanks{patient_name_clause}. We are done.",
                "assistant_response_rules": [
                    "Keep it conversational.",
                ],
                "processing_steps": [
                    "Capture audio and frames.",
                ],
                "steps": [
                    {
                        "key": "orientation",
                        "title": "Orientation",
                        "goal": "Confirm self and place.",
                        "prompt": "What should I call you, and where are you right now?",
                        "rationale": "Checks orientation.",
                        "exit_when": [
                            "The patient gives a name or self-reference.",
                            "The patient gives a place or states uncertainty.",
                        ],
                        "max_follow_ups": 1,
                    },
                    {
                        "key": "recent_story",
                        "title": "Recent Story",
                        "goal": "Collect a sequenced narrative from earlier today.",
                        "guided_transition": "Thanks{patient_name_clause}. I'd like to hear a little about how your day has been going.",
                        "prompt": "How has your day been so far?",
                        "rationale": "Checks narrative continuity.",
                        "exit_when": [
                            "The patient gives at least two events.",
                        ],
                        "max_follow_ups": 1,
                    },
                    {
                        "key": "daily_function",
                        "title": "Daily Function",
                        "goal": "Understand how routines are managed.",
                        "prompt": "How are meals, medicines, or appointments handled at home?",
                        "rationale": "Checks daily-function support.",
                        "exit_when": [
                            "The patient describes at least one routine.",
                        ],
                        "max_follow_ups": 1,
                    },
                    {
                        "key": "delayed_recall",
                        "title": "Wrap-up Recall",
                        "goal": "Check short recall near the end of the interview.",
                        "prompt": "Before we finish, can you tell me one thing you mentioned earlier about your day?",
                        "rationale": "Checks short recall.",
                        "exit_when": [
                            "The patient attempts recall.",
                        ],
                        "max_follow_ups": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=flow_path))

    reply = service._mock_reply(
        1,
        patient_text="My name is Tony and I'm at home.",
        patient_name="Tony",
    )

    assert reply == "Thanks, Tony. I'd like to hear a little about how your day has been going. How has your day been so far?"


def test_extract_patient_name_handles_english_and_chinese(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))

    assert service.orchestrator.extract_patient_name("My name is Tony and I'm at home.") == "Tony"
    assert service.orchestrator.extract_patient_name("我叫小明，我现在在家。") == "小明"


def test_analysis_elevates_risk_for_memory_heavy_conversation(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))
    request = RealtimeAnalysisRequest.model_validate(
        {
            "patient_id": "patient-high",
            "language": "en",
            "transcript": [
                {
                    "role": "patient",
                    "stage": "orientation",
                    "text": "Um, I am not sure where I am. I don't remember the clinic name.",
                },
                {
                    "role": "patient",
                    "stage": "recent_story",
                    "text": "I forgot what I did this morning. I can't remember and I am not sure.",
                },
                {
                    "role": "patient",
                    "stage": "daily_function",
                    "text": "My daughter helps me with medicines and appointments because I need help.",
                },
                {
                    "role": "patient",
                    "stage": "delayed_recall",
                    "text": "I do not remember what I mentioned earlier.",
                },
            ],
            "audio_metrics": {
                "utterance_count": 4,
                "speech_seconds": 29.4,
                "average_turn_seconds": 7.35,
                "audio_chunk_count": 18,
            },
            "visual_metrics": {
                "frames_captured": 12,
                "face_detection_rate": 0.42,
                "average_face_area": 0.16,
                "motion_intensity": 0.21,
                "mean_brightness": 0.51,
            },
        }
    )

    result = service.analyze_session(request)

    assert result.risk_band == "high"
    assert result.risk_score >= 0.6
    assert any("recall" in reason.lower() or "memory" in reason.lower() for reason in result.top_reasons)


def test_analysis_lowers_risk_for_coherent_detailed_conversation(tmp_path: Path) -> None:
    service = RealtimeConversationService(make_settings(tmp_path, flow_path=write_flow_config(tmp_path)))
    request = RealtimeAnalysisRequest.model_validate(
        {
            "patient_id": "patient-low",
            "language": "en",
            "transcript": [
                {
                    "role": "patient",
                    "stage": "orientation",
                    "text": "My name is Mary Chen, and I am at home in Shanghai today.",
                },
                {
                    "role": "patient",
                    "stage": "recent_story",
                    "text": (
                        "This morning I made breakfast, watered the plants, then walked downstairs to buy fruit "
                        "before reading the news."
                    ),
                },
                {
                    "role": "patient",
                    "stage": "daily_function",
                    "text": "I manage my medicines and appointments myself and keep reminders on my phone calendar.",
                },
                {
                    "role": "patient",
                    "stage": "delayed_recall",
                    "text": "Earlier I said I made breakfast and bought fruit.",
                },
            ],
            "audio_metrics": {
                "utterance_count": 4,
                "speech_seconds": 24.2,
                "average_turn_seconds": 6.05,
                "audio_chunk_count": 16,
            },
            "visual_metrics": {
                "frames_captured": 10,
                "face_detection_rate": 0.88,
                "average_face_area": 0.24,
                "motion_intensity": 0.44,
                "mean_brightness": 0.53,
            },
        }
    )

    result = service.analyze_session(request)

    assert result.risk_band == "low"
    assert result.risk_score <= 0.4
    assert any("recall" in reason.lower() or "narrative" in reason.lower() for reason in result.top_reasons)
