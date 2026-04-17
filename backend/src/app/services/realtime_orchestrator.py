"""Configuration-driven conversation orchestration for the realtime interview flow."""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.src.app.models import (
    RealtimeConversationFlow,
    RealtimePromptStep,
)
from backend.src.app.services.patient_memory import extract_patient_name, normalize_patient_name
from clinic.configs.settings import Settings


class RealtimeConversationOrchestrator:
    """Load and expose the shared staged conversation plan used by live and guided modes."""

    DEFAULT_FLOW_FILENAME = "realtime_conversation_flow.json"
    OPENING_MESSAGES: dict[str, str] = {
        "english": "Hi, nice to meet you. What should I call you? And where are you right now?",
        "mandarin": "你好，很高兴见到你。我该怎么称呼你？你现在在哪里？",
        "cantonese": "你好，好高兴见到你。我应该点称呼你？你而家喺边度？",
        "minnan": "你好，很欢喜见着你。我欲按怎称呼你？你这马佇佗位？",
        "malay": "Hai, gembira bertemu dengan anda. Saya patut panggil anda apa? Dan sekarang anda berada di mana?",
        "tamil": "வணக்கம், உங்களை சந்தித்ததில் மகிழ்ச்சி. நான் உங்களை எப்படி அழைக்கலாம்? நீங்கள் இப்போது எங்கே இருக்கிறீர்கள்?",
    }
    RETURNING_OPENING_MESSAGES: dict[str, str] = {
        "english": "Hi, {patient_name}. Nice to see you again. How are you feeling today? What's on your mind this morning?",
        "mandarin": "你好，{patient_name}，很高兴再见到你。你今天感觉怎么样？今天早上你最想说的是什么？",
        "cantonese": "你好，{patient_name}，好高兴再见到你。你今日感觉点呀？今朝你心里面最想讲咩？",
        "minnan": "你好，{patient_name}，很欢喜阁再见着你。你今仔日感觉按怎？今仔早你心肝内上想讲的是啥物？",
        "malay": "Hai, {patient_name}. Gembira berjumpa lagi. Bagaimana perasaan anda hari ini? Apa yang ada dalam fikiran anda pagi ini?",
        "tamil": "வணக்கம், {patient_name}. உங்களை மீண்டும் சந்திப்பதில் மகிழ்ச்சி. இன்று நீங்கள் எப்படி உணர்கிறீர்கள்? இந்த காலை உங்கள் மனதில் என்ன இருக்கிறது?",
    }
    RETURNING_OPENING_MESSAGES_UNNAMED: dict[str, str] = {
        "english": "Hi, nice to see you again. How are you feeling today? What's on your mind this morning?",
        "mandarin": "你好，很高兴再见到你。你今天感觉怎么样？今天早上你最想说的是什么？",
        "cantonese": "你好，好高兴再见到你。你今日感觉点呀？今朝你心里面最想讲咩？",
        "minnan": "你好，很欢喜阁再见着你。你今仔日感觉按怎？今仔早你心肝内上想讲的是啥物？",
        "malay": "Hai, gembira berjumpa lagi. Bagaimana perasaan anda hari ini? Apa yang ada dalam fikiran anda pagi ini?",
        "tamil": "வணக்கம், உங்களை மீண்டும் சந்திப்பதில் மகிழ்ச்சி. இன்று நீங்கள் எப்படி உணர்கிறீர்கள்? இந்த காலை உங்கள் மனதில் என்ன இருக்கிறது?",
    }
    RETURNING_OPEN_PROMPTS: dict[str, str] = {
        "english": "How are you feeling today? What's on your mind this morning?",
        "mandarin": "你今天感觉怎么样？今天早上你最想说的是什么？",
        "cantonese": "你今日感觉点呀？今朝你心里面最想讲咩？",
        "minnan": "你今仔日感觉按怎？今仔早你心肝内上想讲的是啥物？",
        "malay": "Bagaimana perasaan anda hari ini? Apa yang ada dalam fikiran anda pagi ini?",
        "tamil": "இன்று நீங்கள் எப்படி உணர்கிறீர்கள்? இந்த காலை உங்கள் மனதில் என்ன இருக்கிறது?",
    }
    RETURNING_FOLLOW_UP_PROMPTS: dict[str, str] = {
        "english": "Tell me a little more about that.",
        "mandarin": "你可以再多说一点吗？",
        "cantonese": "你可以再讲多一点吗？",
        "minnan": "你会使阁讲较详细一点无？",
        "malay": "Boleh ceritakan sedikit lagi tentang itu?",
        "tamil": "அதைப் பற்றி இன்னும் கொஞ்சம் சொல்ல முடியுமா?",
    }
    RETURNING_CONVERSATION_GOAL = (
        "Collect a natural 90 to 120 second spoken check-in from a returning patient so acoustic cues stay stable and linguistic features can accumulate across sessions."
    )
    RETURNING_COMPLETION_RULE = (
        "Greet the returning patient, ask one open check-in prompt, allow them to speak freely, and use at most one brief follow-up before closing."
    )
    NO_MARKDOWN_RULE = (
        "Do not use markdown, asterisks, underscores, bullets, numbered lists, or stage directions. Speak in plain conversational sentences only."
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
            "putonghua",
            "普通话",
            "国语",
            "中文",
            "汉语",
            "漢語",
        ),
        "cantonese": (
            "yue",
            "zh-hk",
            "cantonese",
            "cantonese chinese",
            "guangdonghua",
            "广东话",
            "廣東話",
            "粤语",
            "粵語",
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

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.flow = self._load_flow()

    @property
    def prompt_steps(self) -> list[RealtimePromptStep]:
        return list(self.flow.steps)

    @property
    def opening_message(self) -> str:
        return self.flow.opening_message

    def opening_message_for_language(
        self,
        language: str | None,
        *,
        patient_name: str | None = None,
        returning_patient: bool = False,
    ) -> str:
        language_key = self._normalize_language_key(language)
        known_name = normalize_patient_name(patient_name)
        if returning_patient:
            if known_name:
                template = self.RETURNING_OPENING_MESSAGES.get(
                    language_key or "",
                    self.RETURNING_OPENING_MESSAGES["english"],
                )
                return template.format(patient_name=known_name)
            return self.RETURNING_OPENING_MESSAGES_UNNAMED.get(
                language_key or "",
                self.RETURNING_OPENING_MESSAGES_UNNAMED["english"],
            )
        if language_key is None:
            return self.flow.opening_message
        return self.OPENING_MESSAGES.get(language_key, self.flow.opening_message)

    @property
    def processing_steps(self) -> list[str]:
        return list(self.flow.processing_steps)

    @property
    def flow_id(self) -> str:
        return self.flow.flow_id

    @property
    def flow_title(self) -> str:
        return self.flow.title

    @property
    def conversation_goal(self) -> str:
        return self.flow.conversation_goal

    @property
    def completion_rule(self) -> str:
        return self.flow.completion_rule

    def guided_reply(
        self,
        committed_turns: int,
        *,
        language: str | None = None,
        patient_text: str | None = None,
        patient_name: str | None = None,
        returning_patient: bool = False,
    ) -> str:
        """Return the next guided-demo reply while keeping the flow conversational."""

        if committed_turns <= 0:
            return self.opening_message_for_language(
                language,
                patient_name=patient_name,
                returning_patient=returning_patient,
            )
        if returning_patient:
            if committed_turns == 1 and self._returning_follow_up_needed(patient_text):
                return self._returning_follow_up(language)
            return self._render_template(self.flow.completion_message, patient_name=patient_name)
        if committed_turns >= len(self.flow.steps):
            return self._render_template(self.flow.completion_message, patient_name=patient_name)

        next_step = self.flow.steps[committed_turns]
        transition = self._guided_transition_for_step(
            next_step,
            patient_text=patient_text,
            patient_name=patient_name,
        )
        prompt = next_step.prompt.strip()
        if not transition:
            return prompt
        return f"{transition} {prompt}".strip()

    def extract_patient_name(self, text: str | None) -> str | None:
        return extract_patient_name(text)

    def build_live_instructions(
        self,
        patient_id: str,
        language: str,
        *,
        patient_name: str | None = None,
        memory: list[str] | None = None,
    ) -> str:
        """Render the staged orchestration plan into one live-realtime instruction block."""

        language_name = language.strip() or "en"
        known_name = normalize_patient_name(patient_name)
        returning_patient = self._is_returning_patient(patient_name=known_name, memory=memory)
        opening_message = self.opening_message_for_language(
            language,
            patient_name=known_name,
            returning_patient=returning_patient,
        )
        response_rules = self.flow.assistant_response_rules or [
            "Sound like a calm, warm human guide rather than a robotic assistant.",
            "Treat the stage plan as hidden objectives, not lines to recite.",
            "Use a brief acknowledgement that fits what the patient just said before steering to the next topic.",
            "Keep replies extremely brief: usually one short sentence, never more than one short sentence plus one short question.",
            "Use plain everyday wording and stop speaking as soon as the next question is clear.",
            "Do not give summaries, long explanations, or multiple follow-up ideas in one turn.",
            "Do not diagnose, score risk, or discuss dementia probability during the live conversation.",
            "After the final stage is complete, thank the patient and say the session is complete.",
        ]
        if self.NO_MARKDOWN_RULE not in response_rules:
            response_rules = [*response_rules, self.NO_MARKDOWN_RULE]
        stage_blocks = "\n\n".join(
            self._format_stage_block(
                index,
                step,
                language=language,
                patient_name=known_name,
                returning_patient=returning_patient,
            )
            for index, step in enumerate(
                self.prompt_steps_for_session(
                    language=language,
                    patient_name=known_name,
                    memory=memory,
                ),
                start=1,
            )
        )
        rule_lines = "\n".join(f"- {rule}" for rule in response_rules)
        known_memory = [item.strip() for item in (memory or []) if str(item).strip()]
        memory_block = ""
        if known_name:
            memory_block += (
                f"The patient's known preferred name is {known_name}. "
                "Do not ask what to call them unless they correct you or offer a new preference.\n"
            )
        if known_memory:
            memory_lines = "\n".join(f"- {item}" for item in known_memory[:4])
            memory_block += (
                "Known patient memory from earlier sessions. Use it only as soft continuity context and let the "
                "patient correct anything that changed:\n"
                f"{memory_lines}\n"
            )
        orientation_rule = ""
        if returning_patient:
            orientation_rule = (
                "This is a returning patient check-in, not a first-time intake. Start with the open check-in prompt, "
                "listen, and use at most one brief follow-up.\n"
            )
        elif known_name:
            orientation_rule = (
                "For the orientation stage, the patient's name is already known. Confirm current place/orientation "
                "without re-asking the name unless the patient corrects it.\n"
            )

        return (
            "You are Reflexion, a calm and natural conversation guide conducting a short clinical intake.\n"
            f"The patient identifier is {patient_id}.\n"
            f"Respond in {language_name} unless the patient clearly switches languages.\n"
            "If the patient switches languages or dialects, immediately continue in that language on the next turn.\n"
            f"{memory_block}"
            f"Conversation flow: {self.flow_title_for_session(patient_name=known_name, memory=memory)}.\n"
            f"Conversation goal: {self.conversation_goal_for_session(patient_name=known_name, memory=memory)}\n"
            f"Completion rule: {self.completion_rule_for_session(patient_name=known_name, memory=memory)}\n"
            "The stage plan below is hidden guidance. The patient should feel like they are in a natural human conversation, not a checklist.\n"
            "Never mention stage names, scoring, assessment logic, or that you are an AI.\n"
            f"{orientation_rule}"
            f'For your first turn only, say exactly this opening in {language_name}: "{opening_message}"\n'
            "After the opening question, stop and wait for the patient's answer.\n"
            "Live response rules:\n"
            f"{rule_lines}\n"
            "Staged plan:\n"
            f"{stage_blocks}"
        )

    def flow_title_for_session(self, *, patient_name: str | None = None, memory: list[str] | None = None) -> str:
        if self._is_returning_patient(patient_name=patient_name, memory=memory):
            return "Returning Patient Daily Check-In"
        return self.flow.title

    def conversation_goal_for_session(self, *, patient_name: str | None = None, memory: list[str] | None = None) -> str:
        if self._is_returning_patient(patient_name=patient_name, memory=memory):
            return self.RETURNING_CONVERSATION_GOAL
        return self.flow.conversation_goal

    def completion_rule_for_session(self, *, patient_name: str | None = None, memory: list[str] | None = None) -> str:
        if self._is_returning_patient(patient_name=patient_name, memory=memory):
            return self.RETURNING_COMPLETION_RULE
        return self.flow.completion_rule

    def prompt_steps_for_session(
        self,
        *,
        language: str | None = None,
        patient_name: str | None = None,
        memory: list[str] | None = None,
    ) -> list[RealtimePromptStep]:
        if not self._is_returning_patient(patient_name=patient_name, memory=memory):
            return list(self.flow.steps)

        language_key = self._normalize_language_key(language)
        open_prompt = self.RETURNING_OPEN_PROMPTS.get(language_key or "", self.RETURNING_OPEN_PROMPTS["english"])
        follow_up_prompt = self.RETURNING_FOLLOW_UP_PROMPTS.get(
            language_key or "",
            self.RETURNING_FOLLOW_UP_PROMPTS["english"],
        )
        return [
            RealtimePromptStep(
                key="orientation",
                title="Open Check-In",
                goal="Invite the returning patient to speak freely about how they are feeling today and what is on their mind.",
                prompt=open_prompt,
                rationale="This produces a more natural block of patient speech for stable acoustic analysis across repeated sessions.",
                exit_when=[
                    "The patient has started a natural spoken update in their own words.",
                    "Do not interrupt quickly; allow enough voiced speech to accumulate before deciding whether a follow-up is needed.",
                ],
                max_follow_ups=0,
            ),
            RealtimePromptStep(
                key="recent_story",
                title="Brief Follow-Up",
                goal="Optionally deepen one concrete thread with a single short follow-up if the first answer was brief or unclear.",
                prompt=follow_up_prompt,
                rationale="One brief follow-up helps capture enough usable speech while keeping the interaction conversational.",
                exit_when=[
                    "Use this only if the first answer was short, vague, or ended too quickly.",
                    "Ask at most one brief follow-up, then thank the patient and close.",
                ],
                max_follow_ups=1,
            ),
        ]

    def _normalize_language_key(self, language_hint: str | None) -> str | None:
        normalized = re.sub(r"[\s_]+", " ", str(language_hint or "")).strip().lower()
        if not normalized:
            return None
        for language_key, aliases in self.LANGUAGE_HINT_ALIASES.items():
            if normalized in aliases:
                return language_key
        return None

    def _load_flow(self) -> RealtimeConversationFlow:
        path = self.settings.realtime_flow_path or self.default_flow_path()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RealtimeConversationFlow.model_validate(payload)

    def default_flow_path(self) -> Path:
        return Path(__file__).resolve().parents[4] / "clinic" / "configs" / self.DEFAULT_FLOW_FILENAME

    def _format_stage_block(
        self,
        index: int,
        step: RealtimePromptStep,
        *,
        language: str | None = None,
        patient_name: str | None = None,
        returning_patient: bool = False,
    ) -> str:
        exit_rules = "\n".join(f"  - {rule}" for rule in step.exit_when) or "  - Move on after one answer."
        transition_line = ""
        if step.guided_transition:
            transition_line = f"  Natural bridge: {step.guided_transition}\n"
        prompt = step.prompt
        known_name = normalize_patient_name(patient_name)
        if not returning_patient and known_name and step.key == "orientation":
            prompt = "Please tell me where you are right now."
        return (
            f"{index}. {step.title} ({step.key})\n"
            f"  Goal: {step.goal}\n"
            f"{transition_line}"
            f"  Primary prompt: {prompt}\n"
            f"  Rationale: {step.rationale}\n"
            f"  Exit when:\n{exit_rules}\n"
            f"  If needed, ask at most {step.max_follow_ups} gentle clarification question(s) before moving on."
        )

    def _guided_transition_for_step(
        self,
        step: RealtimePromptStep,
        *,
        patient_text: str | None,
        patient_name: str | None,
    ) -> str:
        if step.guided_transition:
            return self._render_template(step.guided_transition, patient_name=patient_name)
        return ""

    def _render_template(self, template: str, *, patient_name: str | None) -> str:
        name = (patient_name or "").strip()
        patient_name_clause = f", {name}" if name else ""
        rendered = template.format(
            patient_name=name,
            patient_name_clause=patient_name_clause,
        )
        return re.sub(r"\s+", " ", rendered).strip()

    def _is_returning_patient(self, *, patient_name: str | None = None, memory: list[str] | None = None) -> bool:
        return bool(normalize_patient_name(patient_name) or [item for item in (memory or []) if str(item).strip()])

    def _returning_follow_up(self, language: str | None) -> str:
        language_key = self._normalize_language_key(language)
        return self.RETURNING_FOLLOW_UP_PROMPTS.get(
            language_key or "",
            self.RETURNING_FOLLOW_UP_PROMPTS["english"],
        )

    def _returning_follow_up_needed(self, patient_text: str | None) -> bool:
        normalized = str(patient_text or "").strip()
        if not normalized:
            return True
        token_count = len(re.findall(r"[\w']+", normalized, flags=re.UNICODE))
        return token_count < 18 or len(normalized) < 90
