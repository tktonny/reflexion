"""Configuration-driven conversation orchestration for the realtime interview flow."""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.src.app.models import (
    RealtimeConversationFlow,
    RealtimePromptStep,
)
from clinic.configs.settings import Settings


class RealtimeConversationOrchestrator:
    """Load and expose the shared staged conversation plan used by live and guided modes."""

    DEFAULT_FLOW_FILENAME = "realtime_conversation_flow.json"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.flow = self._load_flow()

    @property
    def prompt_steps(self) -> list[RealtimePromptStep]:
        return list(self.flow.steps)

    @property
    def opening_message(self) -> str:
        return self.flow.opening_message

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
        patient_text: str | None = None,
        patient_name: str | None = None,
    ) -> str:
        """Return the next guided-demo reply while keeping the flow conversational."""

        if committed_turns <= 0:
            return self.flow.opening_message
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
        """Best-effort extraction of the patient's preferred spoken name."""

        normalized = str(text or "").strip()
        if not normalized:
            return None

        patterns = (
            r"\bmy name is\s+([A-Z][A-Za-z' -]{0,40})",
            r"\bi am\s+([A-Z][A-Za-z' -]{0,40})",
            r"\bi'm\s+([A-Z][A-Za-z' -]{0,40})",
            r"\bcall me\s+([A-Z][A-Za-z' -]{0,40})",
            r"我叫([^\s，。,.]{1,12})",
            r"我是([^\s，。,.]{1,12})",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" ,.")
            candidate = re.sub(r"\b(and|but|right now|currently)\b.*$", "", candidate, flags=re.IGNORECASE).strip(
                " ,."
            )
            if not candidate:
                continue
            parts = [part for part in candidate.split() if part]
            if len(parts) > 3:
                candidate = " ".join(parts[:3])
            first_token = candidate.split()[0].lower() if candidate.split() else candidate.lower()
            if first_token in {"at", "in", "home", "here", "hospital", "clinic", "fine", "okay", "good"}:
                continue
            return candidate
        return None

    def build_live_instructions(self, patient_id: str, language: str) -> str:
        """Render the staged orchestration plan into one live-realtime instruction block."""

        language_name = language.strip() or "en"
        response_rules = self.flow.assistant_response_rules or [
            "Sound like a calm, warm human guide rather than a robotic assistant.",
            "Treat the stage plan as hidden objectives, not lines to recite.",
            "Use a brief acknowledgement that fits what the patient just said before steering to the next topic.",
            "Keep replies concise, usually one or two short sentences.",
            "Do not diagnose, score risk, or discuss dementia probability during the live conversation.",
            "After the final stage is complete, thank the patient and say the session is complete.",
        ]
        stage_blocks = "\n\n".join(
            self._format_stage_block(index, step)
            for index, step in enumerate(self.flow.steps, start=1)
        )
        rule_lines = "\n".join(f"- {rule}" for rule in response_rules)

        return (
            "You are Reflexion, a calm and natural conversation guide conducting a short clinical intake.\n"
            f"The patient identifier is {patient_id}.\n"
            f"Respond in {language_name} unless the patient clearly switches languages.\n"
            f"Conversation flow: {self.flow.title}.\n"
            f"Conversation goal: {self.flow.conversation_goal}\n"
            f"Completion rule: {self.flow.completion_rule}\n"
            "The stage plan below is hidden guidance. The patient should feel like they are in a natural human conversation, not a checklist.\n"
            "Never mention stage names, scoring, assessment logic, or that you are an AI.\n"
            "The local interface has already delivered the opening greeting and the first question.\n"
            "The next patient audio turn answers the opening orientation question.\n"
            "Live response rules:\n"
            f"{rule_lines}\n"
            "Staged plan:\n"
            f"{stage_blocks}"
        )

    def _load_flow(self) -> RealtimeConversationFlow:
        path = self.settings.realtime_flow_path or self.default_flow_path()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RealtimeConversationFlow.model_validate(payload)

    def default_flow_path(self) -> Path:
        return Path(__file__).resolve().parents[4] / "clinic" / "configs" / self.DEFAULT_FLOW_FILENAME

    def _format_stage_block(self, index: int, step: RealtimePromptStep) -> str:
        exit_rules = "\n".join(f"  - {rule}" for rule in step.exit_when) or "  - Move on after one answer."
        transition_line = ""
        if step.guided_transition:
            transition_line = f"  Natural bridge: {step.guided_transition}\n"
        return (
            f"{index}. {step.title} ({step.key})\n"
            f"  Goal: {step.goal}\n"
            f"{transition_line}"
            f"  Primary prompt: {step.prompt}\n"
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
