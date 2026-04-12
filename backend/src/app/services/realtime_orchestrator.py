"""Configuration-driven conversation orchestration for the realtime interview flow."""

from __future__ import annotations

import json
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

    def guided_reply(self, committed_turns: int) -> str:
        """Return the next guided-demo assistant reply based on completed patient turns."""

        if committed_turns <= 0:
            return self.flow.opening_message
        if committed_turns >= len(self.flow.steps):
            return self.flow.completion_message

        next_step = self.flow.steps[committed_turns]
        return f"Thank you. {next_step.prompt}"

    def build_live_instructions(self, patient_id: str, language: str) -> str:
        """Render the staged orchestration plan into one live-realtime instruction block."""

        language_name = language.strip() or "en"
        response_rules = self.flow.assistant_response_rules or [
            "Acknowledge briefly before moving to the next step.",
            "Keep every response under 18 words.",
            "Do not diagnose, score risk, or discuss dementia probability during the live conversation.",
            "After the final stage is complete, thank the patient and say the session is complete.",
        ]
        stage_blocks = "\n\n".join(
            self._format_stage_block(index, step)
            for index, step in enumerate(self.flow.steps, start=1)
        )
        rule_lines = "\n".join(f"- {rule}" for rule in response_rules)

        return (
            "You are Reflexion, a calm clinical screening assistant conducting a short structured interview.\n"
            f"The patient identifier is {patient_id}.\n"
            f"Respond in {language_name} unless the patient clearly switches languages.\n"
            f"Conversation flow: {self.flow.title}.\n"
            f"Conversation goal: {self.flow.conversation_goal}\n"
            f"Completion rule: {self.flow.completion_rule}\n"
            "The local interface has already delivered the opening greeting and the first prompt.\n"
            "The next patient audio turn answers stage 1.\n"
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
        return (
            f"{index}. {step.title} ({step.key})\n"
            f"  Goal: {step.goal}\n"
            f"  Primary prompt: {step.prompt}\n"
            f"  Rationale: {step.rationale}\n"
            f"  Exit when:\n{exit_rules}\n"
            f"  If needed, ask at most {step.max_follow_ups} brief clarification question(s) before moving on."
        )
