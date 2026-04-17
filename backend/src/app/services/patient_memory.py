"""Helpers for extracting reusable patient memory from session transcripts."""

from __future__ import annotations

import re
from typing import Any


def normalize_patient_name(name: str | None) -> str | None:
    candidate = re.sub(r"\s+", " ", str(name or "")).strip(" ,.")
    if not candidate:
        return None
    parts = [part for part in candidate.split() if part]
    if len(parts) > 3:
        candidate = " ".join(parts[:3])
    candidate = candidate[:40].rstrip(" ,.")
    return candidate or None


def extract_patient_name(text: str | None) -> str | None:
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
        candidate = re.sub(
            r"\b(and|but|right now|currently)\b.*$",
            "",
            match.group(1).strip(" ,."),
            flags=re.IGNORECASE,
        ).strip(" ,.")
        candidate = normalize_patient_name(candidate)
        if not candidate:
            continue
        first_token = candidate.split()[0].lower() if " " in candidate else candidate.lower()
        if first_token in {"at", "in", "home", "here", "hospital", "clinic", "fine", "okay", "good"}:
            continue
        return candidate
    return None


def build_patient_memory(session_record: dict[str, Any] | None) -> tuple[str | None, list[str]]:
    """Extract a lightweight memory summary from the latest conversation."""

    patient_turns = _patient_turns(session_record)
    if not patient_turns:
        return None, []

    preferred_name = None
    for turn in patient_turns:
        preferred_name = preferred_name or extract_patient_name(turn)

    recent_story = _select_turn_text(
        session_record,
        patient_turns,
        stage_names={"recent_story", "stage-1"},
        fallback_index=1,
    )
    daily_function = _select_turn_text(
        session_record,
        patient_turns,
        stage_names={"daily_function", "stage-2"},
        fallback_index=2,
    )

    memory: list[str] = []
    if preferred_name:
        memory.append(f"Preferred name: {preferred_name}.")
    if recent_story:
        memory.append(f"Recent activity mentioned: {_summarize_memory_text(recent_story)}")
    if daily_function:
        memory.append(f"Routine/support detail: {_summarize_memory_text(daily_function)}")
    return preferred_name, _dedupe(memory)


def _patient_turns(session_record: dict[str, Any] | None) -> list[str]:
    transcript_turns = (
        session_record.get("derivedFeatures", {})
        .get("speech", {})
        .get("transcriptTurns", [])
        if isinstance(session_record, dict)
        else []
    )
    patient_turns: list[str] = []
    for turn in transcript_turns:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role", "")).strip().lower() != "patient":
            continue
        text = str(turn.get("text", "")).strip()
        if text:
            patient_turns.append(text)
    return patient_turns


def _select_turn_text(
    session_record: dict[str, Any] | None,
    patient_turns: list[str],
    *,
    stage_names: set[str],
    fallback_index: int,
) -> str | None:
    transcript_turns = (
        session_record.get("derivedFeatures", {})
        .get("speech", {})
        .get("transcriptTurns", [])
        if isinstance(session_record, dict)
        else []
    )
    for turn in transcript_turns:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role", "")).strip().lower() != "patient":
            continue
        if str(turn.get("stage", "")).strip() not in stage_names:
            continue
        text = str(turn.get("text", "")).strip()
        if text:
            return text
    if 0 <= fallback_index < len(patient_turns):
        return patient_turns[fallback_index]
    return None


def _summarize_memory_text(text: str, *, max_chars: int = 140) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    if len(clean) > max_chars:
        clean = clean[: max_chars - 1].rstrip(" ,.;:") + "…"
    if clean[-1] not in ".!?。！？…":
        clean += "."
    return clean


def _dedupe(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in seen:
            cleaned.append(stripped)
            seen.add(stripped)
    return cleaned
