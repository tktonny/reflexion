"""Helpers for extracting reusable patient memory from session transcripts."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from backend.src.app.core.json_utils import parse_json_object

PREFERRED_NAME_PREFIX = "Preferred name:"
RECENT_ACTIVITY_PREFIX = "Recent activity mentioned:"
ROUTINE_SUPPORT_PREFIX = "Routine/support detail:"
PATIENT_DETAIL_PREFIX = "Patient detail:"
LLM_TRANSCRIPT_CHAR_LIMIT = 2400
LLM_MAX_OTHER_ITEMS = 2

logger = logging.getLogger("uvicorn.error")


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


def build_patient_memory(
    session_record: dict[str, Any] | None,
    *,
    settings: Any | None = None,
) -> tuple[str | None, list[str]]:
    """Extract a lightweight memory summary from the latest conversation."""

    patient_turns = _patient_turns(session_record)
    if not patient_turns:
        return None, []

    fallback_name, fallback_memory = _build_rule_based_patient_memory(
        session_record,
        patient_turns,
    )
    if not _patient_memory_llm_ready(settings):
        return fallback_name, fallback_memory

    try:
        llm_name, llm_memory = _build_patient_memory_with_llm(
            session_record,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Patient memory LLM extraction failed; using rule-based fallback: %s",
            exc,
        )
        return fallback_name, fallback_memory

    resolved_name = fallback_name or normalize_patient_name(llm_name)
    merged_memory = _merge_llm_with_fallback_memory(
        llm_memory=llm_memory,
        fallback_memory=fallback_memory,
        preferred_name=resolved_name,
    )
    return resolved_name, merged_memory or fallback_memory


def _build_rule_based_patient_memory(
    session_record: dict[str, Any] | None,
    patient_turns: list[str],
) -> tuple[str | None, list[str]]:
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
    preferred_name_entry = preferred_name_memory_entry(preferred_name)
    if preferred_name_entry:
        memory.append(preferred_name_entry)
    if recent_story:
        memory.append(f"{RECENT_ACTIVITY_PREFIX} {_summarize_memory_text(recent_story)}")
    if daily_function:
        memory.append(f"{ROUTINE_SUPPORT_PREFIX} {_summarize_memory_text(daily_function)}")
    return preferred_name, _dedupe(memory)


def preferred_name_memory_entry(name: str | None) -> str | None:
    normalized_name = normalize_patient_name(name)
    if not normalized_name:
        return None
    return f"{PREFERRED_NAME_PREFIX} {normalized_name}."


def merge_patient_memories(
    existing_memory: list[str] | None,
    new_memory: list[str] | None,
    *,
    preferred_name: str | None = None,
) -> list[str]:
    merged: list[str] = []
    preferred_entry = preferred_name_memory_entry(preferred_name)
    if preferred_entry:
        merged.append(preferred_entry)
    merged.extend(_without_preferred_name_entries(new_memory or []))
    merged.extend(_without_preferred_name_entries(existing_memory or []))
    return _dedupe(merged)


def _patient_memory_llm_ready(settings: Any | None) -> bool:
    if settings is None:
        return False
    if not bool(getattr(settings, "patient_memory_use_llm", True)):
        return False
    return bool(str(getattr(settings, "patient_memory_api_key", "") or "").strip())


def _build_patient_memory_with_llm(
    session_record: dict[str, Any] | None,
    *,
    settings: Any,
) -> tuple[str | None, list[str]]:
    transcript = _patient_transcript_for_llm(session_record)
    if not transcript:
        return None, []

    payload = _request_patient_memory_payload(
        transcript=transcript,
        settings=settings,
    )
    preferred_name = _normalize_llm_name(payload.get("preferred_name"))

    memory: list[str] = []
    preferred_entry = preferred_name_memory_entry(preferred_name)
    if preferred_entry:
        memory.append(preferred_entry)

    if recent_activity := _llm_memory_entry(
        RECENT_ACTIVITY_PREFIX,
        payload.get("recent_activity"),
    ):
        memory.append(recent_activity)
    if routine_support := _llm_memory_entry(
        ROUTINE_SUPPORT_PREFIX,
        payload.get("routine_support"),
    ):
        memory.append(routine_support)

    raw_other_items = payload.get("other_memory_items")
    if isinstance(raw_other_items, list):
        for item in raw_other_items[:LLM_MAX_OTHER_ITEMS]:
            if memory_item := _llm_memory_entry(PATIENT_DETAIL_PREFIX, item):
                memory.append(memory_item)

    return preferred_name, _dedupe(memory)


def _request_patient_memory_payload(
    *,
    transcript: str,
    settings: Any,
) -> dict[str, Any]:
    api_key = str(getattr(settings, "patient_memory_api_key", "") or "").strip()
    model = str(getattr(settings, "patient_memory_model", "") or "qwen3.5-plus").strip()
    base_url = str(
        getattr(settings, "patient_memory_base_url", "") or "https://coding.dashscope.aliyuncs.com/v1"
    ).rstrip("/")
    timeout = float(getattr(settings, "patient_memory_timeout_seconds", 20.0) or 20.0)
    system_prompt, user_prompt = _build_patient_memory_prompts(transcript)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt}],
                        },
                    ],
                    "temperature": 0.1,
                    "stream": False,
                },
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError("patient memory LLM timed out") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"patient memory LLM request failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"patient memory LLM returned {response.status_code}: {response.text[:400]}"
        )
    try:
        response_payload = response.json()
    except ValueError as exc:
        raise RuntimeError("patient memory LLM did not return JSON") from exc

    return parse_json_object(_extract_chat_text(response_payload))


def _build_patient_memory_prompts(transcript: str) -> tuple[str, str]:
    system_prompt = (
        "You extract lightweight patient memory for future follow-up conversations. "
        "Use only facts clearly stated by the patient. "
        "Return strict JSON only with no markdown or extra commentary."
    )
    user_prompt = (
        "Summarize the patient-only transcript into reusable cross-session memory.\n"
        "Return exactly one JSON object with these keys:\n"
        "{\n"
        '  "preferred_name": string | null,\n'
        '  "recent_activity": string | null,\n'
        '  "routine_support": string | null,\n'
        '  "other_memory_items": string[]\n'
        "}\n\n"
        "Rules:\n"
        "- Use only facts clearly spoken by the patient.\n"
        "- Keep each string under 140 characters.\n"
        "- preferred_name is only the name the patient prefers to be called.\n"
        "- recent_activity is a recent or same-day activity that would help a natural follow-up.\n"
        "- routine_support is a routine, daily-function, reminders, or support detail.\n"
        "- other_memory_items may contain up to 2 additional stable details.\n"
        "- Do not include labels, bullets, diagnosis, risk, scores, speculation, or caregiver-only claims.\n"
        "- If something is unclear, use null or an empty array.\n\n"
        "Patient-only transcript:\n"
        f"{transcript}"
    )
    return system_prompt, user_prompt


def _patient_transcript_for_llm(session_record: dict[str, Any] | None) -> str:
    transcript_turns = (
        session_record.get("derivedFeatures", {})
        .get("speech", {})
        .get("transcriptTurns", [])
        if isinstance(session_record, dict)
        else []
    )
    lines: list[str] = []
    patient_index = 0
    for turn in transcript_turns:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role", "")).strip().lower() != "patient":
            continue
        text = re.sub(r"\s+", " ", str(turn.get("text", "")).strip())
        if not text:
            continue
        patient_index += 1
        stage = str(turn.get("stage", "")).strip() or "unlabeled"
        lines.append(f"{patient_index}. [{stage}] {text}")
    transcript = "\n".join(lines).strip()
    if len(transcript) > LLM_TRANSCRIPT_CHAR_LIMIT:
        transcript = transcript[: LLM_TRANSCRIPT_CHAR_LIMIT - 1].rstrip(" ,.;:") + "…"
    return transcript


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("patient memory LLM response had no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
    raise RuntimeError("patient memory LLM response had no parseable content")


def _normalize_llm_name(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = _strip_memory_prefixes(text)
    return normalize_patient_name(text)


def _llm_memory_entry(prefix: str, value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    clean = _strip_memory_prefixes(raw)
    summarized = _summarize_memory_text(clean)
    if not summarized:
        return None
    return f"{prefix} {summarized}"


def _strip_memory_prefixes(text: str) -> str:
    return re.sub(
        rf"^\s*(?:{re.escape(PREFERRED_NAME_PREFIX)}|{re.escape(RECENT_ACTIVITY_PREFIX)}|"
        rf"{re.escape(ROUTINE_SUPPORT_PREFIX)}|{re.escape(PATIENT_DETAIL_PREFIX)})\s*",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    ).strip(" -:,.")


def _merge_llm_with_fallback_memory(
    *,
    llm_memory: list[str],
    fallback_memory: list[str],
    preferred_name: str | None,
) -> list[str]:
    merged: list[str] = []
    preferred_entry = preferred_name_memory_entry(preferred_name)
    if preferred_entry:
        merged.append(preferred_entry)

    llm_without_pref = _without_preferred_name_entries(llm_memory)
    merged.extend(llm_without_pref)

    if not _find_memory_with_prefix(llm_without_pref, RECENT_ACTIVITY_PREFIX):
        if fallback_recent := _find_memory_with_prefix(fallback_memory, RECENT_ACTIVITY_PREFIX):
            merged.append(fallback_recent)
    if not _find_memory_with_prefix(llm_without_pref, ROUTINE_SUPPORT_PREFIX):
        if fallback_routine := _find_memory_with_prefix(fallback_memory, ROUTINE_SUPPORT_PREFIX):
            merged.append(fallback_routine)
    if not llm_without_pref:
        merged.extend(_without_preferred_name_entries(fallback_memory))

    return _dedupe(merged)


def _find_memory_with_prefix(items: list[str], prefix: str) -> str | None:
    for item in items:
        if str(item).strip().startswith(prefix):
            return item.strip()
    return None


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


def _without_preferred_name_entries(items: list[str]) -> list[str]:
    return [item for item in items if not _is_preferred_name_entry(item)]


def _is_preferred_name_entry(item: str) -> bool:
    return str(item).strip().lower().startswith(PREFERRED_NAME_PREFIX.lower())


def _dedupe(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        stripped = item.strip()
        if stripped and stripped not in seen:
            cleaned.append(stripped)
            seen.add(stripped)
    return cleaned
