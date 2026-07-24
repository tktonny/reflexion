"""Helpers for extracting structured JSON objects from model responses."""

from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    if not payload:
        raise ValueError("empty provider payload")
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(payload)
    if not match:
        raise ValueError("provider output did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("provider JSON root was not an object")
    return parsed
