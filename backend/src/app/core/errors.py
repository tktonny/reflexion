"""Shared backend error types for provider routing and assessment failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderError(Exception):
    code: str
    message: str
    retryable: bool = True
    debug_details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class RoutingError(Exception):
    code: str
    message: str
    provider_trace: list[dict[str, object]]

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
