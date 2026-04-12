"""Shared provider abstractions and helpers for multimodal inference backends."""

from __future__ import annotations

import base64
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError
from backend.src.app.core.json_utils import parse_json_object
from backend.src.app.models import (
    DEFAULT_DISCLAIMER,
    ProviderAssessmentPayload,
    ProviderCapabilities,
    ProviderContext,
    ProviderName,
    ProviderRawResult,
    ProviderStatus,
    normalize_quality_flags,
)


class BaseProvider(ABC):
    """Base contract for all provider adapters in the clinic intelligence layer."""
    def __init__(
        self,
        settings: Settings,
        *,
        name: ProviderName,
        description: str,
        model_name: str,
        capabilities: ProviderCapabilities,
    ) -> None:
        self.settings = settings
        self.name = name
        self.description = description
        self.model_name = model_name
        self.capabilities = capabilities

    @abstractmethod
    def is_configured(self) -> bool:
        raise NotImplementedError

    async def healthcheck(self, fallback_rank: int) -> ProviderStatus:
        configured = self.is_configured()
        available = configured or self.settings.allow_mock_providers
        return ProviderStatus(
            provider=self.name,
            available=available,
            configured=configured,
            mock_mode=bool(not configured and self.settings.allow_mock_providers),
            fallback_rank=fallback_rank,
            description=self.description,
            capabilities=self.capabilities,
        )

    async def prepare(self, context: ProviderContext) -> dict[str, Any]:
        return {
            "standardized_path": context.media.standardized_path,
            "audio_path": context.media.extracted_audio_path,
            "frame_paths": context.media.frame_paths,
        }

    @abstractmethod
    async def analyze(self, provider_input: dict[str, Any], context: ProviderContext) -> ProviderRawResult:
        raise NotImplementedError

    def normalize(self, raw_result: ProviderRawResult) -> ProviderAssessmentPayload:
        payload = raw_result.payload
        payload.quality_flags = normalize_quality_flags(payload.quality_flags)
        if not payload.disclaimer:
            payload.disclaimer = DEFAULT_DISCLAIMER
        return payload

    def mock_result(self, context: ProviderContext) -> ProviderRawResult:
        seed = hashlib.sha256(
            f"{self.name}:{context.patient_id}:{Path(context.media.standardized_path).name}".encode()
        ).hexdigest()
        confidence = 0.42 + (int(seed[:2], 16) / 2550)
        risk_label = "cognitive_risk" if confidence >= 0.5 else "HC"
        quality_flags = []
        if not context.media.frame_paths:
            quality_flags.append("limited_visual_sampling")
        if not context.media.extracted_audio_path:
            quality_flags.append("poor_audio")
        payload = ProviderAssessmentPayload(
            visual_findings=[
                {
                    "label": "mock_visual_screen",
                    "summary": f"{self.name} mock analysis produced a placeholder visual review.",
                    "evidence": "Enable a real API key to replace mock findings.",
                    "confidence": 0.35,
                }
            ],
            body_findings=[
                {
                    "label": "mock_body_screen",
                    "summary": "Body movement findings are placeholder values in mock mode.",
                    "confidence": 0.3,
                }
            ],
            voice_findings=[
                {
                    "label": "mock_voice_screen",
                    "summary": "Voice findings are placeholder values in mock mode.",
                    "confidence": 0.3,
                }
            ],
            content_findings=[
                {
                    "label": "mock_content_screen",
                    "summary": "Content findings are placeholder values in mock mode.",
                    "confidence": 0.3,
                }
            ],
            risk_score=round(min(confidence, 0.95), 2),
            risk_label=risk_label,  # type: ignore[arg-type]
            quality_flags=quality_flags or ["mock_provider"],
            session_usability="usable_with_caveats",
            disclaimer=DEFAULT_DISCLAIMER,
        )
        return ProviderRawResult(
            payload=payload,
            request_id=f"mock-{seed[:12]}",
            raw_status="mock",
        )

    async def _request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        data: dict[str, Any] | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    files=files,
                    data=data,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", f"{self.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("provider_unavailable", f"{self.name} request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                "provider_unavailable",
                f"{self.name} returned {response.status_code}: {response.text[:400]}",
            )
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError("invalid_provider_output", f"{self.name} did not return JSON") from exc

    def _encode_data_url(self, path: str, mime_type: str) -> str:
        raw = Path(path).read_bytes()
        if len(raw) > self.settings.max_inline_video_mb * 1024 * 1024:
            raise ProviderError(
                "unsupported_media",
                f"{self.name} inline media exceeds {self.settings.max_inline_video_mb}MB limit",
            )
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _parse_text_response(self, text: str) -> ProviderAssessmentPayload:
        try:
            data = parse_json_object(text)
        except ValueError as exc:
            raise ProviderError(
                "invalid_provider_output",
                f"{self.name} response did not contain a valid JSON object",
                debug_details=self._build_text_debug_details(text),
            ) from exc
        data = self._normalize_assessment_payload_dict(data)
        try:
            return ProviderAssessmentPayload.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            debug_details = self._build_text_debug_details(text)
            debug_details["parsed_json_preview"] = json.dumps(data, ensure_ascii=False)[:12000]
            raise ProviderError(
                "invalid_provider_output",
                f"{self.name} returned an invalid assessment payload",
                debug_details=debug_details,
            ) from exc

    def _extract_chat_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderError("invalid_provider_output", f"{self.name} response had no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
        raise ProviderError("invalid_provider_output", f"{self.name} response had no parseable content")

    def _build_text_debug_details(self, text: str) -> dict[str, object]:
        return {
            "raw_text_length": len(text),
            "raw_text_preview": text[:12000],
        }

    def _normalize_assessment_payload_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(data)

        raw_risk_label = normalized.get("risk_label")
        raw_screening_classification = normalized.get("screening_classification")

        if (
            self._normalize_risk_label(raw_risk_label) is None
            and self._normalize_screening_classification(raw_screening_classification) is None
            and self._normalize_risk_label(raw_screening_classification) is not None
            and self._normalize_screening_classification(raw_risk_label) is not None
        ):
            normalized["risk_label"] = raw_screening_classification
            normalized["screening_classification"] = raw_risk_label
            raw_risk_label = normalized.get("risk_label")
            raw_screening_classification = normalized.get("screening_classification")

        normalized_risk_label = self._normalize_risk_label(raw_risk_label)
        if normalized_risk_label is not None:
            normalized["risk_label"] = normalized_risk_label

        normalized_screening_classification = self._normalize_screening_classification(
            raw_screening_classification
        )
        if normalized_screening_classification is not None:
            normalized["screening_classification"] = normalized_screening_classification

        normalized_risk_tier = self._normalize_risk_tier(normalized.get("risk_tier"))
        if normalized_risk_tier is not None:
            normalized["risk_tier"] = normalized_risk_tier

        return normalized

    def _normalize_risk_label(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        risk_label_aliases = {
            "hc": "HC",
            "healthy_control": "HC",
            "control": "HC",
            "cognitive_risk": "cognitive_risk",
            "cognitive_decline_risk": "cognitive_risk",
        }
        return risk_label_aliases.get(cleaned)

    def _normalize_screening_classification(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        screening_aliases = {
            "healthy": "healthy",
            "needs_observation": "needs_observation",
            "observe": "needs_observation",
            "monitor": "needs_observation",
            "monitoring": "needs_observation",
            "dementia": "dementia",
        }
        return screening_aliases.get(cleaned)

    def _normalize_risk_tier(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        risk_tier_aliases = {
            "low": "low",
            "medium": "medium",
            "moderate": "medium",
            "mid": "medium",
            "high": "high",
        }
        return risk_tier_aliases.get(cleaned)
