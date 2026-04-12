"""Qwen 3.5 Omni adapter for native video-plus-audio clinic assessments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError
from backend.src.app.models import ProviderCapabilities, ProviderContext, ProviderRawResult
from clinic.intelligence.prompts import build_provider_prompt
from clinic.intelligence.providers.base import BaseProvider


class QwenOmniProvider(BaseProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            name="qwen_omni",
            description="Primary full multimodal provider using Qwen 3.5 Omni native video-plus-audio understanding.",
            model_name=settings.qwen_omni_model,
            capabilities=ProviderCapabilities(
                native_video=True,
                native_audio_in_video=True,
                structured_output=True,
                requires_preprocessing=False,
            ),
        )
        self.api_key = settings.qwen_omni_api_key
        self.base_url = settings.qwen_omni_base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def prepare(self, context: ProviderContext) -> dict[str, Any]:
        provider_input = await super().prepare(context)
        if not self.is_configured():
            return provider_input
        if context.media.mime_type == "video/webm":
            raise ProviderError(
                "unsupported_media",
                (
                    "qwen_omni cannot analyze browser-recorded WebM uploads directly. "
                    "Install ffmpeg on the server so the upload can be standardized to MP4 first."
                ),
                debug_details={
                    "standardized_path": context.media.standardized_path,
                    "mime_type": context.media.mime_type,
                },
            )
        provider_limit_bytes = self.settings.qwen_omni_inline_video_mb * 1024 * 1024
        if context.media.size_bytes > provider_limit_bytes:
            raise ProviderError(
                "unsupported_media",
                (
                    f"qwen_omni inline video exceeds the provider limit of "
                    f"{self.settings.qwen_omni_inline_video_mb}MB."
                ),
                debug_details={
                    "standardized_path": context.media.standardized_path,
                    "mime_type": context.media.mime_type,
                    "size_bytes": context.media.size_bytes,
                    "provider_limit_bytes": provider_limit_bytes,
                    "file_name": Path(context.media.standardized_path).name,
                },
            )
        provider_input["inline_video_url"] = self._encode_data_url(
            context.media.standardized_path,
            context.media.mime_type,
        )
        return provider_input

    async def analyze(self, provider_input: dict[str, Any], context: ProviderContext) -> ProviderRawResult:
        if not self.is_configured():
            if self.settings.allow_mock_providers:
                return self.mock_result(context)
            raise ProviderError("provider_unavailable", "qwen_omni is not configured")

        prompt = build_provider_prompt(context.patient_id, context.language, provider_mode="omni")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.user_prompt},
                        {
                            "type": "video_url",
                            "video_url": {"url": provider_input["inline_video_url"]},
                        },
                    ],
                },
            ],
            "stream": True,
            "modalities": ["text"],
            "temperature": 0.1,
        }

        try:
            request_id, text = await self._stream_text(payload)
            return ProviderRawResult(
                payload=self._parse_text_response(text),
                request_id=request_id,
                raw_status="ok",
            )
        except ProviderError as exc:
            debug_details = dict(exc.debug_details or {})
            debug_details.setdefault(
                "request_summary",
                {
                    "model": self.model_name,
                    "provider_mode": "omni",
                    "stream": True,
                    "modalities": ["text"],
                },
            )
            debug_details.setdefault(
                "media_summary",
                {
                    "standardized_path": context.media.standardized_path,
                    "mime_type": context.media.mime_type,
                    "size_bytes": context.media.size_bytes,
                    "duration_seconds": context.media.duration_seconds,
                },
            )
            exc.debug_details = debug_details
            raise

    async def _stream_text(self, payload: dict[str, Any]) -> tuple[str | None, str]:
        request_id: str | None = None
        chunks: list[str] = []

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise ProviderError(
                            "provider_unavailable",
                            f"qwen_omni returned {response.status_code}: {body[:400].decode(errors='ignore')}",
                            debug_details={
                                "response_status_code": response.status_code,
                                "response_body_preview": body[:12000].decode(errors="ignore"),
                            },
                        )
                    request_id = response.headers.get("x-request-id")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        request_id = request_id or event.get("id")
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            chunks.extend(self._extract_delta_text(delta))
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", "qwen_omni timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("provider_unavailable", f"qwen_omni request failed: {exc}") from exc

        text = "".join(chunks).strip()
        if not text:
            raise ProviderError(
                "invalid_provider_output",
                "qwen_omni returned no parseable text",
                debug_details={
                    "response_chunk_count": len(chunks),
                    "raw_text_preview": "".join(chunks)[:12000],
                },
            )
        return request_id, text

    def _extract_delta_text(self, delta: dict[str, Any]) -> list[str]:
        content = delta.get("content")
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
            return parts
        return []
