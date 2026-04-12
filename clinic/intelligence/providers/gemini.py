"""Gemini adapter for native multimodal clinic video assessments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError
from backend.src.app.models import ProviderCapabilities, ProviderContext, ProviderRawResult
from clinic.intelligence.prompts import build_provider_prompt
from clinic.intelligence.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            name="gemini",
            description="Google Gemini 3 native multimodal video understanding.",
            model_name=settings.gemini_model,
            capabilities=ProviderCapabilities(
                native_video=True,
                native_audio_in_video=True,
                structured_output=True,
                requires_preprocessing=False,
            ),
        )
        self.api_key = settings.gemini_api_key
        self.base_url = settings.gemini_base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def prepare(self, context: ProviderContext) -> dict[str, Any]:
        provider_input = await super().prepare(context)
        if not self.is_configured():
            return provider_input
        file_uri = await self._upload_file(Path(context.media.standardized_path), context.media.mime_type)
        provider_input["file_uri"] = file_uri
        return provider_input

    async def analyze(self, provider_input: dict[str, Any], context: ProviderContext) -> ProviderRawResult:
        if not self.is_configured():
            if self.settings.allow_mock_providers:
                return self.mock_result(context)
            raise ProviderError("provider_unavailable", "gemini is not configured")

        prompt = build_provider_prompt(context.patient_id, context.language, provider_mode="omni")
        payload = {
            "system_instruction": {"parts": [{"text": prompt.system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": context.media.mime_type,
                                "file_uri": provider_input["file_uri"],
                            }
                        },
                        {"text": prompt.user_prompt},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": prompt.response_schema,
            },
        }
        response = await self._request_json(
            method="POST",
            url=(
                f"{self.base_url}/v1beta/models/{self.model_name}:generateContent"
                f"?key={self.api_key}"
            ),
            headers={"Content-Type": "application/json"},
            json_body=payload,
        )
        candidates = response.get("candidates") or []
        if not candidates:
            raise ProviderError("invalid_provider_output", "gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = next((part.get("text") for part in parts if part.get("text")), "")
        if not text:
            raise ProviderError("invalid_provider_output", "gemini returned no text payload")
        return ProviderRawResult(
            payload=self._parse_text_response(text),
            request_id=response.get("responseId"),
            raw_status="ok",
        )

    async def _upload_file(self, path: Path, mime_type: str) -> str:
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        metadata = {"file": {"display_name": path.name}}
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                start_response = await client.post(
                    f"{self.base_url}/upload/v1beta/files?key={self.api_key}",
                    headers=headers,
                    json=metadata,
                )
        except httpx.HTTPError as exc:
            raise ProviderError("upload_failed", f"gemini upload initialization failed: {exc}") from exc
        if start_response.status_code >= 400:
            raise ProviderError(
                "upload_failed",
                f"gemini upload initialization failed: {start_response.text[:400]}",
            )
        upload_url = start_response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise ProviderError("upload_failed", "gemini upload URL missing")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                upload_response = await client.post(
                    upload_url,
                    headers={
                        "X-Goog-Upload-Command": "upload, finalize",
                        "X-Goog-Upload-Offset": "0",
                        "Content-Length": str(path.stat().st_size),
                    },
                    content=path.read_bytes(),
                )
        except httpx.HTTPError as exc:
            raise ProviderError("upload_failed", f"gemini upload failed: {exc}") from exc
        if upload_response.status_code >= 400:
            raise ProviderError("upload_failed", f"gemini upload failed: {upload_response.text[:400]}")
        payload = upload_response.json()
        file_uri = payload.get("file", {}).get("uri")
        if not file_uri:
            raise ProviderError("upload_failed", "gemini upload did not return a file uri")
        return file_uri
