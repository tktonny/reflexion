"""OpenAI-based fallback providers for fusion and audio-only clinic review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from clinic.configs.settings import Settings
from backend.src.app.core.errors import ProviderError
from backend.src.app.models import ProviderCapabilities, ProviderContext, ProviderRawResult
from clinic.intelligence.prompts import build_provider_prompt
from clinic.intelligence.providers.base import BaseProvider


class _OpenAIFallbackBase(BaseProvider):
    def __init__(
        self,
        settings: Settings,
        *,
        name: str,
        description: str,
        model_name: str,
    ) -> None:
        super().__init__(
            settings,
            name=name,
            description=description,
            model_name=model_name,
            capabilities=ProviderCapabilities(
                native_video=False,
                native_audio_in_video=False,
                structured_output=True,
                requires_preprocessing=True,
                experimental=False,
            ),
        )
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url.rstrip("/")
        self.transcription_model = settings.openai_transcription_model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def _transcribe(self, audio_path: Path) -> str:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")},
                    data={"model": self.transcription_model},
                )
        except httpx.HTTPError as exc:
            raise ProviderError("provider_unavailable", f"{self.name} transcription failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                "provider_unavailable",
                f"{self.name} transcription failed: {response.text[:400]}",
            )
        payload = response.json()
        return payload.get("text", "")

    async def _run_chat_completion(self, user_content: list[dict[str, Any]], prompt, context: ProviderContext) -> ProviderRawResult:
        response = await self._request_json(
            method="POST",
            url=f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        )
        text = self._extract_chat_text(response)
        return ProviderRawResult(
            payload=self._parse_text_response(text),
            request_id=response.get("id"),
            raw_status="ok",
        )


class FusionProvider(_OpenAIFallbackBase):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            name="fusion",
            description="Fallback fusion provider using transcript plus key frames.",
            model_name=settings.openai_fusion_model,
        )

    async def analyze(self, provider_input: dict[str, Any], context: ProviderContext) -> ProviderRawResult:
        if not self.is_configured():
            if self.settings.allow_mock_providers:
                return self.mock_result(context)
            raise ProviderError("provider_unavailable", "fusion provider is not configured")

        transcript = ""
        audio_path = provider_input.get("audio_path")
        if audio_path:
            transcript = await self._transcribe(Path(audio_path))

        frame_paths = provider_input.get("frame_paths") or []
        if not frame_paths and not transcript:
            raise ProviderError("unsupported_media", "fusion provider requires audio or extracted frames")

        prompt = build_provider_prompt(context.patient_id, context.language, provider_mode="fusion")
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt.user_prompt}\n\n"
                    "This is a fallback fusion review.\n"
                    "Use the attached video frames as the visual channel and the transcript as the speech channel.\n"
                    "Do not infer prosody beyond what is explicitly supported by transcript or clear audiovisual evidence.\n"
                    f"Transcript:\n{transcript or '[transcript unavailable]'}"
                ),
            }
        ]

        for frame_path in frame_paths:
            data_url = self._encode_data_url(frame_path, "image/jpeg")
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        return await self._run_chat_completion(user_content, prompt, context)


class AudioOnlyProvider(_OpenAIFallbackBase):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            name="audio_only",
            description="Final fallback provider using only transcript-derived speech evidence.",
            model_name=settings.openai_text_model,
        )

    async def analyze(self, provider_input: dict[str, Any], context: ProviderContext) -> ProviderRawResult:
        if not self.is_configured():
            if self.settings.allow_mock_providers:
                return self.mock_result(context)
            raise ProviderError("provider_unavailable", "audio_only provider is not configured")

        audio_path = provider_input.get("audio_path")
        if not audio_path:
            raise ProviderError("unsupported_media", "audio_only provider requires extracted audio")
        transcript = await self._transcribe(Path(audio_path))
        if not transcript.strip():
            raise ProviderError("unsupported_media", "audio_only provider could not obtain a transcript")

        prompt = build_provider_prompt(context.patient_id, context.language, provider_mode="audio_only")
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt.user_prompt}\n\n"
                    "This is the final audio-only fallback.\n"
                    "No visual channel is available for this review.\n"
                    "Return empty visual_findings and body_findings unless explicitly null because the session is unusable.\n"
                    "Downgrade confidence and prefer needs_observation when visual evidence is required for a stronger call.\n"
                    f"Transcript:\n{transcript}"
                ),
            }
        ]

        return await self._run_chat_completion(user_content, prompt, context)
