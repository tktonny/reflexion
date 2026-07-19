"""DashScope-backed ASR plus Qwen 3.5 transcript classification for verification."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from backend.src.app.core.errors import ProviderError
from backend.src.app.models import ProviderCapabilities, ProviderContext, ProviderRawResult
from clinic.configs.settings import get_settings
from clinic.intelligence.prompts import build_provider_prompt
from clinic.intelligence.providers.base import BaseProvider
from verification.config import VerificationSettings
from verification.model_routing import resolve_model_endpoint


MAX_DATA_URL_AUDIO_BYTES = 7 * 1024 * 1024


class QwenAudioOnlyVerifier(BaseProvider):
    def __init__(self, verification_settings: VerificationSettings) -> None:
        super().__init__(
            get_settings(),
            name="audio_only",
            description="Verification-only Qwen ASR plus Qwen 3.5 transcript classifier.",
            model_name=verification_settings.qwen_text_model,
            capabilities=ProviderCapabilities(
                native_video=False,
                native_audio_in_video=False,
                structured_output=True,
                requires_preprocessing=True,
            ),
        )
        self.verification_settings = verification_settings
        self.classifier_endpoint = resolve_model_endpoint(
            verification_settings,
            verification_settings.qwen_text_model,
        )
        self.asr_endpoint = resolve_model_endpoint(
            verification_settings,
            verification_settings.qwen_asr_model,
        )
        self.api_key = self.classifier_endpoint.api_key
        self.base_url = self.classifier_endpoint.base_url
        self.asr_model = verification_settings.qwen_asr_model

    def is_configured(self) -> bool:
        return bool(self.classifier_endpoint.api_key and self.asr_endpoint.api_key)

    async def analyze(self, provider_input: dict[str, Any], context: ProviderContext) -> ProviderRawResult:
        raise NotImplementedError("Verification calls transcribe_audio and classify_transcript directly")

    async def transcribe_audio(
        self,
        *,
        audio_path: str,
        mime_type: str,
        language: str | None = None,
    ) -> tuple[str, str | None]:
        if not self.asr_endpoint.api_key:
            raise ProviderError(
                "provider_unavailable",
                f"{self.asr_model} is not configured. Set QWEN_API_KEY for standard Qwen models.",
            )

        payload = {
            "model": self.asr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": self._encode_audio_data_url(Path(audio_path), mime_type),
                            },
                        }
                    ],
                }
            ],
            "stream": False,
            "extra_body": {
                "asr_options": {
                    "enable_itn": False,
                    **({"language": language} if language else {}),
                }
            },
        }
        response = await self._request_json(
            method="POST",
            url=f"{self.asr_endpoint.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.asr_endpoint.api_key}",
                "Content-Type": "application/json",
            },
            json_body=payload,
            timeout=self.verification_settings.request_timeout_seconds,
        )
        return self._extract_chat_text(response).strip(), response.get("id")

    async def classify_transcript(
        self,
        *,
        patient_id: str,
        language: str,
        transcript: str,
    ) -> tuple[dict[str, Any], str | None]:
        if not self.classifier_endpoint.api_key:
            raise ProviderError(
                "provider_unavailable",
                f"{self.model_name} is not configured. Set DASHSCOPE_API_KEY for Coding Plan models.",
            )

        prompt = build_provider_prompt(patient_id, language, provider_mode="audio_only")
        response = await self._request_json(
            method="POST",
            url=f"{self.classifier_endpoint.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.classifier_endpoint.api_key}",
                "Content-Type": "application/json",
            },
            json_body={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompt.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{prompt.user_prompt}\n\n"
                                    "This verification run is transcript-only and follows the clinic "
                                    "audio_only protocol.\n"
                                    "Return empty visual_findings and body_findings unless the session "
                                    "is unusable.\n"
                                    "Transcript:\n"
                                    f"{transcript}"
                                ),
                            }
                        ],
                    },
                ],
                "temperature": 0.1,
                "stream": False,
            },
            timeout=self.verification_settings.request_timeout_seconds,
        )
        text = self._extract_chat_text(response)
        payload = self._parse_text_response(text)
        return payload.model_dump(mode="json"), response.get("id")

    def _encode_audio_data_url(self, path: Path, mime_type: str) -> str:
        raw = path.read_bytes()
        if len(raw) > MAX_DATA_URL_AUDIO_BYTES:
            raise ProviderError(
                "unsupported_media",
                f"audio file exceeds {MAX_DATA_URL_AUDIO_BYTES} bytes before base64 encoding",
            )
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
