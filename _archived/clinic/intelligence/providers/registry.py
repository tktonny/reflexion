"""Provider registry factory for the clinic multimodal routing layer."""

from __future__ import annotations

from backend.src.app.models import ProviderName
from clinic.configs.settings import Settings
from clinic.intelligence.providers.base import BaseProvider
from clinic.intelligence.providers.gemini import GeminiProvider
from clinic.intelligence.providers.openai import (
    AudioOnlyProvider,
    FusionProvider,
)
from clinic.intelligence.providers.qwen_omni import QwenOmniProvider


def build_provider_registry(settings: Settings) -> dict[ProviderName, BaseProvider]:
    return {
        "qwen_omni": QwenOmniProvider(settings),
        "gemini": GeminiProvider(settings),
        "fusion": FusionProvider(settings),
        "audio_only": AudioOnlyProvider(settings),
    }
