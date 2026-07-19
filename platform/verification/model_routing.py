"""Model-to-credential routing for verification providers."""

from __future__ import annotations

from dataclasses import dataclass

from verification.config import VerificationSettings


CODING_PLAN_MODELS = {
    "qwen3.5-plus",
    "qwen3-max-2026-01-23",
    "qwen3-coder-next",
    "qwen3-coder-plus",
    "glm-5",
    "glm-4.7",
    "kimi-k2.5",
    "MiniMax-M2.5",
}


@dataclass(frozen=True)
class ModelEndpoint:
    model_name: str
    api_key: str | None
    base_url: str
    credential_source: str


def resolve_model_endpoint(
    settings: VerificationSettings,
    model_name: str,
) -> ModelEndpoint:
    if model_name in CODING_PLAN_MODELS:
        return ModelEndpoint(
            model_name=model_name,
            api_key=settings.coding_plan_api_key,
            base_url=settings.coding_plan_base_url,
            credential_source="coding_plan",
        )
    return ModelEndpoint(
        model_name=model_name,
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        credential_source="qwen",
    )
