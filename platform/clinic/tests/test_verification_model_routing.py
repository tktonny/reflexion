from verification.config import VerificationSettings
from verification.model_routing import resolve_model_endpoint


def make_settings() -> VerificationSettings:
    return VerificationSettings(
        project_root=None,  # type: ignore[arg-type]
        data_dir=None,  # type: ignore[arg-type]
        results_dir=None,  # type: ignore[arg-type]
        prepared_dir=None,  # type: ignore[arg-type]
        qwen_api_key="qwen-key",
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        coding_plan_api_key="coding-key",
        coding_plan_base_url="https://coding.dashscope.aliyuncs.com/v1",
        qwen_text_model="qwen3.5-plus",
        qwen_asr_model="qwen3-asr-flash",
        request_timeout_seconds=60.0,
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        talkbank_auth_base_url="https://sla2.talkbank.org",
        talkbank_media_base_url="https://media.talkbank.org",
        talkbank_email=None,
        talkbank_password=None,
    )


def test_coding_plan_models_use_coding_plan_credentials() -> None:
    endpoint = resolve_model_endpoint(make_settings(), "qwen3.5-plus")
    assert endpoint.api_key == "coding-key"
    assert endpoint.base_url == "https://coding.dashscope.aliyuncs.com/v1"
    assert endpoint.credential_source == "coding_plan"


def test_asr_model_uses_standard_qwen_credentials() -> None:
    endpoint = resolve_model_endpoint(make_settings(), "qwen3-asr-flash")
    assert endpoint.api_key == "qwen-key"
    assert endpoint.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert endpoint.credential_source == "qwen"


def test_non_qwen_coding_plan_models_use_coding_plan_credentials() -> None:
    endpoint = resolve_model_endpoint(make_settings(), "glm-5")
    assert endpoint.api_key == "coding-key"
    assert endpoint.base_url == "https://coding.dashscope.aliyuncs.com/v1"
    assert endpoint.credential_source == "coding_plan"
