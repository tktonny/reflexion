"""Standalone Flask server for Raspberry Pi WAV uploads and audio-only analysis."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from backend.src.app.core.errors import ProviderError
from backend.src.app.models import PreparedMedia, ProviderContext, utc_now
from clinic.configs.settings import get_settings
from clinic.intelligence.providers.openai import AudioOnlyProvider


logger = logging.getLogger("audio_server")

DEFAULT_HOST = os.getenv("AUDIO_SERVER_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("AUDIO_SERVER_PORT", "5001"))
DEFAULT_PATIENT_ID = os.getenv("AUDIO_SERVER_DEFAULT_PATIENT_ID", "raspi-patient")
DEFAULT_LANGUAGE = os.getenv("AUDIO_SERVER_DEFAULT_LANGUAGE", "en")
OPENAI_TRANSCRIPTION_MAX_BYTES = 25 * 1024 * 1024
WAV_EXTENSIONS = {".wav"}
WAV_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = min(
    int(os.getenv("AUDIO_SERVER_MAX_UPLOAD_MB", "25")) * 1024 * 1024,
    OPENAI_TRANSCRIPTION_MAX_BYTES,
)


def _json_error(
    message: str,
    status_code: int,
    *,
    error: str,
    **extra: Any,
):
    payload: dict[str, Any] = {
        "ok": False,
        "error": error,
        "message": message,
    }
    payload.update(extra)
    return jsonify(payload), status_code


def _provider_error_status(code: str) -> int:
    if code == "unsupported_media":
        return 422
    if code == "invalid_provider_output":
        return 502
    if code == "timeout":
        return 504
    if code == "provider_unavailable":
        return 503
    return 500


def _validate_upload(filename: str, content_type: str | None) -> str:
    safe_name = secure_filename(filename) or "audio.wav"
    suffix = Path(safe_name).suffix.lower()
    if suffix in WAV_EXTENSIONS:
        return safe_name
    if content_type and content_type.lower() in WAV_MIME_TYPES:
        return f"{Path(safe_name).stem or 'audio'}.wav"
    raise ValueError("file must be a WAV upload sent as form-data field 'file'")


async def _analyze_saved_wav(audio_path: Path, patient_id: str, language: str) -> dict[str, Any]:
    settings = get_settings()
    provider = AudioOnlyProvider(settings)
    context = ProviderContext(
        assessment_id=uuid.uuid4().hex,
        patient_id=patient_id,
        language=language,
        media=PreparedMedia(
            original_path=str(audio_path),
            standardized_path=str(audio_path),
            extracted_audio_path=str(audio_path),
            mime_type="audio/wav",
            size_bytes=audio_path.stat().st_size,
        ),
    )

    started = time.perf_counter()
    provider_input = await provider.prepare(context)
    raw_result = await provider.analyze(provider_input, context)
    payload = provider.normalize(raw_result)
    latency_ms = int((time.perf_counter() - started) * 1000)

    response_payload = payload.model_dump(mode="json")
    response_payload.update(
        {
            "assessment_id": context.assessment_id,
            "patient_id": patient_id,
            "language": language,
            "created_at": utc_now().isoformat(),
            "provider_meta": {
                "final_provider": provider.name,
                "model_name": provider.model_name,
                "request_id": raw_result.request_id,
                "latency_ms": latency_ms,
                "raw_status": raw_result.raw_status,
            },
            "provider_trace": [
                {
                    "provider": provider.name,
                    "attempt_order": 1,
                    "status": "success",
                    "latency_ms": latency_ms,
                }
            ],
            "fallback_message": None,
        }
    )
    return response_payload


def analyze_saved_wav(audio_path: Path, patient_id: str, language: str) -> dict[str, Any]:
    return asyncio.run(_analyze_saved_wav(audio_path, patient_id, language))


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_exc: RequestEntityTooLarge):
    max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return _json_error(
        f"uploaded file exceeds the {max_mb}MB server limit",
        413,
        error="file_too_large",
    )


@app.post("/analyze")
def analyze():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _json_error(
            "missing WAV file in multipart field 'file'",
            400,
            error="missing_file",
        )

    try:
        safe_name = _validate_upload(upload.filename, upload.mimetype)
    except ValueError as exc:
        return _json_error(str(exc), 415, error="invalid_file_type")

    patient_id = request.form.get("patient_id", DEFAULT_PATIENT_ID).strip() or DEFAULT_PATIENT_ID
    language = request.form.get("language", DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE

    with tempfile.TemporaryDirectory(prefix="audio_server_") as temp_dir:
        audio_path = Path(temp_dir) / safe_name
        upload.save(audio_path)
        size_bytes = audio_path.stat().st_size
        if size_bytes <= 0:
            return _json_error("uploaded WAV file is empty", 400, error="empty_file")
        if size_bytes > OPENAI_TRANSCRIPTION_MAX_BYTES:
            return _json_error(
                "uploaded WAV exceeds the 25MB transcription limit",
                413,
                error="file_too_large",
            )

        try:
            result = analyze_saved_wav(audio_path, patient_id, language)
        except ProviderError as exc:
            logger.warning("audio analysis failed code=%s message=%s", exc.code, exc.message)
            return _json_error(
                exc.message,
                _provider_error_status(exc.code),
                error=exc.code,
                debug_details=exc.debug_details,
            )
        except Exception:  # noqa: BLE001
            logger.exception("unexpected audio analysis failure")
            return _json_error(
                "unexpected server error while analyzing audio",
                500,
                error="internal_error",
            )

    result["received_filename"] = safe_name
    result["ok"] = True
    return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("AUDIO_SERVER_LOG_LEVEL", "INFO"))
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT)
