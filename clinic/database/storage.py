"""Local file-system persistence for uploaded media and assessment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from clinic.configs.settings import Settings
from backend.src.app.models import ClinicAssessment
from backend.src.app.models.identity import IdentityLinkRecord, IdentityProfile
from backend.src.app.models.longitudinal import LongitudinalProfile


class LocalStorage:
    """Persist raw uploads and normalized assessments under the configured data root."""
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def longitudinal_dir(self) -> Path:
        target = self.settings.storage_dir / "longitudinal"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def feature_snapshots_dir(self) -> Path:
        target = self.settings.storage_dir / "feature_snapshots"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def identity_profiles_dir(self) -> Path:
        target = self.settings.storage_dir / "identity_profiles"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def identity_links_dir(self) -> Path:
        target = self.settings.storage_dir / "identity_links"
        target.mkdir(parents=True, exist_ok=True)
        return target

    async def save_upload(self, assessment_id: str, upload: UploadFile) -> Path:
        target_dir = self.settings.uploads_dir / assessment_id
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload.filename or "video.bin").suffix or ".bin"
        target_path = target_dir / f"input{suffix}"
        content = await upload.read()
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(f"upload exceeds {self.settings.max_upload_mb}MB")
        target_path.write_bytes(content)
        return target_path

    def save_upload_sidecar(
        self,
        assessment_id: str,
        *,
        filename: str,
        payload: dict[str, Any],
    ) -> Path:
        target_dir = self.settings.uploads_dir / assessment_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return target_path

    def save_assessment(self, assessment: ClinicAssessment) -> Path:
        target = self.settings.assessments_dir / f"{assessment.assessment_id}.json"
        target.write_text(
            assessment.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return target

    def load_assessment(self, assessment_id: str) -> dict[str, Any]:
        target = self.settings.assessments_dir / f"{assessment_id}.json"
        if not target.exists():
            raise FileNotFoundError(assessment_id)
        return json.loads(target.read_text(encoding="utf-8"))

    def list_assessments(self) -> list[dict[str, Any]]:
        assessments: list[dict[str, Any]] = []
        for path in sorted(self.settings.assessments_dir.glob("*.json")):
            assessments.append(json.loads(path.read_text(encoding="utf-8")))
        return assessments

    def save_longitudinal_profile(self, profile: LongitudinalProfile) -> Path:
        target = self.longitudinal_dir / f"{profile.patient_id}.json"
        target.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return target

    def load_longitudinal_profile(self, patient_id: str) -> dict[str, Any]:
        target = self.longitudinal_dir / f"{patient_id}.json"
        if not target.exists():
            raise FileNotFoundError(patient_id)
        return json.loads(target.read_text(encoding="utf-8"))

    def list_longitudinal_profiles(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.longitudinal_dir.glob("*.json")):
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
        return profiles

    def save_feature_snapshot(self, assessment_id: str, payload: dict[str, Any]) -> Path:
        target = self.feature_snapshots_dir / f"{assessment_id}.json"
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    def load_feature_snapshot(self, assessment_id: str) -> dict[str, Any]:
        target = self.feature_snapshots_dir / f"{assessment_id}.json"
        if not target.exists():
            raise FileNotFoundError(assessment_id)
        return json.loads(target.read_text(encoding="utf-8"))

    def save_identity_profile(self, profile: IdentityProfile) -> Path:
        target = self.identity_profiles_dir / f"{profile.patient_id}.json"
        target.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return target

    def load_identity_profile(self, patient_id: str) -> dict[str, Any]:
        target = self.identity_profiles_dir / f"{patient_id}.json"
        if not target.exists():
            raise FileNotFoundError(patient_id)
        return json.loads(target.read_text(encoding="utf-8"))

    def list_identity_profiles(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.identity_profiles_dir.glob("*.json")):
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
        return profiles

    def save_identity_link(self, link: IdentityLinkRecord) -> Path:
        target = self.identity_links_dir / f"{link.assessment_id}.json"
        target.write_text(
            link.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return target

    def load_identity_link(self, assessment_id: str) -> dict[str, Any]:
        target = self.identity_links_dir / f"{assessment_id}.json"
        if not target.exists():
            raise FileNotFoundError(assessment_id)
        return json.loads(target.read_text(encoding="utf-8"))

    def list_identity_links(self) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []
        for path in sorted(self.identity_links_dir.glob("*.json")):
            links.append(json.loads(path.read_text(encoding="utf-8")))
        return links
