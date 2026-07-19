"""TalkBank auth, access checks, and lightweight dataset download helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from verification.config import VerificationSettings


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    url: str


class _DirectoryLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._href: str | None = None
        self._chunks: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        self._href = attr_map.get("href")
        self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        text = "".join(self._chunks).strip()
        self.links.append((text, self._href))
        self._href = None
        self._chunks = []


class TalkBankClient:
    def __init__(self, settings: VerificationSettings) -> None:
        self.settings = settings
        self.client = httpx.Client(
            base_url=self.settings.talkbank_auth_base_url,
            follow_redirects=True,
            timeout=120.0,
        )

    def close(self) -> None:
        self.client.close()

    def login(self, *, email: str, password: str) -> dict[str, Any]:
        response = self.client.post(
            "/logInUser",
            json={"email": email, "pswd": password},
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"TalkBank login failed: {payload}")
        return payload

    def session_has_auth(self, path: str) -> dict[str, Any]:
        response = self.client.post(
            "/sessionHasAuth",
            json={"rootName": "data", "path": path},
        )
        response.raise_for_status()
        return response.json()

    def get_anno_path_trees(self) -> dict[str, Any]:
        response = self.client.post("/getAnnoPathTrees", json={})
        response.raise_for_status()
        return response.json()

    def list_directory(self, url: str) -> list[DirectoryEntry]:
        response = self.client.get(url)
        response.raise_for_status()
        if b"Not authorized" in response.content:
            raise RuntimeError(f"not authorized to access directory: {url}")
        parser = _DirectoryLinkParser()
        parser.feed(response.text)
        entries: list[DirectoryEntry] = []
        for text, href in parser.links:
            if not text or text == "Parent Directory":
                continue
            resolved_url = urljoin(str(response.url), href)
            if text == "⇩":
                if entries:
                    last_entry = entries[-1]
                    entries[-1] = DirectoryEntry(name=last_entry.name, url=resolved_url)
                continue
            entries.append(DirectoryEntry(name=text, url=resolved_url))
        return entries

    def download_if_authorized(self, url: str, dest_path: Path) -> bool:
        response = self.client.get(url)
        if response.status_code >= 400:
            return False
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type and b"Not authorized" in response.content:
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(response.content)
        return True

    def download_to_path(self, url: str, dest_path: Path, *, overwrite: bool = False) -> Path:
        if dest_path.exists() and not overwrite:
            return dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            buffered = bytearray()
            with dest_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        if "text/html" in content_type and len(buffered) < 8192:
                            remaining = 8192 - len(buffered)
                            buffered.extend(chunk[:remaining])
                        handle.write(chunk)
            if "text/html" in content_type and b"Not authorized" in buffered:
                dest_path.unlink(missing_ok=True)
                raise RuntimeError(f"not authorized to download: {url}")
        return dest_path


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
