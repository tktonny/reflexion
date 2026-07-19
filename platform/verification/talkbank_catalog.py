"""Build a reusable TalkBank corpus catalog from the authorized annotation tree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AUDIO_MEDIA_TYPES = {"audio"}
VIDEO_MEDIA_TYPES = {"video"}
LABEL_MAP = {
    "control": "HC",
    "hc": "HC",
    "healthy": "HC",
    "healthycontrol": "HC",
    "normal": "HC",
    "dementia": "cognitive_risk",
    "ad": "cognitive_risk",
    "alzheimers": "cognitive_risk",
    "alzheimer": "cognitive_risk",
    "mci": "cognitive_risk",
}


def build_talkbank_catalog(tree_payload: dict[str, Any]) -> dict[str, Any]:
    root = tree_payload.get("respMsg", {}).get("dementia", {}).get("dementia", {})
    languages: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    total_corpora = 0
    total_leaf_files = 0
    for language_name in sorted(root):
        language_node = root[language_name]
        corpus_entries: list[dict[str, Any]] = []
        for corpus_name in sorted(language_node):
            corpus_node = language_node[corpus_name]
            corpus_entry = _build_corpus_entry(language_name, corpus_name, corpus_node)
            corpus_entries.append(corpus_entry)
            total_corpora += 1
            total_leaf_files += corpus_entry["leaf_file_count"]
            if corpus_entry["classification_ready"]:
                candidates.append(
                    {
                        "language": language_name,
                        "corpus": corpus_name,
                        "task_type": "binary_audio_classification",
                        "label_groups": corpus_entry["label_groups"],
                        "suggested_root_paths": corpus_entry["suggested_root_paths"],
                        "audio_file_count": corpus_entry["audio_file_count"],
                    }
                )

        languages[language_name] = {
            "corpus_count": len(corpus_entries),
            "corpora": corpus_entries,
        }

    return {
        "summary": {
            "language_count": len(languages),
            "corpus_count": total_corpora,
            "leaf_file_count": total_leaf_files,
            "classification_ready_corpus_count": len(candidates),
        },
        "languages": languages,
        "verification_candidates": candidates,
    }


def write_talkbank_catalog(
    tree_payload: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    catalog = build_talkbank_catalog(tree_payload)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return catalog


def _build_corpus_entry(language_name: str, corpus_name: str, corpus_node: Any) -> dict[str, Any]:
    leaf_entries = list(_walk_leaves(corpus_node))
    audio_file_count = sum(1 for item in leaf_entries if item["media_base"] in AUDIO_MEDIA_TYPES)
    video_file_count = sum(1 for item in leaf_entries if item["media_base"] in VIDEO_MEDIA_TYPES)
    missing_file_count = sum(1 for item in leaf_entries if item["missing"])
    media_types = sorted({item["media_base"] for item in leaf_entries if item["media_base"]})

    label_groups = _detect_label_groups(language_name, corpus_name, corpus_node)
    classification_ready = len(label_groups) >= 2 and video_file_count == 0 and audio_file_count > 0

    return {
        "language": language_name,
        "corpus": corpus_name,
        "leaf_file_count": len(leaf_entries),
        "audio_file_count": audio_file_count,
        "video_file_count": video_file_count,
        "missing_file_count": missing_file_count,
        "media_types": media_types,
        "label_groups": label_groups,
        "classification_ready": classification_ready,
        "suggested_root_paths": [
            f"dementia/{language_name}/{corpus_name}/{group['group_name']}/"
            for group in label_groups
        ] or [f"dementia/{language_name}/{corpus_name}/"],
    }


def _walk_leaves(node: Any, prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if _is_leaf(node):
        media_value = str(node.get("media", "")).strip().lower()
        media_parts = [part.strip() for part in media_value.split(",") if part.strip()]
        media_base = media_parts[0] if media_parts else ""
        return [
            {
                "path": prefix,
                "media": media_value,
                "media_base": media_base,
                "missing": "missing" in media_parts,
            }
        ]

    entries: list[dict[str, Any]] = []
    if isinstance(node, dict):
        for key in sorted(node):
            entries.extend(_walk_leaves(node[key], prefix + (key,)))
    return entries


def _detect_label_groups(language_name: str, corpus_name: str, corpus_node: Any) -> list[dict[str, Any]]:
    if not isinstance(corpus_node, dict):
        return []

    groups: list[dict[str, Any]] = []
    for group_name in sorted(corpus_node):
        mapped_label = _map_label(group_name)
        if mapped_label is None:
            continue
        leaf_entries = _walk_leaves(corpus_node[group_name], prefix=(group_name,))
        if not leaf_entries:
            continue
        audio_file_count = sum(1 for item in leaf_entries if item["media_base"] in AUDIO_MEDIA_TYPES)
        video_file_count = sum(1 for item in leaf_entries if item["media_base"] in VIDEO_MEDIA_TYPES)
        groups.append(
            {
                "group_name": group_name,
                "mapped_label": mapped_label,
                "leaf_file_count": len(leaf_entries),
                "audio_file_count": audio_file_count,
                "video_file_count": video_file_count,
                "relative_path": f"dementia/{language_name}/{corpus_name}/{group_name}/",
            }
        )
    return groups


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and node.get("file") is True


def _map_label(raw_name: str) -> str | None:
    key = "".join(ch for ch in raw_name.strip().lower() if ch.isalnum())
    return LABEL_MAP.get(key)
