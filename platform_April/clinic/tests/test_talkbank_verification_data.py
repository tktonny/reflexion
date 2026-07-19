from pathlib import Path

from verification.models import VerificationRecord
from verification.talkbank_verification_data import (
    build_case_id,
    build_dataset_name,
    dedupe_records,
    load_selected_targets,
    parse_corpus_spec,
    select_preferred_remote_entries,
    slugify,
)


def test_parse_corpus_spec_and_slugify() -> None:
    assert parse_corpus_spec("English/Pitt") == ("English", "Pitt")
    assert slugify("Pitt-orig") == "pitt_orig"
    assert build_dataset_name("Mandarin", "Chou") == "talkbank_mandarin_chou"


def test_build_case_id_uses_full_relative_path() -> None:
    case_id = build_case_id("English", "Pitt", Path("Control/cookie/002-0.mp3"))
    assert case_id == "english__pitt__control__cookie__002_0"


def test_load_selected_targets_looks_up_catalog_candidates() -> None:
    catalog = {
        "verification_candidates": [
            {
                "language": "English",
                "corpus": "Lu",
                "label_groups": [
                    {"group_name": "Control", "mapped_label": "HC", "relative_path": "dementia/English/Lu/Control/", "audio_file_count": 1},
                    {"group_name": "Dementia", "mapped_label": "cognitive_risk", "relative_path": "dementia/English/Lu/Dementia/", "audio_file_count": 2},
                ],
                "audio_file_count": 3,
            }
        ]
    }

    targets = load_selected_targets(catalog, ["English/Lu"])

    assert len(targets) == 1
    assert targets[0].spec == "English/Lu"
    assert targets[0].audio_file_count == 3


def test_dedupe_records_prefers_mp3_over_wav_when_logical_source_matches() -> None:
    records = [
        VerificationRecord(
            case_id="english__pitt__control__cookie__002_0_mp3",
            dataset="talkbank_english_pitt",
            split="full",
            label="HC",
            media_path="/tmp/cookie/002-0.mp3",
            media_type="audio",
            language="en",
            metadata={"source_relative_path": "Control/cookie/002-0.mp3"},
        ),
        VerificationRecord(
            case_id="english__pitt__control__cookie__0wav__002_0_wav",
            dataset="talkbank_english_pitt",
            split="full",
            label="HC",
            media_path="/tmp/cookie/0wav/002-0.wav",
            media_type="audio",
            language="en",
            metadata={"source_relative_path": "Control/cookie/0wav/002-0.wav"},
        ),
    ]

    deduped = dedupe_records(records)

    assert len(deduped) == 1
    assert deduped[0].media_path.endswith("002-0.mp3")


def test_select_preferred_remote_entries_prefers_mp3() -> None:
    selected = select_preferred_remote_entries(
        [
            {"relative_parts": ("cookie", "002-0.wav"), "url": "wav"},
            {"relative_parts": ("cookie", "002-0.mp3"), "url": "mp3"},
        ],
        dataset_name="talkbank_english_pitt",
        group_name="Control",
    )

    assert len(selected) == 1
    assert selected[0]["url"] == "mp3"
