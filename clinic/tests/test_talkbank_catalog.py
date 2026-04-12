from verification.talkbank_catalog import build_talkbank_catalog


def test_build_talkbank_catalog_detects_classification_ready_audio_corpora() -> None:
    payload = {
        "respMsg": {
            "dementia": {
                "dementia": {
                    "English": {
                        "Lu": {
                            "Control": {
                                "F01": {"file": True, "media": "audio"},
                            },
                            "Dementia": {
                                "F02": {"file": True, "media": "audio"},
                            },
                        }
                    },
                    "Mandarin": {
                        "Chou": {
                            "HC": {
                                "001": {
                                    "001_Daddy": {"file": True, "media": "audio"},
                                }
                            },
                            "MCI": {
                                "101": {
                                    "101_Daddy": {"file": True, "media": "audio"},
                                }
                            },
                        }
                    },
                }
            }
        }
    }

    catalog = build_talkbank_catalog(payload)

    assert catalog["summary"]["language_count"] == 2
    assert catalog["summary"]["classification_ready_corpus_count"] == 2
    ready = {
        (item["language"], item["corpus"]): item
        for item in catalog["verification_candidates"]
    }
    assert ("English", "Lu") in ready
    assert ("Mandarin", "Chou") in ready
    assert ready[("English", "Lu")]["label_groups"][0]["mapped_label"] == "HC"


def test_build_talkbank_catalog_marks_video_only_corpus_not_ready() -> None:
    payload = {
        "respMsg": {
            "dementia": {
                "dementia": {
                    "English": {
                        "GR": {
                            "Anita": {"file": True, "media": "video"},
                            "James": {"file": True, "media": "video"},
                        }
                    }
                }
            }
        }
    }

    catalog = build_talkbank_catalog(payload)
    corpus = catalog["languages"]["English"]["corpora"][0]

    assert corpus["video_file_count"] == 2
    assert corpus["classification_ready"] is False
    assert catalog["verification_candidates"] == []
