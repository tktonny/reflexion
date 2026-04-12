import httpx

from verification.download_talkbank import load_public_task1_rows


def test_load_public_task1_rows_uses_cached_copy_when_refresh_fails(
    monkeypatch, tmp_path
) -> None:
    class FailingClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str):
            raise httpx.ConnectError("tls failure")

    monkeypatch.setattr("verification.download_talkbank.httpx.Client", FailingClient)

    task1_path = tmp_path / "task1.csv"
    task1_path.write_text("ID,Label\ncase-1,HC\ncase-2,AD\n", encoding="utf-8")

    rows, public_assets = load_public_task1_rows(task1_path)

    assert [row["ID"] for row in rows] == ["case-1", "case-2"]
    assert public_assets["task1_source"] == "cached"
    assert public_assets["task1_case_count"] == 2
    assert "Using cached task1.csv" in public_assets["task1_warning"]
