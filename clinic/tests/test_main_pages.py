from fastapi.testclient import TestClient

from backend.src.app.main import app


client = TestClient(app)


def test_clinic_page_route_returns_html() -> None:
    response = client.get("/clinic")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
