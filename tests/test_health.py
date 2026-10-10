"""Tests for health check and system inspection endpoints."""

def test_health_endpoint(client):
    """Verify that /health responds with valid diagnostic information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "r2_configured" in data
    assert "r2_connection" in data
    assert "model_adapter" in data
    assert data["database_status"] == "ready"
    assert data["embedding_dimension"] in (128, 512)
    assert isinstance(data["indexed_images_count"], int)
    assert isinstance(data["total_faces_count"], int)


def test_docs_accessible(client):
    """Verify that OpenAPI documentation is enabled and reachable."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_root_serves_ui(client):
    """Verify that root / and /ui serve the test console frontend."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Purusharth Selfie Matcher" in res_root.text

    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "Purusharth Selfie Matcher" in res_ui.text
