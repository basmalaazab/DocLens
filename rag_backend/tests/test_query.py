import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Patch both startup steps so tests don't need a real vector store,
    # YOLO weights, or Ollama server on disk — this only tests the API layer.
    with patch("app.main.init_vector_store"), patch("app.main.init_vision_model"):
        from app.main import app

        with TestClient(app) as test_client:
            yield test_client


def test_health_check(client):
    with patch("app.api.routes.query.get_chunk_count", return_value=42):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "collection_count": 42}


def test_query_happy_path(client):
    with patch("app.api.routes.query.generate_answer") as mock_generate_answer:
        mock_generate_answer.return_value = (
            "Backpropagation is an algorithm used to train neural networks.",
            [{"source": "book.pdf", "page": 42}],
        )

        response = client.post("/query", json={"question": "What is backpropagation?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["sources"] == ["book.pdf - Page 42"]


def test_query_invalid_input_returns_422(client):
    # Missing the required "question" field
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_image_happy_path(client):
    with patch("app.api.routes.query.query_with_image") as mock_query_with_image:
        mock_query_with_image.return_value = {
            "answer": "This table compares three optimizers.",
            "content_type": "table",
            "detected_elements": "Image content type: table. Detected elements: Table.",
            "sources": [{"source": "book.pdf", "page": 87}],
        }

        fake_image = io.BytesIO(b"fake image bytes")
        response = client.post(
            "/query-image",
            data={"question": "What does this table show?"},
            files={"image": ("table.jpg", fake_image, "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "table"
    assert body["sources"] == ["book.pdf - Page 87"]


def test_query_image_missing_image_returns_422(client):
    response = client.post("/query-image", data={"question": "What is this?"})
    assert response.status_code == 422
