"""Pytest fixtures for purusharth-selfie-api."""

import io
from pathlib import Path
import tempfile
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.config import settings
from app.index_store import SQLiteIndexStore
from app.main import app


@pytest.fixture
def temp_db(monkeypatch):
    """Provide a clean temporary SQLite database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    test_store = SQLiteIndexStore(db_path=temp_path)
    monkeypatch.setattr(settings, "face_engine_type", "mock")
    monkeypatch.setattr("app.main.index_store", test_store)
    monkeypatch.setattr("app.matcher.index_store", test_store)
    monkeypatch.setattr("app.index_store.index_store", test_store)
    from app.matcher import face_matcher
    from app.face_engine import get_face_engine
    monkeypatch.setattr(face_matcher, "store", test_store)
    monkeypatch.setattr(face_matcher, "engine", get_face_engine("mock"))

    yield test_store

    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def client(temp_db):
    """Test client with isolated SQLite database."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_face_image_bytes():
    """Generate a realistic test image with face-like gradients."""
    img = Image.new("RGB", (256, 256), color=(200, 160, 120))
    # Draw contrast features
    for x in range(80, 176):
        for y in range(80, 176):
            img.putpixel((x, y), (140, 100, 80))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_flat_image_bytes():
    """Generate a flat monochrome image where 0 faces are detected."""
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()
