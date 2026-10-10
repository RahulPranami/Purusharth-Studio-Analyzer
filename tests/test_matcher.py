"""Tests for face extraction, vector indexing, and selfie matching."""

import io
from PIL import Image
import pytest

from app.face_engine import MockFaceEngine
from app.matcher import FaceMatcher


def test_mock_face_engine_detects_face(sample_face_image_bytes):
    """Verify that MockFaceEngine detects face and generates unit-normalized vector."""
    img = Image.open(io.BytesIO(sample_face_image_bytes))
    engine = MockFaceEngine(dimension=128)
    results = engine.detect_and_embed(img)

    assert len(results) >= 1
    face = results[0]
    assert face.face_index == 0
    assert len(face.embedding) == 128
    # Test L2 unit normalization
    import numpy as np
    norm = np.linalg.norm(face.embedding)
    assert pytest.approx(norm, rel=1e-4) == 1.0


def test_mock_face_engine_flat_image_returns_no_faces(sample_flat_image_bytes):
    """Verify that featureless flat images yield 0 detected faces."""
    img = Image.open(io.BytesIO(sample_flat_image_bytes))
    engine = MockFaceEngine(dimension=128)
    results = engine.detect_and_embed(img)
    assert len(results) == 0


def test_index_and_search_face(temp_db, sample_face_image_bytes):
    """Verify storing face embeddings and vector cosine search retrieval."""
    img = Image.open(io.BytesIO(sample_face_image_bytes))
    engine = MockFaceEngine(dimension=128)
    faces = engine.detect_and_embed(img)

    temp_db.record_indexed_image(
        image_key="photos/person_a_01.jpg",
        collection_scope="test-scope",
        etag="test-etag-1",
        size_bytes=len(sample_face_image_bytes),
        status="indexed",
        faces=faces,
        model_version=engine.model_version,
    )

    stats = temp_db.get_stats()
    assert stats["indexed_images"] == 1
    assert stats["total_faces"] >= 1

    # Search with identical embedding
    query_vec = faces[0].embedding
    matches = temp_db.search_similar_faces(
        query_embedding=query_vec,
        collection_scope="test-scope",
        threshold=0.90,
    )

    assert len(matches) >= 1
    assert matches[0]["image_key"] == "photos/person_a_01.jpg"
    assert matches[0]["similarity"] >= 0.99
    assert matches[0]["distance"] <= 0.01


def test_incremental_indexing_skips_unchanged(temp_db, sample_face_image_bytes):
    """Verify that unchanged ETags are skipped by should_index."""
    img = Image.open(io.BytesIO(sample_face_image_bytes))
    engine = MockFaceEngine(dimension=128)
    faces = engine.detect_and_embed(img)

    temp_db.record_indexed_image(
        image_key="photos/photo_01.jpg",
        collection_scope="test-scope",
        etag="etag-v1",
        size_bytes=1000,
        status="indexed",
        faces=faces,
        model_version=engine.model_version,
    )

    # Same etag without force
    assert temp_db.should_index("photos/photo_01.jpg", etag="etag-v1", force_reindex=False) is False
    # Same etag with force
    assert temp_db.should_index("photos/photo_01.jpg", etag="etag-v1", force_reindex=True) is True
    # Different etag
    assert temp_db.should_index("photos/photo_01.jpg", etag="etag-v2", force_reindex=False) is True


def test_match_upload_endpoint(client, sample_face_image_bytes, temp_db):
    """Verify uploading a selfie through POST /match/upload endpoint."""
    # First seed the index with a photo
    img = Image.open(io.BytesIO(sample_face_image_bytes))
    engine = MockFaceEngine(dimension=128)
    faces = engine.detect_and_embed(img)
    temp_db.record_indexed_image(
        image_key="photos/target_person.jpg",
        collection_scope="default",
        etag="etag-sample",
        size_bytes=len(sample_face_image_bytes),
        status="indexed",
        faces=faces,
        model_version=engine.model_version,
    )

    response = client.post(
        "/match/upload",
        files={"file": ("selfie.jpg", sample_face_image_bytes, "image/jpeg")},
        data={"collection_scope": "default", "threshold": 0.5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_faces_in_query"] >= 1
    assert data["matched_photos_count"] >= 1
    assert data["top_matches"][0]["image_key"] == "photos/target_person.jpg"
    assert data["top_matches"][0]["best_similarity"] >= 0.95


def test_match_upload_no_face_error(client, sample_flat_image_bytes):
    """Verify that uploading an image without detectable faces returns 422."""
    response = client.post(
        "/match/upload",
        files={"file": ("blank.jpg", sample_flat_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 422
    assert "No face could be detected" in response.json()["detail"]


def test_selfie_analyze_endpoint(client, sample_face_image_bytes):
    """Verify that /selfie/analyze returns face metadata and suitability metrics."""
    response = client.post(
        "/selfie/analyze",
        files={"file": ("selfie.jpg", sample_face_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["faces_detected_count"] >= 1
    assert data["is_valid_selfie"] is True
    assert len(data["faces"]) >= 1
    face = data["faces"][0]
    assert "bounding_box" in face
    assert face["bounding_box"]["width"] > 0
    assert face["confidence"] > 0.5

