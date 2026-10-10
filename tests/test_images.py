"""Tests for image decoding, validation, dimension bounds, and security checks."""

import io
import pytest
from PIL import Image

from app.image_utils import (
    ImageValidationError,
    validate_and_decode_image,
    validate_object_key,
)


def test_valid_image_decoding(sample_face_image_bytes):
    """Verify that valid JPEG bytes are decoded into an RGB PIL Image."""
    img, fmt = validate_and_decode_image(sample_face_image_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert fmt == "JPEG"
    assert img.size == (256, 256)


def test_png_format_decoding():
    """Verify that PNG images are supported and converted to RGB."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    decoded, fmt = validate_and_decode_image(buffer.getvalue())
    assert decoded.mode == "RGB"
    assert fmt == "PNG"


def test_corrupt_image_rejection():
    """Verify that corrupt byte streams trigger ImageValidationError."""
    with pytest.raises(ImageValidationError):
        validate_and_decode_image(b"not-an-image-payload-just-garbage")


def test_empty_image_rejection():
    """Verify that empty payloads are rejected."""
    with pytest.raises(ImageValidationError):
        validate_and_decode_image(b"")


def test_image_size_limit_rejection(sample_face_image_bytes):
    """Verify that images exceeding max_bytes are rejected."""
    with pytest.raises(ImageValidationError, match="exceeds limit"):
        validate_and_decode_image(sample_face_image_bytes, max_bytes=100)


def test_image_min_dimension_rejection():
    """Verify that images smaller than min_dimension are rejected."""
    small = Image.new("RGB", (16, 16), color=(200, 200, 200))
    buf = io.BytesIO()
    small.save(buf, format="JPEG")
    with pytest.raises(ImageValidationError, match="smaller than minimum"):
        validate_and_decode_image(buf.getvalue(), min_dimension=32)


def test_object_key_validation_valid():
    """Verify that well-formed keys pass validation."""
    assert validate_object_key("photos/events/wedding_001.jpg") == "photos/events/wedding_001.jpg"
    assert validate_object_key("abc-123.png") == "abc-123.png"


@pytest.mark.parametrize(
    "bad_key",
    [
        "../secret.txt",
        "photos/../../etc/passwd",
        "/leading/slash.jpg",
        "\\backslash\\path.jpg",
        "null\x00byte.jpg",
        "",
        "   ",
    ],
)
def test_object_key_validation_rejects_insecure_keys(bad_key):
    """Verify that traversal and illegal keys are rejected with ValueError."""
    with pytest.raises(ValueError):
        validate_object_key(bad_key)
