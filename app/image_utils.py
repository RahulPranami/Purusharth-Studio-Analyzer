"""Image validation, decoding, and preprocessing utilities."""

import io
import re
from typing import Tuple
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings

# Enforce reasonable pixel bounds to prevent decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 50_000_000

# Allowed MIME / format extensions
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


class ImageValidationError(Exception):
    """Raised when an image fails format, size, or dimension validation."""
    pass


def validate_object_key(key: str) -> str:
    """
    Validate that an S3/R2 object key is safe and well-formed.
    Rejects directory traversal, null bytes, and path anomalies.
    """
    if not key or not isinstance(key, str):
        raise ValueError("Object key must be a non-empty string.")

    cleaned_key = key.strip()
    if not cleaned_key:
        raise ValueError("Object key cannot be empty or whitespace.")

    if len(cleaned_key) > 1024:
        raise ValueError("Object key exceeds maximum permitted length (1024 characters).")

    if "\x00" in cleaned_key:
        raise ValueError("Object key contains illegal null character.")

    if cleaned_key.startswith("/") or cleaned_key.startswith("\\"):
        raise ValueError("Object key cannot start with leading slash or backslash.")

    # Guard against directory traversal
    parts = re.split(r"[/\\]", cleaned_key)
    if ".." in parts:
        raise ValueError("Object key contains forbidden directory traversal sequence '..'.")

    return cleaned_key


def validate_and_decode_image(
    image_bytes: bytes,
    max_bytes: int = settings.max_image_size_bytes,
    min_dimension: int = settings.min_image_dimension,
    max_dimension: int = settings.max_image_dimension,
) -> Tuple[Image.Image, str]:
    """
    Safely validate and decode raw bytes into a normalized RGB PIL Image.

    Returns:
        Tuple of (PIL.Image.Image in RGB mode, detected format)

    Raises:
        ImageValidationError: If format, size, or dimensions violate constraints.
    """
    if not image_bytes:
        raise ImageValidationError("Image data is empty.")

    if len(image_bytes) > max_bytes:
        raise ImageValidationError(
            f"Image payload size ({len(image_bytes)} bytes) exceeds limit ({max_bytes} bytes)."
        )

    try:
        buffer = io.BytesIO(image_bytes)
        img = Image.open(buffer)
        image_format = (img.format or "").upper()

        if image_format not in ALLOWED_FORMATS:
            raise ImageValidationError(
                f"Unsupported image format '{image_format}'. Allowed formats: {sorted(list(ALLOWED_FORMATS))}"
            )

        # Handle EXIF rotation if present
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass  # Fall back to raw orientation if EXIF transpose fails

        # Force decode image into memory to detect corrupt data
        img.load()

        width, height = img.size
        if width < min_dimension or height < min_dimension:
            raise ImageValidationError(
                f"Image dimension ({width}x{height}) is smaller than minimum required ({min_dimension}x{min_dimension})."
            )

        if width > max_dimension or height > max_dimension:
            raise ImageValidationError(
                f"Image dimension ({width}x{height}) exceeds maximum allowed ({max_dimension}x{max_dimension})."
            )

        # Convert to standard RGB color space
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img, image_format

    except UnidentifiedImageError as err:
        raise ImageValidationError("File is not a valid or recognizable image.") from err
    except (ImageValidationError, ValueError):
        raise
    except Exception as err:
        raise ImageValidationError(f"Failed to process image: {str(err)}") from err
