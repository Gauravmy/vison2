"""Image decoding, preprocessing, and measurable suitability checks."""

from __future__ import annotations

from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import MAX_IMAGE_PIXELS, MAX_IMAGE_UPLOAD_BYTES


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    blur_variance: float
    mean_brightness: float
    clipped_pixel_fraction: float
    suitable: bool
    concerns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_dimensions(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions are invalid.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS // 1_000_000} megapixel processing limit.")


def _decode_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError("The selected image file is unavailable or cannot be read.") from error
    if size > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError(f"Image file exceeds the {MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    _validate_image_header(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The image file could not be decoded.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_bytes(data: bytes | bytearray) -> np.ndarray:
    if len(data) > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError(f"Image data exceeds the {MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    _validate_image_header(BytesIO(data))
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("The image data could not be decoded.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _validate_image_header(source: Any) -> None:
    try:
        with Image.open(source) as image:
            width, height = image.size
            image_format = image.format
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("Image format is invalid or unsupported. Use JPG, PNG, or WEBP.") from error
    if image_format not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError("Unsupported image format. Use JPG, JPEG, PNG, or WEBP.")
    _validate_dimensions(width, height)


def _coerce_array(source: Any) -> np.ndarray:
    if hasattr(source, "convert") and hasattr(source, "size"):
        _validate_dimensions(*source.size)
        source = np.asarray(source.convert("RGB"))
    array = np.asarray(source)
    if array.ndim == 2:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or array.size == 0:
        raise ValueError("Expected a non-empty RGB image with three channels.")
    _validate_dimensions(array.shape[1], array.shape[0])
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255
        array = np.nan_to_num(array, nan=0, posinf=255, neginf=0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def load_image(source: Any) -> np.ndarray:
    """Return a contiguous RGB uint8 image from a path, bytes, PIL, or array."""
    if isinstance(source, (str, Path)):
        return _decode_file(source)
    if isinstance(source, (bytes, bytearray)):
        return _decode_bytes(source)
    return _coerce_array(source)


def preprocess_image(image: np.ndarray, max_dimension: int = 1280) -> np.ndarray:
    """Bound inference dimensions while preserving aspect ratio and RGB ordering."""
    if max_dimension < 1:
        raise ValueError("Maximum image dimension must be positive.")
    image = load_image(image)
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def assess_image_quality(
    image: np.ndarray,
    *,
    min_dimension: int = 160,
    min_blur_variance: float = 40.0,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
    max_clipped_fraction: float = 0.35,
) -> ImageQuality:
    image = load_image(image)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    concerns: list[str] = []
    if min(width, height) < min_dimension:
        concerns.append(f"Image is below the {min_dimension}px minimum dimension.")
    if blur < min_blur_variance:
        concerns.append("Image appears blurry based on Laplacian variance.")
    if brightness < min_brightness:
        concerns.append("Image is too dark.")
    elif brightness > max_brightness:
        concerns.append("Image is too bright.")
    if clipped > max_clipped_fraction:
        concerns.append("A large fraction of pixels are near black or clipped white.")
    return ImageQuality(width, height, blur, brightness, clipped, not concerns, tuple(concerns))