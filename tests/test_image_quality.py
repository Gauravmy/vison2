import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.image_quality import assess_image_quality, load_image, preprocess_image


class ImageQualityTests(unittest.TestCase):
    def test_rgb_array_load_and_preprocess(self):
        image = np.full((200, 300, 3), 128, dtype=np.uint8)
        loaded = load_image(image)
        resized = preprocess_image(loaded, max_dimension=100)
        self.assertEqual(loaded.shape, (200, 300, 3))
        self.assertEqual(max(resized.shape[:2]), 100)

    def test_image_file_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            Image.fromarray(np.full((200, 240, 3), 128, dtype=np.uint8)).save(path)
            loaded = load_image(path)
        self.assertEqual(loaded.shape, (200, 240, 3))

    def test_jpeg_png_and_webp_files_decode(self):
        image = Image.new("RGB", (32, 24), color=(90, 130, 160))
        with tempfile.TemporaryDirectory() as directory:
            for extension, format_name in (("jpg", "JPEG"), ("png", "PNG"), ("webp", "WEBP")):
                path = Path(directory) / f"sample.{extension}"
                image.save(path, format=format_name)
                self.assertEqual(load_image(path).shape, (24, 32, 3))

    def test_invalid_bytes_have_recoverable_error(self):
        with self.assertRaisesRegex(ValueError, "invalid or unsupported"):
            load_image(b"not an image")

    def test_file_size_and_pixel_limits_are_enforced(self):
        image = Image.new("RGB", (32, 24), color=(90, 130, 160))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.png"
            image.save(path, format="PNG")
            with patch("src.image_quality.MAX_IMAGE_UPLOAD_BYTES", path.stat().st_size - 1):
                with self.assertRaisesRegex(ValueError, "upload limit"):
                    load_image(path)
            with patch("src.image_quality.MAX_IMAGE_PIXELS", 100):
                with self.assertRaisesRegex(ValueError, "megapixel processing limit"):
                    load_image(path)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ValueError):
            load_image(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_blurry_image_is_flagged_for_recapture(self):
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        quality = assess_image_quality(image)
        self.assertFalse(quality.suitable)
        self.assertIn("blurry", " ".join(quality.concerns))

    def test_dark_image_is_flagged(self):
        quality = assess_image_quality(np.zeros((256, 256, 3), dtype=np.uint8))
        self.assertFalse(quality.suitable)
        self.assertIn("dark", " ".join(quality.concerns))


if __name__ == "__main__":
    unittest.main()