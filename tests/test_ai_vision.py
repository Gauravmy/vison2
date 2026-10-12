import unittest
from unittest.mock import patch

import numpy as np
from unittest.mock import patch

from src.ai_vision import VisionProviderError, analyze_image, provider_status
from src.defect_inspection import DefectInspector


class AIVisionTests(unittest.TestCase):
    def test_missing_api_key_does_not_call_provider(self):
        with self.assertRaisesRegex(VisionProviderError, "API key is missing"):
            analyze_image(np.zeros((256, 256, 3), dtype=np.uint8), "Marker", provider="gemini", api_key="")

    def test_unconfigured_provider_is_reported(self):
        with self.assertRaisesRegex(VisionProviderError, "No supported vision provider"):
            analyze_image(np.zeros((256, 256, 3), dtype=np.uint8), "Marker", provider="")

    def test_missing_defect_weights_disable_defect_decisions(self):
        inspector = DefectInspector(model_path="missing-test-weights.pt", defect_classes=("crack",))
        self.assertFalse(inspector.available)
        self.assertIn("not installed", inspector.error)

    def test_status_reports_missing_gemini_key(self):
        with patch("src.ai_vision.AI_PROVIDER", "gemini"), patch("src.ai_vision.GEMINI_API_KEY", ""):
            status = provider_status()
        self.assertFalse(status["configured"])
        self.assertIn("not configured", status["message"])

    def test_mocked_rate_limit_is_reported(self):
        class FailingModels:
            def generate_content(self, **_kwargs):
                raise RuntimeError("429 quota exceeded")

        class FailingClient:
            models = FailingModels()

        with patch("google.genai.Client", return_value=FailingClient()):
            with self.assertRaisesRegex(VisionProviderError, "rate limit or quota"):
                analyze_image(
                    np.full((256, 256, 3), 128, dtype=np.uint8), "Marker",
                    provider="gemini", api_key="unit-test-placeholder", client=None,
                )


if __name__ == "__main__":
    unittest.main()