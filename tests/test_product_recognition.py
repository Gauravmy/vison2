import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.product_recognition import PRODUCT_CATEGORY_MIN_CONFIDENCE, ProductRecognizer


class ProductRecognitionTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.recognizer = ProductRecognizer(model_path=Path(temporary_directory.name) / "absent.pt")

    def available_recognizer(self):
        self.recognizer.model = object()
        self.recognizer.validated = True
        return self.recognizer

    def test_exact_custom_category_class_can_identify_product(self):
        result = self.available_recognizer().identify([
            {"label": "plastic_bottle", "confidence": PRODUCT_CATEGORY_MIN_CONFIDENCE + 0.1}
        ])
        self.assertEqual(result.category, "Plastic bottle")
        self.assertTrue(result.confident)
        self.assertEqual(result.model_label, "plastic_bottle")

    def test_generic_bottle_suggests_but_does_not_confirm_material(self):
        result = self.available_recognizer().identify([{"label": "bottle", "confidence": 0.95}])
        self.assertEqual(result.category, "Plastic bottle")
        self.assertFalse(result.confident)
        self.assertIn("does not confirm", result.explanation)

    def test_low_confidence_exact_label_requires_manual_confirmation(self):
        result = self.available_recognizer().identify([{"label": "marker", "confidence": 0.5}])
        self.assertEqual(result.category, "Marker")
        self.assertFalse(result.confident)
        self.assertIn("below", result.explanation)

    def test_competing_categories_require_manual_confirmation(self):
        result = self.available_recognizer().identify([
            {"label": "marker", "confidence": 0.82},
            {"label": "phone_charger", "confidence": 0.78},
        ])
        self.assertFalse(result.confident)
        self.assertIn("competing", result.explanation)

    def test_unrelated_labels_do_not_become_supported_products(self):
        result = self.available_recognizer().identify([{"label": "cell phone", "confidence": 0.99}])
        self.assertIsNone(result.category)
        self.assertFalse(result.confident)

    def test_missing_weights_report_unavailable_not_unknown_inference(self):
        result = self.recognizer.identify([])
        self.assertFalse(self.recognizer.available)
        self.assertEqual(result.source, "unavailable")
        self.assertIsNone(result.category)

    def test_exact_class_requires_validation_attestation(self):
        self.recognizer.model = object()
        result = self.recognizer.identify([{"label": "marker", "confidence": 0.99}])
        self.assertEqual(result.category, "Marker")
        self.assertFalse(result.confident)
        self.assertIn("not been marked validated", result.explanation)

    def test_mocked_ultralytics_classifier_uses_actual_top_class_scores(self):
        class FakeScore:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeProbabilities:
            top5 = [0, 1]
            top5conf = [FakeScore(0.93), FakeScore(0.07)]

        class FakeResult:
            probs = FakeProbabilities()
            boxes = None

            def plot(self):
                return np.zeros((32, 32, 3), dtype=np.uint8)

        class FakeModel:
            def predict(self, **_kwargs):
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as directory:
            recognizer = ProductRecognizer(model_path=Path(directory) / "not-loaded.pt")
            recognizer.model = FakeModel()
            recognizer.error = None
            recognizer.validated = True
            recognizer.classes = {0: "marker", 1: "bottle"}
            detections, _annotated, elapsed = recognizer.detect(np.zeros((32, 32, 3), dtype=np.uint8))
            identification = recognizer.identify(detections)

        self.assertTrue(elapsed is not None)
        self.assertEqual(identification.category, "Marker")
        self.assertTrue(identification.confident)
        self.assertEqual(identification.confidence, 0.93)
        self.assertEqual(detections[0]["kind"], "product classification candidate")


if __name__ == "__main__":
    unittest.main()