import unittest
from pathlib import Path

import numpy as np

from src.defect_inspection import DefectInspector, product_for_defect_class


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = FakeScalar(class_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [np.asarray(xyxy, dtype=float)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def predict(self, **_kwargs):
        return [FakeResult([
            FakeBox(0, 0.91, [5, 6, 40, 50]),
            FakeBox(1, 0.84, [60, 70, 100, 120]),
        ])]


class DefectInspectionTests(unittest.TestCase):
    def test_dataset_taxonomy_classes_have_product_scope(self):
        self.assertEqual(product_for_defect_class("marker_barrel_crack"), "Marker")
        self.assertEqual(product_for_defect_class("bottle_body_crack"), "Plastic bottle")
        self.assertEqual(product_for_defect_class("charger_pin_damage"), "Phone charger")
        self.assertIsNone(product_for_defect_class("generic_surface_crack"))

    def test_missing_weights_disable_defect_inference(self):
        inspector = DefectInspector(model_path=Path("missing-defect-weights.pt"), validated=True)
        detections, annotated, elapsed = inspector.inspect(np.zeros((128, 128, 3), dtype=np.uint8), "Marker")
        self.assertFalse(inspector.available)
        self.assertEqual(detections, [])
        self.assertIsNone(annotated)
        self.assertIsNone(elapsed)

    def test_mocked_model_output_is_filtered_by_selected_product(self):
        inspector = DefectInspector(
            model_path=Path("mocked-test-weights.pt"),
            defect_classes=("marker_barrel_crack", "bottle_body_crack"),
            validated=True,
        )
        inspector.model = FakeModel()
        inspector.error = None
        inspector.classes = {0: "marker_barrel_crack", 1: "bottle_body_crack"}
        inspector.class_products = {"marker_barrel_crack": "Marker", "bottle_body_crack": "Plastic bottle"}
        image = np.zeros((140, 140, 3), dtype=np.uint8)

        marker_detections, marker_image, _ = inspector.inspect(image, "Marker")
        bottle_detections, _, _ = inspector.inspect(image, "Plastic bottle")

        self.assertEqual([item["label"] for item in marker_detections], ["marker_barrel_crack"])
        self.assertEqual([item["label"] for item in bottle_detections], ["bottle_body_crack"])
        self.assertEqual(marker_image.shape, image.shape)


if __name__ == "__main__":
    unittest.main()