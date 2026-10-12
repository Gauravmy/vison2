import tempfile
import unittest
import re
from unittest.mock import patch
from pathlib import Path

import gradio as gr
import numpy as np

from src.database import InspectionDatabase
from src.product_recognition import ProductIdentification


class AppIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app

        cls.app_module = app

    def test_gradio_app_imports_and_builds_blocks(self):
        self.assertIsInstance(self.app_module.app, gr.Blocks)
        labels = [component.get("props", {}).get("value") for component in self.app_module.app.config["components"]]
        self.assertTrue(labels)

    def test_recapture_inspection_persists_to_temporary_database(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = InspectionDatabase(root / "test.sqlite3")
            dark_image = np.zeros((256, 256, 3), dtype=np.uint8)

            with (
                patch.object(self.app_module, "DATABASE", database),
                patch.object(self.app_module.config, "INSPECTION_IMAGE_DIR", root / "images"),
            ):
                result = self.app_module.run_inspection(dark_image, "Marker", "Local CV", False)

            self.assertEqual(result[2], "RECAPTURE REQUIRED")
            records = database.list_inspections()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["quality"]["suitable"], False)
            self.assertTrue(Path(records[0]["original_image_path"]).is_file())

    def test_upload_adapter_record_view_and_reset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = InspectionDatabase(root / "test.sqlite3")
            dark_image = np.zeros((256, 256, 3), dtype=np.uint8)
            with (
                patch.object(self.app_module, "DATABASE", database),
                patch.object(self.app_module.config, "INSPECTION_IMAGE_DIR", root / "images"),
            ):
                result = self.app_module.inspect_selected_source(
                    None, dark_image, "Upload", "Marker", self.app_module.LOCAL_MODE, False,
                )
                self.assertFalse(result[0]["visible"])
                self.assertIn("RECAPTURE REQUIRED", result[1]["value"])
                self.assertEqual(result[4], [])
                audit = result[7]["audit"]
                inspection_id = re.search(r"Inspection ID: ([0-9a-f-]+)", audit).group(1)
                opened = self.app_module.open_history_record(inspection_id)
                self.assertTrue(opened[1]["visible"])
                self.assertTrue(opened[2]["visible"])
                self.assertEqual(opened[3]["value"]["inspection ID"], inspection_id)
                reset = self.app_module.reset_inspection_ui()

            self.assertEqual(len(reset), 15)
            self.assertEqual(reset[2], "Camera")
            self.assertEqual(reset[3], "Auto-detect")
            self.assertFalse(reset[4]["visible"])
            self.assertTrue(reset[7]["visible"])

    def test_auto_mode_without_recognition_weights_records_unknown_and_review(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = InspectionDatabase(root / "auto.sqlite3")
            suitable_image = np.random.default_rng(7).integers(60, 200, (256, 256, 3), dtype=np.uint8)
            with (
                patch.object(self.app_module, "DATABASE", database),
                patch.object(self.app_module.config, "INSPECTION_IMAGE_DIR", root / "images"),
            ):
                result = self.app_module.inspect_selected_source(
                    None, suitable_image, "Upload", "Marker", self.app_module.LOCAL_MODE, False, "Auto-detect",
                )

            self.assertIn("REVIEW REQUIRED", result[1]["value"])
            self.assertIn("category is unknown", result[1]["value"])
            record = database.list_inspections()[0]
            self.assertEqual(record["product"], "Unknown")
            self.assertEqual(record["outcome"], "REVIEW REQUIRED")

    def test_manual_override_records_category_but_never_passes_without_defect_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = InspectionDatabase(root / "manual.sqlite3")
            suitable_image = np.random.default_rng(13).integers(60, 200, (256, 256, 3), dtype=np.uint8)
            with (
                patch.object(self.app_module, "DATABASE", database),
                patch.object(self.app_module.config, "INSPECTION_IMAGE_DIR", root / "images"),
            ):
                result = self.app_module.inspect_selected_source(
                    None, suitable_image, "Upload", "Phone charger", self.app_module.LOCAL_MODE,
                    False, "Manual override",
                )

            record = database.list_inspections()[0]
            self.assertIn("REVIEW REQUIRED", result[1]["value"])
            self.assertEqual(record["product"], "Phone charger")
            self.assertIn("manually set to Phone charger", record["explanation"])
            self.assertNotEqual(record["outcome"], "PASS / ACCEPTED")

    def test_mocked_exact_category_class_drives_auto_selection(self):
        class MockedRecognizer:
            available = True
            model_path = Path("mocked-product-classifier.pt")

            def detect(self, image):
                rows = [{
                    "class_id": 0, "label": "marker", "confidence": 0.91,
                    "box_xyxy": None, "kind": "product classification candidate",
                }]
                return rows, image.copy(), 1.2

            def identify(self, _detections):
                return ProductIdentification(
                    "Marker", "marker", 0.91, True, "local product model",
                    "The model identified Marker from its 'marker' class.",
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = InspectionDatabase(root / "mocked-recognition.sqlite3")
            suitable_image = np.random.default_rng(11).integers(60, 200, (256, 256, 3), dtype=np.uint8)
            with (
                patch.object(self.app_module, "DATABASE", database),
                patch.object(self.app_module.config, "INSPECTION_IMAGE_DIR", root / "images"),
                patch.object(self.app_module, "PRODUCT_RECOGNIZER", MockedRecognizer()),
            ):
                result = self.app_module.inspect_selected_source(
                    None, suitable_image, "Upload", "Plastic bottle", self.app_module.LOCAL_MODE,
                    False, "Auto-detect",
                )

            record = database.list_inspections()[0]
            self.assertEqual(record["product"], "Marker")
            self.assertEqual(record["outcome"], "REVIEW REQUIRED")
            self.assertEqual(record["model_version"], "product:mocked-product-classifier.pt")
            self.assertIn("Marker", result[7]["inspection method"])
            self.assertIn("0.91", result[7]["inspection method"])

    def test_dashboard_callback_uses_filtered_records_and_rate_rules(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = InspectionDatabase(Path(temporary_directory) / "dashboard.sqlite3")
            common = {
                "mode": self.app_module.LOCAL_MODE, "provider": "OpenCV", "explanation": "Recorded inspection.",
                "detections": [], "ai_observations": [], "quality": {"suitable": True},
            }
            database.add_inspection({
                **common, "inspection_id": "dash-marker", "timestamp": "2026-10-12T10:00:00+00:00",
                "product": "Marker", "outcome": "PASS / ACCEPTED",
            })
            database.add_inspection({
                **common, "inspection_id": "dash-bottle", "timestamp": "2026-10-12T11:00:00+00:00",
                "product": "Plastic bottle", "outcome": "REVIEW REQUIRED",
            })
            with patch.object(self.app_module, "DATABASE", database):
                filtered = self.app_module.refresh_dashboard("Marker", "All", "2026-10-12", "2026-10-12")
                outcome_filtered = self.app_module.refresh_dashboard("All", "REVIEW REQUIRED", "", "")

        self.assertIn("<strong>1</strong>", filtered[0])
        self.assertIn("100.0% of 1 inspections", filtered[0])
        self.assertEqual(len(filtered[4]), 1)
        self.assertIn("Unavailable while filtering by outcome", outcome_filtered[0])


if __name__ == "__main__":
    unittest.main()