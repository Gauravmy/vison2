import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.database import InspectionDatabase
from src.reporting import export_csv, generate_inspection_report


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = InspectionDatabase(self.root / "test.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_dashboard_has_zero_counts(self):
        data = self.database.dashboard_data()
        self.assertEqual(data["total"], 0)
        self.assertEqual(sum(data["counts"].values()), 0)
        self.assertEqual(data["accepted_rate"], 0)
        self.assertEqual(data["recent"], [])

    def test_history_filters_and_get_by_id(self):
        self.database.add_inspection({
            "inspection_id": "inspection-001", "product": "Marker", "mode": "Local CV",
            "provider": "OpenCV", "outcome": "REVIEW REQUIRED", "explanation": "No validated policy.",
            "detections": [], "ai_observations": [], "quality": {"suitable": True},
        })
        found = self.database.get_inspection("inspection-001")
        self.assertEqual(found["quality"], {"suitable": True})
        self.assertEqual(len(self.database.list_inspections(product="Marker")), 1)
        self.assertEqual(self.database.list_inspections(product="Plastic bottle"), [])

    def test_dashboard_filters_restrict_all_aggregates(self):
        base_record = {
            "mode": "Local CV", "provider": "OpenCV", "explanation": "Recorded test inspection.",
            "detections": [], "ai_observations": [], "quality": {"suitable": True},
        }
        self.database.add_inspection({
            **base_record, "inspection_id": "filter-marker", "timestamp": "2026-10-12T10:00:00+00:00",
            "product": "Marker", "outcome": "PASS / ACCEPTED",
        })
        self.database.add_inspection({
            **base_record, "inspection_id": "filter-bottle", "timestamp": "2026-10-12T11:00:00+00:00",
            "product": "Plastic bottle", "outcome": "FAIL / REJECTED",
        })

        data = self.database.dashboard_data(product="Marker", outcome="PASS / ACCEPTED", date_from="2026-10-12", date_to="2026-10-12")

        self.assertEqual(data["total"], 1)
        self.assertEqual(data["counts"]["PASS / ACCEPTED"], 1)
        self.assertEqual(data["by_category"][0]["product"], "Marker")
        self.assertEqual(data["recent"][0]["inspection_id"], "filter-marker")

    def test_csv_export_contains_real_records(self):
        record = {
            "inspection_id": "inspection-002", "timestamp": "2026-10-12T10:00:00+00:00",
            "product": "Plastic bottle", "mode": "Local CV", "provider": "OpenCV",
            "outcome": "RECAPTURE REQUIRED", "explanation": "Blurry image", "detections": [],
            "ai_observations": [], "quality": {"suitable": False},
        }
        path = export_csv([record], self.root / "export.csv")
        with Path(path).open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["inspection_id"], "inspection-002")
        self.assertEqual(rows[0]["outcome"], "RECAPTURE REQUIRED")

    def test_individual_report_is_written(self):
        images = self.root / "images"
        images.mkdir()
        original_path = images / "original.png"
        annotated_path = images / "annotated.png"
        Image.new("RGB", (8, 8), color="red").save(original_path)
        Image.new("RGB", (8, 8), color="green").save(annotated_path)
        record = {
            "inspection_id": "inspection-003", "product": "Marker", "mode": "Local CV",
            "provider": "OpenCV", "outcome": "REVIEW REQUIRED", "explanation": "No model",
            "quality": {}, "detections": [], "ai_observations": [],
            "original_image_path": str(original_path), "annotated_image_path": str(annotated_path),
        }
        with patch("src.reporting.config.INSPECTION_IMAGE_DIR", images):
            path = generate_inspection_report(record, self.root / "reports")
        self.assertTrue(Path(path).is_file())
        report = Path(path).read_text(encoding="utf-8")
        self.assertIn("inspection-003", report)
        self.assertIn("data:image/png;base64,", report)


if __name__ == "__main__":
    unittest.main()