"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

APP_NAME = "VisionInspect Pro"
APP_VERSION = "1.0.0"
DATABASE_PATH = Path(os.getenv("VISIONINSPECT_DATABASE", ROOT_DIR / "artifacts" / "inspections.sqlite3"))
INSPECTION_IMAGE_DIR = Path(os.getenv("VISIONINSPECT_IMAGE_DIR", ROOT_DIR / "artifacts" / "inspections"))
PRODUCT_MODEL_PATH = Path(os.getenv("PRODUCT_MODEL_PATH", ROOT_DIR / "models" / "product_detector.pt"))
PRODUCT_MODEL_VALIDATED = os.getenv("PRODUCT_MODEL_VALIDATED", "false").strip().lower() == "true"
DEFECT_MODEL_PATH = Path(os.getenv("DEFECT_MODEL_PATH", ROOT_DIR / "models" / "defect_detector.pt"))
DEFECT_CLASS_NAMES = tuple(
    name.strip() for name in os.getenv("DEFECT_CLASS_NAMES", "").split(",") if name.strip()
)
MODEL_CONFIDENCE = min(1.0, max(0.0, float(os.getenv("MODEL_CONFIDENCE", "0.35"))))
PRODUCT_CATEGORY_CONFIDENCE = min(1.0, max(0.0, float(os.getenv("PRODUCT_CATEGORY_CONFIDENCE", "0.70"))))
DEFECT_MODEL_VALIDATED = os.getenv("DEFECT_MODEL_VALIDATED", "false").strip().lower() == "true"
AUTOMATED_PASS_ENABLED = os.getenv("AUTOMATED_PASS_ENABLED", "false").strip().lower() == "true"
AI_PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
MAX_IMAGE_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000