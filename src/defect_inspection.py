"""Optional category-specific YOLO defect model with explicit class validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import DEFECT_CLASS_NAMES, DEFECT_MODEL_PATH, DEFECT_MODEL_VALIDATED, MODEL_CONFIDENCE

SUPPORTED_PRODUCTS = ("Marker", "Plastic bottle", "Phone charger")
PRODUCT_CLASS_PREFIXES = {
    "Marker": ("marker",),
    "Plastic bottle": ("plastic_bottle", "bottle"),
    "Phone charger": ("phone_charger", "charger"),
}


def product_for_defect_class(label: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    for product, prefixes in PRODUCT_CLASS_PREFIXES.items():
        if any(normalized == prefix or normalized.startswith(prefix + "_") for prefix in prefixes):
            return product
    return None


class DefectInspector:
    def __init__(
        self,
        model_path: Path = DEFECT_MODEL_PATH,
        defect_classes: tuple[str, ...] = DEFECT_CLASS_NAMES,
        confidence: float = MODEL_CONFIDENCE,
        validated: bool = DEFECT_MODEL_VALIDATED,
    ):
        self.model_path = Path(model_path)
        self.configured_defect_classes = {name.casefold() for name in defect_classes}
        self.confidence = confidence
        self.validated = validated
        self.model: Any = None
        self.classes: dict[int, str] = {}
        self.class_products: dict[str, str] = {}
        self.error: str | None = None
        if not self.model_path.is_file():
            self.error = "Defect-specific weights are not installed."
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.classes = {int(key): str(value) for key, value in self.model.names.items()}
            self.class_products = {
                label.casefold(): product
                for label in self.classes.values()
                if (product := product_for_defect_class(label)) is not None
            }
            if not self.configured_defect_classes:
                self.error = "DEFECT_CLASS_NAMES is empty; model classes are not approved as defined defects."
                self.model = None
            elif self.configured_defect_classes != {label.casefold() for label in self.classes.values()}:
                self.error = "Configured defect classes must exactly match all classes in the model."
                self.model = None
            elif len(self.class_products) != len(self.classes):
                unscoped = sorted(set(self.classes.values()) - {label for label in self.classes.values() if product_for_defect_class(label)})
                self.error = "Every defect class must be product-scoped. Add marker_, bottle_/plastic_bottle_, or charger_/phone_charger_ prefixes: " + ", ".join(unscoped)
                self.model = None
            elif not validated:
                self.error = "Set DEFECT_MODEL_VALIDATED=true only after evaluating this model for its intended scope."
                self.model = None
        except Exception as error:
            self.error = f"Defect model could not be loaded: {error}"
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.error is None

    def inspect(self, image: Any, selected_product: str) -> tuple[list[dict[str, Any]], Any | None, float | None]:
        if not self.available or selected_product not in SUPPORTED_PRODUCTS:
            return [], None, None
        import time

        started = time.perf_counter()
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = results[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = self.classes.get(class_id, str(class_id))
                if label.casefold() not in self.configured_defect_classes:
                    continue
                if self.class_products.get(label.casefold()) != selected_product:
                    continue
                detections.append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": float(box.conf.item()),
                    "box_xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    "product": selected_product,
                    "kind": "defined defect",
                })
        annotated = image.copy()
        if detections:
            import cv2

            for detection in detections:
                left, top, right, bottom = (int(value) for value in detection["box_xyxy"])
                cv2.rectangle(annotated, (left, top), (right, bottom), (180, 68, 58), 2)
                label = f"{detection['label']} {detection['confidence']:.2f}"
                cv2.putText(annotated, label, (left, max(18, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 68, 58), 1, cv2.LINE_AA)
        return detections, annotated, elapsed_ms

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.model_path),
            "model_classes": list(self.classes.values()),
            "configured_defect_classes": sorted(self.configured_defect_classes),
            "classes_by_product": self.class_products,
            "confidence_threshold": self.confidence,
            "validation_enabled": self.validated,
            "error": self.error,
        }