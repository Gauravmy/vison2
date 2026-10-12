"""Optional YOLO product localization; it does not establish product quality."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import MODEL_CONFIDENCE, PRODUCT_CATEGORY_CONFIDENCE, PRODUCT_MODEL_PATH, PRODUCT_MODEL_VALIDATED


PRODUCT_CATEGORIES = ("Marker", "Plastic bottle", "Phone charger")
PRODUCT_CATEGORY_MIN_CONFIDENCE = PRODUCT_CATEGORY_CONFIDENCE


@dataclass(frozen=True)
class ProductIdentification:
    category: str | None
    model_label: str | None
    confidence: float | None
    confident: bool
    source: str
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _category_for_label(label: str) -> tuple[str, bool] | None:
    normalized = re.sub(r"[_-]+", " ", label.casefold()).strip()
    if normalized == "marker":
        return "Marker", True
    if normalized == "plastic bottle":
        return "Plastic bottle", True
    if normalized in {"phone charger", "mobile phone charger"}:
        return "Phone charger", True
    if normalized == "bottle":
        return "Plastic bottle", False
    if normalized == "charger":
        return "Phone charger", False
    return None


class ProductRecognizer:
    def __init__(
        self,
        model_path: Path = PRODUCT_MODEL_PATH,
        confidence: float = MODEL_CONFIDENCE,
        validated: bool = PRODUCT_MODEL_VALIDATED,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.validated = validated
        self.model: Any = None
        self.error: str | None = None
        self.classes: dict[int, str] = {}
        if not self.model_path.is_file():
            self.error = "Product-recognition weights are not installed."
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.classes = {int(key): str(value) for key, value in self.model.names.items()}
        except Exception as error:
            self.error = f"Product model could not be loaded: {error}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, image: Any) -> tuple[list[dict[str, Any]], Any | None, float | None]:
        if self.model is None:
            return [], None, None
        import time

        started = time.perf_counter()
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = results[0]
        detections = []
        if result.probs is not None:
            top_classes = result.probs.top5
            top_confidences = result.probs.top5conf
            for rank, class_id in enumerate(top_classes):
                confidence = float(top_confidences[rank].item())
                if confidence < self.confidence:
                    continue
                class_id = int(class_id)
                detections.append({
                    "class_id": class_id,
                    "label": self.classes.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "box_xyxy": None,
                    "kind": "product classification candidate",
                })
        elif result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({
                    "class_id": class_id,
                    "label": self.classes.get(class_id, str(class_id)),
                    "confidence": float(box.conf.item()),
                    "box_xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    "kind": "product localization",
                })
        return detections, result.plot()[:, :, ::-1], elapsed_ms

    def identify(self, detections: list[dict[str, Any]]) -> ProductIdentification:
        """Map only actual model labels; generic labels remain explicitly uncertain."""
        if not self.available:
            return ProductIdentification(
                None, None, None, False, "unavailable", "No product-recognition weights are configured."
            )
        candidates = []
        for detection in detections:
            mapped = _category_for_label(str(detection.get("label", "")))
            if mapped is not None:
                category, label_is_specific = mapped
                confidence = float(detection.get("confidence", 0.0))
                candidates.append((confidence, category, str(detection.get("label", "")), label_is_specific))
        if not candidates:
            return ProductIdentification(
                None, None, None, False, "local product model",
                "The loaded model returned no class that maps to a supported product category.",
            )
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        confidence, category, label, label_is_specific = candidates[0]
        competing_category = next(
            (candidate for candidate in candidates[1:] if candidate[1] != category), None
        )
        clear_margin = competing_category is None or confidence - competing_category[0] >= 0.10
        confident = (
            self.validated and label_is_specific
            and confidence >= PRODUCT_CATEGORY_MIN_CONFIDENCE and clear_margin
        )
        if not label_is_specific:
            explanation = f"The model detected '{label}', which suggests {category} but does not confirm its material or subtype."
        elif not self.validated:
            explanation = "The model's category classes are explicit, but this artifact has not been marked validated on held-out data."
        elif confidence < PRODUCT_CATEGORY_MIN_CONFIDENCE:
            explanation = f"The model suggested {category}, but confidence is below the configured category threshold."
        elif not clear_margin:
            explanation = "The model returned competing supported product categories; manual confirmation is required."
        else:
            explanation = f"The model identified {category} from its '{label}' class."
        return ProductIdentification(category, label, confidence, confident, "local product model", explanation)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.model_path),
            "classes": list(self.classes.values()),
            "error": self.error,
            "minimum_category_confidence": PRODUCT_CATEGORY_MIN_CONFIDENCE,
            "validation_enabled": self.validated,
            "supported_category_classes": ["marker", "plastic bottle", "phone charger"],
            "limitation": "Product localization does not detect product defects or justify acceptance.",
        }