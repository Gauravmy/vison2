"""VisionInspect Pro Gradio application entry point."""

from __future__ import annotations

import json
import html
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

import config
from src.ai_vision import VisionProviderError, analyze_image, provider_status
from src.camera import camera_status
from src.database import InspectionDatabase
from src.decision_engine import Outcome, decide
from src.defect_inspection import DefectInspector
from src.image_quality import assess_image_quality, load_image, preprocess_image
from src.product_recognition import ProductIdentification, ProductRecognizer
from src.reporting import export_csv, generate_inspection_report, timestamped_csv_path
from src.voice import speak_async, voice_status


PRODUCTS = ["Marker", "Plastic bottle", "Phone charger"]
PRODUCT_FILTER_OPTIONS = ["All", *PRODUCTS, "Unknown"]
LOCAL_MODE = "Local Computer Vision"
AI_MODE = "AI-assisted image analysis"
LOCAL_PROVIDER_LABEL = "Local product-recognition model + OpenCV; no defect model"
AI_FALLBACK_PROVIDER_LABEL = "AI-assisted requested; provider unavailable, local checks only"
MODES = [LOCAL_MODE, AI_MODE]
OUTCOMES = [outcome.value for outcome in Outcome]
OUTCOME_PRESENTATION = {
    Outcome.PASS.value: ("pass", "Accepted"),
    Outcome.FAIL.value: ("fail", "Rejected"),
    Outcome.REVIEW.value: ("review", "Manual review required"),
    Outcome.RECAPTURE.value: ("recapture", "Better image required"),
}
DATABASE = InspectionDatabase(config.DATABASE_PATH)
PRODUCT_RECOGNIZER = ProductRecognizer()
DEFECT_INSPECTOR = DefectInspector()


def _save_rgb_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(load_image(image)).save(path, format="PNG", optimize=True)


def _annotate_status(image: Any, quality: dict[str, Any], model_error: str | None) -> Any:
    import cv2
    import numpy as np

    annotated = load_image(image).copy()
    lines = [
        f"Image {quality['width']} x {quality['height']} | blur variance {quality['blur_variance']:.1f}",
        "Defect model unavailable or not validated" if model_error else "No model annotation produced",
    ]
    for index, line in enumerate(lines):
        cv2.putText(annotated, line, (12, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (12, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 45, 42), 1, cv2.LINE_AA)
    return np.ascontiguousarray(annotated)


def _run_local_models(prepared: Any, manual_product: str | None):
    detections: list[dict[str, Any]] = []
    annotated = None
    model_versions: list[str] = []
    elapsed_ms = 0.0
    errors: list[str] = []
    product_rows: list[dict[str, Any]] = []
    identification = PRODUCT_RECOGNIZER.identify(product_rows)
    if PRODUCT_RECOGNIZER.available:
        try:
            product_rows, annotated, elapsed = PRODUCT_RECOGNIZER.detect(prepared)
            detections.extend(product_rows)
            identification = PRODUCT_RECOGNIZER.identify(product_rows)
            elapsed_ms += elapsed or 0.0
            model_versions.append(f"product:{PRODUCT_RECOGNIZER.model_path.name}")
        except Exception as error:
            errors.append(f"Product localization failed: {error}")
    selected_product = manual_product or (identification.category if identification.confident else None)
    if DEFECT_INSPECTOR.available and selected_product:
        try:
            defect_rows, defect_image, elapsed = DEFECT_INSPECTOR.inspect(prepared, selected_product)
            detections.extend(defect_rows)
            if defect_image is not None:
                annotated = defect_image
            elapsed_ms += elapsed or 0.0
            model_versions.append(f"defect:{DEFECT_INSPECTOR.model_path.name}")
        except Exception as error:
            errors.append(f"Defect model inference failed: {error}")
    return (
        detections, annotated, "; ".join(model_versions) or None, elapsed_ms, "; ".join(errors) or None,
        identification, selected_product,
    )


def _run_ai_analysis(mode: str, suitable: bool, image: Any, product: str | None):
    if mode != AI_MODE or not suitable:
        return None, None
    try:
        return analyze_image(image, product), None
    except VisionProviderError as error:
        return None, str(error)


def _decision_for(
    quality: dict[str, Any], detections: list[dict[str, Any]], ai_result: Any,
    inference_error: str | None, category_confident: bool,
):
    return decide(
        image_suitable=quality["suitable"],
        category_confident=category_confident,
        supported_defect_detected=any(item.get("kind") == "defined defect" for item in detections),
        validated_policy_passed=(
            config.AUTOMATED_PASS_ENABLED
            and DEFECT_INSPECTOR.available
            and inference_error is None
            and not any(item.get("kind") == "defined defect" for item in detections)
        ),
        ai_recommendation=ai_result.recommended_outcome if ai_result else None,
    )


def _provider_label(mode: str, ai_result: Any, model_version: str | None, detections: list[dict[str, Any]]) -> str:
    if mode == AI_MODE:
        if ai_result:
            return "Gemini vision API (AI-assisted)"
        if model_version:
            return f"Gemini vision unavailable; local model(s) used: {model_version}"
        return AI_FALLBACK_PROVIDER_LABEL
    if model_version and "defect:" in model_version:
        return "Validated local YOLO defect model + OpenCV"
    if model_version and "product:" in model_version:
        return LOCAL_PROVIDER_LABEL
    return "OpenCV image-quality checks; no defect model"


def _evaluate_image(original: Any, prepared: Any, manual_product: str | None, mode: str):
    quality = assess_image_quality(original).as_dict()
    detections = []
    elapsed_model_ms = 0.0
    model_version = None
    inference_error = None
    annotated = None
    identification = PRODUCT_RECOGNIZER.identify([])
    selected_product = manual_product
    if quality["suitable"]:
        (
            detections, annotated, model_version, elapsed_model_ms, inference_error,
            identification, selected_product,
        ) = _run_local_models(prepared, manual_product)
    ai_result, ai_error = _run_ai_analysis(mode, quality["suitable"], prepared, selected_product)
    if ai_error:
        inference_error = "; ".join(filter(None, [inference_error, ai_error]))
    ai_payload = ai_result.model_dump(mode="json") if ai_result else None
    ai_defects = (ai_payload or {}).get("potential_visible_defects", [])
    category_confident = manual_product is not None or (identification.confident and selected_product is not None)
    decision = _decision_for(quality, detections, ai_result, inference_error, category_confident)
    if annotated is None:
        annotated = _annotate_status(original, quality, DEFECT_INSPECTOR.error)
    explanation = decision.explanation
    if quality["concerns"]:
        explanation += " " + " ".join(quality["concerns"])
    if inference_error:
        explanation += f" Analysis note: {inference_error}"
    if quality["suitable"] and model_version and not any(item.get("kind") == "defined defect" for item in detections) and decision.outcome == Outcome.REVIEW:
        explanation += " No defined defects were detected, but automated acceptance criteria are not configured."
    if not category_confident:
        explanation += " " + identification.explanation
    elif manual_product is not None:
        explanation += f" Product category manually set to {manual_product}."
    else:
        explanation += f" Product category identified from model class '{identification.model_label}'."
    return (
        quality, detections, elapsed_model_ms, model_version, inference_error,
        ai_payload, ai_defects, annotated, decision, explanation,
        identification, selected_product, category_confident,
    )


def _ai_display(ai_payload: dict[str, Any] | None, mode: str, inference_error: str | None) -> str:
    if ai_payload:
        return json.dumps(ai_payload, indent=2, ensure_ascii=False)
    if mode == AI_MODE:
        return inference_error or "Vision analysis was not run because the image needs recapture."
    return "Not requested."


def run_inspection(image: Any, product: str | None, mode: str, voice_enabled: bool):
    if image is None:
        raise gr.Error("Capture a frame or upload an image before inspecting.")
    started = time.perf_counter()
    try:
        original = load_image(image)
        prepared = preprocess_image(original)
    except (ValueError, TypeError) as error:
        raise gr.Error(f"Image could not be loaded: {error}") from error

    (
        quality, detections, elapsed_model_ms, model_version, inference_error,
        ai_payload, ai_defects, annotated, decision, explanation,
        identification, selected_product, category_confident,
    ) = _evaluate_image(original, prepared, product, mode)
    if annotated is None:
        annotated = _annotate_status(original, quality, DEFECT_INSPECTOR.error)

    elapsed_ms = (time.perf_counter() - started) * 1000
    inspection_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config.INSPECTION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    original_path = config.INSPECTION_IMAGE_DIR / f"{inspection_id}-original.png"
    annotated_path = config.INSPECTION_IMAGE_DIR / f"{inspection_id}-annotated.png"
    _save_rgb_image(original, original_path)
    _save_rgb_image(annotated, annotated_path)

    provider = _provider_label(mode, ai_payload, model_version, detections)
    record = {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "product": selected_product or "Unknown",
        "mode": mode,
        "provider": provider,
        "outcome": decision.outcome.value,
        "explanation": explanation,
        "detections": detections,
        "ai_observations": ai_defects,
        "quality": quality,
        "model_version": model_version,
        "processing_ms": round(elapsed_ms, 2),
        "original_image_path": str(original_path),
        "annotated_image_path": str(annotated_path),
    }
    DATABASE.add_inspection(record)

    if voice_enabled:
        voice_message = {
            Outcome.RECAPTURE: "Please recapture the image.",
            Outcome.REVIEW: "Inspection requires manual review.",
            Outcome.FAIL: "Inspection completed. Please review the result.",
            Outcome.PASS: "Inspection completed.",
        }[decision.outcome]
        speak_async(voice_message)

    detection_columns = ["kind", "label", "confidence", "box_xyxy", "product"]
    detections_frame = pd.DataFrame(detections, columns=detection_columns)
    ai_text = _ai_display(ai_payload, mode, inference_error)
    method = (
        f"Product category: {selected_product or 'Unknown'}\n"
        f"Category source: {'manual override' if product else identification.source}\n"
        f"Category identification: {identification.explanation}\n"
        f"AI category suggestion: {(ai_payload or {}).get('suspected_category', 'not provided') if not product and not category_confident else 'not used'}\n"
        f"Category model label: {identification.model_label or 'none'}\n"
        f"Category model confidence: {identification.confidence if identification.confidence is not None else 'not available'}\n"
        f"Source: {provider}\nModel: {model_version or 'none'}\n"
        f"Inference time: {elapsed_model_ms:.2f} ms\nTotal processing time: {elapsed_ms:.2f} ms"
    )
    metadata = f"Inspection ID: {inspection_id}\nTimestamp: {timestamp}\nHuman review required: {'Yes' if decision.human_review_required else 'No'}\nCategory confidently identified: {'Yes' if category_confident else 'No'}"
    return (
        original, annotated, decision.outcome.value, explanation, detections_frame,
        ai_text, method, quality, metadata,
    )


def render_status_card(outcome: str, explanation: str) -> str:
    status_class, label = OUTCOME_PRESENTATION.get(outcome, ("review", "Manual review required"))
    return (
        f'<section class="vi-status vi-status--{status_class}" role="status">'
        f'<span class="vi-status-dot" aria-hidden="true"></span>'
        f'<div><strong>{html.escape(outcome)}</strong><span>{html.escape(label)}</span>'
        f'<p>{html.escape(explanation)}</p></div></section>'
    )


def inspect_selected_source(
    camera_image: Any,
    upload_image: Any,
    source: str,
    product: str,
    mode: str,
    voice_enabled: bool,
    category_mode: str = "Manual override",
):
    image = camera_image if source == "Camera" else upload_image
    manual_product = product if category_mode == "Manual override" else None
    result = run_inspection(image, manual_product, mode, voice_enabled)
    original, annotated, outcome, explanation, detections, ai_text, method, quality, metadata = result
    try:
        ai_data = json.loads(ai_text)
    except (json.JSONDecodeError, TypeError):
        ai_data = {"message": ai_text}
    return (
        gr.update(visible=False),
        gr.update(value=render_status_card(outcome, explanation), visible=True),
        gr.update(value=original, visible=True),
        gr.update(value=annotated, visible=True),
        detections.to_dict(orient="records"),
        ai_data,
        quality,
        {"inspection method": method, "audit": metadata},
    )


def toggle_image_source(source: str):
    camera_selected = source == "Camera"
    prompt = (
        "Camera permission required; capture a frame to inspect."
        if camera_selected else "Choose a product photo from this device."
    )
    return gr.update(visible=camera_selected), gr.update(visible=not camera_selected), prompt


def toggle_category_mode(category_mode: str):
    manual = category_mode == "Manual override"
    message = (
        "The chosen category is recorded as a manual override."
        if manual else "Uncertain match? Review required."
    )
    return gr.update(visible=manual), message


def source_feedback(image: Any, source: str) -> str:
    if image is not None:
        return "Image captured. Ready to inspect." if source == "Camera" else "Image uploaded. Ready to inspect."
    if source == "Camera":
        return "Camera permission required; capture a frame to inspect."
    return "Place a product in view or upload an image."


def reset_inspection_ui():
    return (
        gr.update(value=None, visible=True), gr.update(value=None, visible=False), "Camera",
        "Auto-detect", gr.update(value=PRODUCTS[0], visible=False),
        "Uncertain match? Review required.",
        "Camera permission required; capture a frame to inspect.",
        gr.update(visible=True), gr.update(visible=False),
        gr.update(value=None, visible=False), gr.update(value=None, visible=False),
        {}, {"message": "Not requested."}, {}, {},
    )


def _dashboard_frames(data: dict[str, Any]):
    observations = pd.DataFrame(data["common_observations"], columns=["observation", "count"])
    recent_rows = [
        {key: record[key] for key in ("inspection_id", "timestamp", "product", "provider", "outcome")}
        for record in data["recent"]
    ]
    recent = pd.DataFrame(recent_rows, columns=["inspection_id", "timestamp", "product", "provider", "outcome"])
    figure = go.Figure()
    category_figure = go.Figure()
    colors = {
        Outcome.PASS.value: "#3f795d",
        Outcome.FAIL.value: "#b4534d",
        Outcome.REVIEW.value: "#ad7a30",
        Outcome.RECAPTURE.value: "#78817d",
    }
    if data["trend"]:
        trend = pd.DataFrame(data["trend"])
        for outcome, entries in trend.groupby("outcome"):
            figure.add_trace(go.Scatter(x=entries["day"], y=entries["count"], mode="lines+markers", name=outcome, line={"color": colors.get(outcome, "#78817d")}))
    else:
        figure.add_annotation(text="No inspections recorded", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    if data["by_category"]:
        category = pd.DataFrame(data["by_category"])
        for outcome, entries in category.groupby("outcome"):
            category_figure.add_trace(go.Bar(x=entries["product"], y=entries["count"], name=outcome, marker_color=colors.get(outcome, "#78817d")))
    else:
        category_figure.add_annotation(text="No inspections match these filters", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    common_layout = {"paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff", "font": {"color": "#39413e", "family": "Segoe UI, Arial, sans-serif", "size": 12}, "margin": {"l": 36, "r": 16, "t": 20, "b": 38}, "height": 290, "legend": {"orientation": "h", "y": 1.12, "x": 0}}
    figure.update_layout(**common_layout, xaxis_title="Date", yaxis_title="Inspections", legend_title_text="Outcome")
    category_figure.update_layout(**common_layout, barmode="stack", xaxis_title="Product", yaxis_title="Inspections", legend_title_text="Outcome")
    return observations, recent, figure, category_figure


def refresh_dashboard(product: str = "All", outcome: str = "All", date_from: str = "", date_to: str = ""):
    data = DATABASE.dashboard_data(product=product, outcome=outcome, date_from=date_from or "", date_to=date_to or "")
    observations, recent, figure, category_figure = _dashboard_frames(data)
    counts = data["counts"]
    if outcome != "All":
        acceptance_rate = "Unavailable while filtering by outcome"
    elif data["total"]:
        acceptance_rate = f"{data['accepted_rate']:.1%} of {data['total']} inspections"
    else:
        acceptance_rate = "Unavailable until an inspection is recorded"
    metrics = _metric_summary(data["total"], counts, acceptance_rate)
    return (
        metrics, figure, category_figure, observations, recent,
    )


def _metric_summary(total: int, counts: dict[str, int], acceptance_rate: str) -> str:
    metrics = [
        ("Total inspections", total),
        ("Accepted", counts[Outcome.PASS.value]),
        ("Rejected", counts[Outcome.FAIL.value]),
        ("Manual review", counts[Outcome.REVIEW.value]),
        ("Recapture", counts[Outcome.RECAPTURE.value]),
    ]
    cells = "".join(
        f'<div class="vi-metric"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in metrics
    )
    return f'<div class="vi-metrics">{cells}<p>Acceptance rate <strong>{html.escape(acceptance_rate)}</strong></p></div>'


def _history_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    fields = ["inspection_id", "timestamp", "product", "outcome", "provider", "explanation"]
    return pd.DataFrame([{key: row.get(key) for key in fields} for row in records], columns=fields)


def search_history(search: str, product: str, outcome: str, method: str, date_from: str, date_to: str):
    records = DATABASE.list_inspections(
        search=search or "", product=product, outcome=outcome, method=method,
        date_from=date_from or "", date_to=date_to or "",
    )
    return _history_frame(records), f"{len(records)} matching inspection(s)."


def download_history(search: str, product: str, outcome: str, method: str, date_from: str, date_to: str):
    records = DATABASE.list_inspections(
        search=search or "", product=product, outcome=outcome, method=method,
        date_from=date_from or "", date_to=date_to or "",
    )
    path = timestamped_csv_path(config.ROOT_DIR / "artifacts" / "exports")
    return export_csv(records, path)


def create_report(inspection_id: str):
    record = DATABASE.get_inspection((inspection_id or "").strip())
    if record is None:
        return None, "No inspection found for that ID."
    path = generate_inspection_report(record, config.ROOT_DIR / "artifacts" / "reports")
    return path, f"Report generated for {record['inspection_id']}."


def open_history_record(inspection_id: str):
    record = DATABASE.get_inspection((inspection_id or "").strip())
    if record is None:
        return (
            gr.update(value='<div class="vi-record-note">No inspection found. Check the ID and try again.</div>', visible=True),
            gr.update(value=None, visible=False), gr.update(value=None, visible=False), {},
        )
    detail = {
        "inspection ID": record["inspection_id"],
        "timestamp": record["timestamp"],
        "product": record["product"],
        "method": record["provider"],
        "processing time (ms)": record["processing_ms"],
        "quality checks": record["quality"],
        "detections": record["detections"],
        "AI-assisted observations": record["ai_observations"],
    }
    original_path = record.get("original_image_path")
    annotated_path = record.get("annotated_image_path")
    original_value = original_path if original_path and Path(original_path).is_file() else None
    annotated_value = annotated_path if annotated_path and Path(annotated_path).is_file() else None
    return (
        gr.update(value=render_status_card(record["outcome"], record["explanation"]), visible=True),
        gr.update(value=original_value, visible=original_value is not None),
        gr.update(value=annotated_value, visible=annotated_value is not None),
        gr.update(value=detail, visible=True),
    )


def system_status():
    try:
        DATABASE.dashboard_data()
        database_state = {"available": True, "path": str(config.DATABASE_PATH)}
    except Exception as error:
        database_state = {"available": False, "error": str(error)}
    return {
        "application": {"name": config.APP_NAME, "version": config.APP_VERSION},
        "camera": camera_status(config.CAMERA_INDEX),
        "local_models": {
            "defect_inspection": DEFECT_INSPECTOR.status(),
            "product_localization": PRODUCT_RECOGNIZER.status(),
        },
        "decision_policy": {
            "defect_model_validation_attested": config.DEFECT_MODEL_VALIDATED,
            "automated_pass_enabled": config.AUTOMATED_PASS_ENABLED,
            "note": "Both settings require documented domain-specific evaluation; neither is enabled by default.",
        },
        "image_quality_thresholds": {
            "minimum_dimension_px": 160,
            "minimum_laplacian_variance": 40.0,
            "mean_brightness_range": [35.0, 220.0],
            "maximum_clipped_pixel_fraction": 0.35,
        },
        "image_limits": {
            "upload_size_mb": config.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024),
            "maximum_pixels": config.MAX_IMAGE_PIXELS,
        },
        "vision_api": provider_status(),
        "database": database_state,
        "voice": voice_status(),
        "storage": {"inspection_images": str(config.INSPECTION_IMAGE_DIR)},
        "warnings": [
            "Generic product recognition is not defect detection.",
            "A validated model and automated pass policy are not configured by default; clean images require review.",
            "On cloud deployments, webcam permission and hardware camera access depend on the visitor's browser and host setup.",
        ],
    }


def system_indicator() -> str:
    if PRODUCT_RECOGNIZER.available and PRODUCT_RECOGNIZER.validated and DEFECT_INSPECTOR.available:
        state = "Product and validated defect models loaded"
    elif PRODUCT_RECOGNIZER.available and PRODUCT_RECOGNIZER.validated:
        state = "Validated category model loaded · defect decisions require review"
    elif PRODUCT_RECOGNIZER.available and not PRODUCT_RECOGNIZER.validated:
        state = "Product model loaded · category validation required"
    elif PRODUCT_RECOGNIZER.available:
        state = "Product recognition available · defect decisions require review"
    elif DEFECT_INSPECTOR.available:
        state = "Validated defect model loaded · automatic category detection unavailable"
    else:
        state = "Local image checks available · product and defect models not configured"
    return html.escape(state)


CUSTOM_CSS = """
.gradio-container {
    --vi-bg: #f3f2ed;
    --vi-surface: #fffefa;
    --vi-text: #242b28;
    --vi-muted: #68716d;
    --vi-line: #dfdfd7;
    --vi-accent: #285546;
    --vi-accent-soft: #e9f0eb;
    --body-background-fill: #f3f2ed;
    --body-text-color: #242b28;
    --background-fill-primary: #f3f2ed;
    --background-fill-secondary: #fffefa;
    --block-background-fill: #fffefa;
    --block-border-color: #dfdfd7;
    --block-label-background-fill: transparent;
    --block-label-text-color: #68716d;
    --input-background-fill: #fffefa;
    --input-border-color: #dfdfd7;
    --button-primary-background-fill: #285546;
    --button-primary-background-fill-hover: #21483b;
    --button-primary-text-color: #fffefa;
    --radius-lg: 8px;
    --radius-md: 6px;
    max-width: none !important;
    min-height: 100vh;
    background: var(--vi-bg) !important;
    color: var(--vi-text) !important;
    color-scheme: light;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
}
.vi-page { max-width: 1440px; margin: 0 auto; padding: 18px 34px 48px; }
.vi-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 4px 0 16px; margin-bottom: 10px; border-bottom: 1px solid var(--vi-line); }
.vi-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.vi-mark { position: relative; display: block; flex: 0 0 30px; width: 30px; height: 30px; border: 1.5px solid var(--vi-accent); border-radius: 7px; }
.vi-mark::before, .vi-mark::after { content: ""; position: absolute; border: 1.5px solid var(--vi-accent); border-radius: 2px; }
.vi-mark::before { inset: 6px; }
.vi-mark::after { width: 5px; height: 5px; right: 5px; top: 5px; background: var(--vi-accent); }
.vi-brand-name { margin: 0; color: var(--vi-text); font-size: 17px; line-height: 1.25; font-weight: 650; }
.vi-brand-subtitle { margin-top: 3px; color: var(--vi-muted); font-size: 12px; }
.vi-system { display: inline-flex; align-items: center; gap: 8px; color: var(--vi-muted); font-size: 12px; white-space: normal; }
.vi-system-dot, .vi-status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #73827b; }
.vi-system-dot { background: #547b61; }
.vi-main-tabs [role="tablist"] { gap: 20px; border-bottom: 1px solid var(--vi-line); background: transparent !important; }
.vi-main-tabs [role="tab"] { border: 0 !important; border-radius: 0 !important; background: transparent !important; color: var(--vi-muted) !important; padding: 11px 3px 10px !important; font-size: 13px !important; box-shadow: none !important; }
.vi-main-tabs [role="tab"][aria-selected="true"] { color: var(--vi-accent) !important; border-bottom: 2px solid var(--vi-accent) !important; font-weight: 650; }
.vi-workspace { align-items: flex-start !important; gap: 26px !important; padding-top: 14px; }
.vi-capture-column { min-width: 290px !important; padding-right: 4px; gap: 8px !important; }
.vi-capture-column > .block, .vi-capture-column > .form, .vi-capture-column > .row { margin-block: 0 !important; }
.vi-result-column { min-width: 0 !important; }
.vi-section-title { margin: 0 0 5px; color: var(--vi-text); font-size: 14px; font-weight: 650; }
.vi-section-note { margin: 0 0 9px; color: var(--vi-muted); font-size: 12px; line-height: 1.45; }
.vi-source-choice { margin: 10px 0 8px; }
.vi-source-choice { border: 0 !important; padding: 0 !important; }
.vi-source-choice label { border: 1px solid var(--vi-line) !important; border-radius: 6px !important; background: #fffefa !important; color: var(--vi-muted) !important; padding: 6px 11px !important; font-size: 12px !important; }
.vi-source-choice label:has(input:checked) { border-color: #9caf9f !important; background: #e9f0eb !important; color: #285546 !important; }
.vi-source-choice input { accent-color: #285546 !important; }
.vi-source-image { min-height: 190px; }
.vi-source-image img { object-fit: contain !important; }
.vi-source-feedback { min-height: 18px; margin: 0; padding: 0 !important; color: var(--vi-muted); font-size: 11px; line-height: 1.4; }
.vi-controls-row { gap: 14px !important; }
.vi-primary-button button { min-height: 42px !important; border: 1px solid #244c3f !important; border-radius: 7px !important; background: #285546 !important; color: #fffefa !important; font-weight: 600 !important; box-shadow: none !important; }
.vi-primary-button button:hover { background: #21483b !important; }
.vi-clear-button button { min-height: 40px; border: 1px solid var(--vi-line) !important; border-radius: 7px !important; background: transparent !important; color: var(--vi-muted) !important; box-shadow: none !important; }
.vi-result-empty { display: flex; min-height: 220px; align-items: center; justify-content: center; border: 1px dashed #d4d5ce; border-radius: 8px; background: rgba(255, 254, 250, .66); text-align: center; }
.vi-empty-inner { max-width: 330px; padding: 24px; }
.vi-empty-mark { display: grid; width: 34px; height: 34px; margin: 0 auto 12px; place-items: center; border: 1px solid #aab7af; border-radius: 8px; color: var(--vi-accent); font-size: 15px; }
.vi-empty-title { margin: 0 0 5px; color: var(--vi-text); font-size: 15px; font-weight: 650; }
.vi-empty-copy { margin: 0; color: var(--vi-muted); font-size: 12px; line-height: 1.55; }
.vi-status { display: flex; align-items: flex-start; gap: 11px; margin: 0 0 14px; padding: 14px 16px; border-left: 3px solid #78817d; border-radius: 6px; background: var(--vi-surface); }
.vi-status-dot { flex: 0 0 9px; width: 9px; height: 9px; margin-top: 5px; }
.vi-status strong { display: block; color: var(--vi-text); font-size: 14px; }
.vi-status span:not(.vi-status-dot) { display: block; margin-top: 2px; color: var(--vi-muted); font-size: 12px; }
.vi-status p { margin: 8px 0 0; color: #424b47; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; }
.vi-status--pass { border-color: #3f795d; }.vi-status--pass .vi-status-dot { background: #3f795d; }
.vi-status--fail { border-color: #b4534d; }.vi-status--fail .vi-status-dot { background: #b4534d; }
.vi-status--review { border-color: #ad7a30; }.vi-status--review .vi-status-dot { background: #ad7a30; }
.vi-status--recapture { border-color: #78817d; }.vi-status--recapture .vi-status-dot { background: #78817d; }
.vi-result-images { gap: 12px !important; }
.vi-result-image { min-width: 0 !important; }
.vi-result-image img { max-height: 330px !important; object-fit: contain !important; }
.vi-evidence-accordion, .vi-settings-accordion { margin-top: 14px; border-top: 1px solid var(--vi-line) !important; border-radius: 0 !important; background: transparent !important; }
.vi-evidence-accordion summary, .vi-settings-accordion summary { padding: 12px 2px !important; color: var(--vi-text) !important; font-size: 13px !important; font-weight: 600 !important; }
.vi-evidence-tabs [role="tablist"] { gap: 12px; border-bottom: 1px solid var(--vi-line); background: transparent !important; }
.vi-evidence-tabs [role="tab"] { border: 0 !important; border-radius: 0 !important; background: transparent !important; color: var(--vi-muted) !important; box-shadow: none !important; }
.vi-filter-row { align-items: end !important; gap: 12px !important; margin: 16px 0 12px; }
.vi-dashboard-charts { gap: 16px !important; margin-top: 10px; }
.vi-chart-surface { min-width: 0 !important; padding: 10px 12px 4px; border-radius: 8px; background: var(--vi-surface); }
.vi-dashboard-table { margin-top: 16px; }
.vi-metrics { display: grid; grid-template-columns: repeat(5, minmax(100px, 1fr)); gap: 0; margin: 14px 0 0; padding: 8px 0 12px; border-top: 1px solid var(--vi-line); border-bottom: 1px solid var(--vi-line); }
.vi-metric { display: flex; flex-direction: column; gap: 6px; padding: 7px 14px; border-right: 1px solid var(--vi-line); }
.vi-metric:first-child { padding-left: 0; }.vi-metric:nth-child(5) { border-right: 0; }
.vi-metric span { color: var(--vi-muted); font-size: 11px; }.vi-metric strong { color: var(--vi-text); font-size: 22px; font-weight: 600; }
.vi-metrics > p { grid-column: 1 / -1; margin: 9px 0 0; color: var(--vi-muted); font-size: 11px; }
.vi-metrics > p strong { margin-left: 5px; color: var(--vi-text); font-weight: 550; }
.vi-record-layout { gap: 16px !important; margin-top: 14px; }
.vi-record-images { gap: 12px !important; }
.vi-record-image img { max-height: 280px !important; object-fit: contain !important; }
.vi-record-note { padding: 13px 0; color: var(--vi-muted); font-size: 13px; }
.vi-model-intro { margin: 15px 0; padding: 13px 15px; border-left: 3px solid #78817d; background: #eeeee8; color: #4e5853; font-size: 13px; line-height: 1.55; }
.vi-training-copy { max-width: 920px; color: #48514d; font-size: 13px; line-height: 1.65; }
.vi-training-copy h3 { margin-top: 20px; color: var(--vi-text); font-size: 14px; }
.vi-training-copy code { color: #244c3f; }
.vi-label { color: var(--vi-muted); font-size: 12px; }
.vi-page button, .vi-page input, .vi-page textarea { font-family: inherit !important; }
.vi-page input, .vi-page textarea { border-color: var(--vi-line) !important; border-radius: 6px !important; background: var(--vi-surface) !important; color: var(--vi-text) !important; box-shadow: none !important; }
.vi-page [role="radiogroup"] { color: var(--vi-text); }
.vi-page [data-testid="block-info"] { border: 0 !important; border-radius: 0 !important; padding: 0 !important; background: transparent !important; color: var(--vi-muted) !important; box-shadow: none !important; }
footer { display: none !important; }
@media (max-width: 760px) {
    .vi-page { padding: 14px 18px 34px; }
    .vi-workspace { flex-direction: column !important; gap: 22px !important; }
    .vi-capture-column, .vi-result-column { min-width: 100% !important; width: 100% !important; }
    .vi-filter-row { flex-wrap: wrap !important; }
    .vi-filter-row > * { min-width: min(100%, 210px) !important; }
}
@media (max-width: 620px) {
    .vi-page { padding: 12px 12px 28px; }
    .vi-header { align-items: flex-start; flex-direction: column; gap: 10px; }
    .vi-system { max-width: 100%; }
    .vi-main-tabs [role="tablist"] { gap: 12px; overflow-x: auto; }
    .vi-main-tabs [role="tab"] { white-space: nowrap; font-size: 12px !important; }
    .vi-metrics { grid-template-columns: repeat(2, minmax(110px, 1fr)); }
    .vi-metric { border-right: 1px solid var(--vi-line); border-bottom: 1px solid var(--vi-line); }
    .vi-metric:nth-child(2n) { border-right: 0; }
    .vi-metric:nth-child(5) { border-bottom: 0; }
    .vi-result-images, .vi-record-images { flex-direction: column !important; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=config.APP_NAME,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    ) as demo:
        with gr.Column(elem_classes="vi-page"):
            gr.HTML(
                '<header class="vi-header"><div class="vi-brand"><span class="vi-mark" aria-hidden="true"></span>'
                '<div><div class="vi-brand-name">VisionInspect Pro</div><div class="vi-brand-subtitle">Quality inspection workspace</div></div></div>'
                f'<div class="vi-system"><span class="vi-system-dot" aria-hidden="true"></span>{system_indicator()}</div></header>',
                padding=False,
            )
            with gr.Tabs(elem_classes="vi-main-tabs"):
                with gr.Tab("Inspect"):
                    with gr.Row(elem_classes="vi-workspace"):
                        with gr.Column(scale=4, min_width=310, elem_classes="vi-capture-column"):
                            gr.HTML('<h2 class="vi-section-title">Capture</h2><p class="vi-section-note">Capture a product or upload a JPG, PNG, or WEBP image.</p>', padding=False)
                            source_choice = gr.Radio(
                                ["Camera", "Upload"], value="Camera", label="Image source",
                                elem_classes="vi-source-choice", container=False,
                            )
                            camera_image = gr.Image(
                                sources=["webcam"], type="numpy", label="Camera frame",
                                height=210, elem_classes="vi-source-image",
                            )
                            upload_image = gr.Image(
                                sources=["upload"], type="numpy", label="Product photo",
                                height=210, visible=False, elem_classes="vi-source-image",
                            )
                            source_message = gr.Markdown(
                                "Camera permission required; capture a frame to inspect.",
                                elem_classes="vi-source-feedback",
                            )
                            category_mode = gr.Radio(
                                ["Auto-detect", "Manual override"], value="Auto-detect",
                                label="Product category", elem_classes="vi-source-choice", container=False,
                            )
                            product_input = gr.Dropdown(PRODUCTS, value=PRODUCTS[0], label="Manual category", visible=False)
                            category_message = gr.Markdown(
                                "Uncertain match? Review required.",
                                elem_classes="vi-source-feedback",
                            )
                            with gr.Row(elem_classes="vi-controls-row"):
                                mode_input = gr.Dropdown(MODES, value=MODES[0], label="Inspection method", scale=3)
                                voice_input = gr.Checkbox(value=False, label="Voice feedback", elem_classes="vi-label", scale=1)
                            with gr.Row(elem_classes="vi-controls-row"):
                                inspect_button = gr.Button("Inspect product", variant="primary", elem_classes="vi-primary-button", scale=2)
                                reset_button = gr.Button("Clear", elem_classes="vi-clear-button", scale=1)
                        with gr.Column(scale=8, min_width=380, elem_classes="vi-result-column"):
                            empty_state = gr.HTML(
                                '<div class="vi-result-empty"><div class="vi-empty-inner"><span class="vi-empty-mark" aria-hidden="true">+</span>'
                                '<h2 class="vi-empty-title">Ready for inspection</h2><p class="vi-empty-copy">Capture an image or upload a product photo to begin. Results and evidence will appear here.</p></div></div>',
                                padding=False,
                                elem_classes="vi-empty-state",
                            )
                            status_card = gr.HTML(visible=False, elem_classes="vi-status-card", padding=False)
                            with gr.Row(elem_classes="vi-result-images"):
                                original_output = gr.Image(label="Original", interactive=False, visible=False, elem_classes="vi-result-image")
                                annotated_output = gr.Image(label="Annotated", interactive=False, visible=False, elem_classes="vi-result-image")
                            with gr.Accordion("Inspection evidence and measurements", open=False, elem_classes="vi-evidence-accordion"):
                                with gr.Tabs(elem_classes="vi-evidence-tabs"):
                                    with gr.Tab("Detections"):
                                        detection_output = gr.JSON(label="Actual model detections")
                                    with gr.Tab("AI observations"):
                                        ai_output = gr.JSON(label="AI-assisted observations")
                                    with gr.Tab("Image quality"):
                                        quality_output = gr.JSON(label="Measured image-quality checks")
                                    with gr.Tab("Method and audit"):
                                        metadata_output = gr.JSON(label="Inspection method, timing, and audit ID")

                    source_choice.change(toggle_image_source, inputs=source_choice, outputs=[camera_image, upload_image, source_message])
                    category_mode.change(toggle_category_mode, inputs=category_mode, outputs=[product_input, category_message])
                    camera_image.change(lambda image: source_feedback(image, "Camera"), inputs=camera_image, outputs=source_message)
                    upload_image.change(lambda image: source_feedback(image, "Upload"), inputs=upload_image, outputs=source_message)
                    inspection_outputs = [empty_state, status_card, original_output, annotated_output, detection_output, ai_output, quality_output, metadata_output]
                    inspect_button.click(
                        inspect_selected_source,
                        inputs=[camera_image, upload_image, source_choice, product_input, mode_input, voice_input, category_mode],
                        outputs=inspection_outputs,
                    )
                    reset_button.click(
                        reset_inspection_ui,
                        outputs=[camera_image, upload_image, source_choice, category_mode, product_input, category_message, source_message, empty_state, status_card, original_output, annotated_output, detection_output, ai_output, quality_output, metadata_output],
                    )

                with gr.Tab("Dashboard"):
                    gr.HTML('<h2 class="vi-section-title">Inspection activity</h2><p class="vi-section-note">All figures below are calculated from recorded inspections.</p>')
                    with gr.Row(elem_classes="vi-filter-row"):
                        dashboard_product = gr.Dropdown(PRODUCT_FILTER_OPTIONS, value="All", label="Product", scale=1)
                        dashboard_outcome = gr.Dropdown(["All", *OUTCOMES], value="All", label="Outcome", scale=1)
                        dashboard_from = gr.Textbox(label="From date", placeholder="YYYY-MM-DD", scale=1)
                        dashboard_to = gr.Textbox(label="To date", placeholder="YYYY-MM-DD", scale=1)
                        dashboard_apply = gr.Button("Apply filters", variant="primary", elem_classes="vi-primary-button", scale=1)
                    metrics_html = gr.HTML(elem_classes="vi-dashboard-metrics")
                    with gr.Row(elem_classes="vi-dashboard-charts"):
                        trend_plot = gr.Plot(label="Inspection trend · last 30 days", elem_classes="vi-chart-surface")
                        category_plot = gr.Plot(label="Outcomes by product", elem_classes="vi-chart-surface")
                    with gr.Row(elem_classes="vi-dashboard-charts"):
                        observation_table = gr.Dataframe(label="Common recorded defect classes", interactive=False, elem_classes="vi-dashboard-table")
                        recent_table = gr.Dataframe(label="Recent inspections", interactive=False, wrap=True, elem_classes="vi-dashboard-table")
                    dashboard_outputs = [metrics_html, trend_plot, category_plot, observation_table, recent_table]
                    dashboard_inputs = [dashboard_product, dashboard_outcome, dashboard_from, dashboard_to]
                    dashboard_apply.click(refresh_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)
                    demo.load(refresh_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)

                with gr.Tab("History"):
                    gr.HTML('<h2 class="vi-section-title">Inspection history</h2><p class="vi-section-note">Search saved inspections, inspect evidence, or export the filtered records.</p>')
                    with gr.Row(elem_classes="vi-filter-row"):
                        history_search = gr.Textbox(label="Inspection ID", placeholder="Search by ID", scale=2)
                        history_product = gr.Dropdown(PRODUCT_FILTER_OPTIONS, value="All", label="Product", scale=1)
                        history_outcome = gr.Dropdown(["All", *OUTCOMES], value="All", label="Outcome", scale=1)
                        history_from = gr.Textbox(label="From date", placeholder="YYYY-MM-DD", scale=1)
                        history_to = gr.Textbox(label="To date", placeholder="YYYY-MM-DD", scale=1)
                    history_method_state = gr.State("All")
                    with gr.Row(elem_classes="vi-controls-row"):
                        search_button = gr.Button("Search history", variant="primary", elem_classes="vi-primary-button")
                        export_button = gr.Button("Export CSV", elem_classes="vi-clear-button")
                    history_message = gr.Markdown(elem_classes="vi-label")
                    history_table = gr.Dataframe(
                        headers=["Inspection ID", "Date and time", "Product", "Outcome", "Method", "Decision summary"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        label="Records", interactive=False, wrap=True,
                    )
                    search_inputs = [history_search, history_product, history_outcome, history_method_state, history_from, history_to]
                    search_button.click(search_history, inputs=search_inputs, outputs=[history_table, history_message])
                    export_file = gr.File(label="CSV export", elem_classes="vi-dashboard-table")
                    export_button.click(download_history, inputs=search_inputs, outputs=export_file)
                    with gr.Accordion("Open a saved inspection", open=False, elem_classes="vi-evidence-accordion"):
                        with gr.Row(elem_classes="vi-controls-row"):
                            record_id = gr.Textbox(label="Inspection ID", placeholder="Paste an ID from the table", scale=3)
                            open_record_button = gr.Button("Open record", variant="primary", elem_classes="vi-primary-button", scale=1)
                            report_button = gr.Button("Generate report", elem_classes="vi-clear-button", scale=1)
                        record_status = gr.HTML('<div class="vi-record-note">Enter an inspection ID to view its saved images and decision.</div>', padding=False)
                        with gr.Row(elem_classes="vi-record-images"):
                            record_original = gr.Image(label="Original evidence", interactive=False, visible=False, elem_classes="vi-record-image")
                            record_annotated = gr.Image(label="Annotated evidence", interactive=False, visible=False, elem_classes="vi-record-image")
                        record_details = gr.JSON(label="Recorded inspection details", visible=False)
                        open_record_button.click(open_history_record, inputs=record_id, outputs=[record_status, record_original, record_annotated, record_details])
                        report_message = gr.Markdown(elem_classes="vi-label")
                        report_file = gr.File(label="Individual inspection report")
                        report_button.click(create_report, inputs=record_id, outputs=[report_file, report_message])

                with gr.Tab("Models & settings"):
                    gr.HTML('<h2 class="vi-section-title">Models and system settings</h2><p class="vi-section-note">Readiness reflects installed models and configured providers, not settings fields alone.</p>')
                    gr.HTML('<div class="vi-model-intro"><strong>Current inspection scope:</strong> image quality is measured locally. No product-defect weights are included by default, so defect-free quality is not asserted and review is required.</div>')
                    status_refresh = gr.Button("Refresh system status", elem_classes="vi-clear-button")
                    with gr.Accordion("Runtime, models, and providers", open=True, elem_classes="vi-settings-accordion"):
                        status_json = gr.JSON(label="Live system status")
                    with gr.Accordion("Dataset and model training", open=False, elem_classes="vi-settings-accordion"):
                        gr.Markdown("""
### Defect dataset layout
Place category-specific images and YOLO-format labels under `data/dataset/images/{train,val}` and `data/dataset/labels/{train,val}`. Keep train, validation, and final holdout sets separate by physical item or capture session to prevent leakage.

### Product category recognition
Use a separate category dataset at `data/product_dataset/` with the classes `marker`, `plastic_bottle`, and `phone_charger` from `data/product_dataset.yaml`. Train and evaluate it independently:
```powershell
python scripts/train_detector.py --data data/product_dataset.yaml --project runs/product-recognition --epochs 100 --imgsz 640
python scripts/validate_model.py --weights runs/product-recognition/run/weights/best.pt --data data/product_dataset.yaml --split val
```
After reviewing held-out metrics, copy weights to `models/product_detector.pt`, select `PRODUCT_CATEGORY_CONFIDENCE`, and only then set `PRODUCT_MODEL_VALIDATED=true`. A generic `bottle` class only suggests a plastic bottle and remains uncertain; use Manual override to confirm the category. The model's actual classes and validation state appear in runtime status.

### Defect annotation policy
Annotate only visible, consistently defined external defects, such as a named crack or deformation. Prefix every class with product scope (`marker_`, `bottle_`/`plastic_bottle_`, or `charger_`/`phone_charger_`) so defects cannot cross-reject another category. Use a product-specific taxonomy, document borderline examples, include normal/negative images, and have annotations independently reviewed. A generic COCO model is not a defect detector. Images cannot establish charger electrical safety, internal damage, bottle leakage, structural strength, or unseen conditions.

### Training and validation
From the project root in PowerShell:
```powershell
python scripts/train_detector.py --data data/dataset.yaml --epochs 100 --imgsz 640
python scripts/validate_model.py --weights runs/defect-detection/weights/best.pt --data data/dataset.yaml
```
Validation reports held-out metrics and class compatibility; it does not certify a model or enable automatic acceptance. After a domain expert reviews the holdout results and documents the scope, set `DEFECT_MODEL_PATH`, exact comma-separated `DEFECT_CLASS_NAMES`, and `DEFECT_MODEL_VALIDATED=true` in the deployment environment. Set `MODEL_CONFIDENCE` to a threshold chosen during that validation. Enable `AUTOMATED_PASS_ENABLED=true` only if the documented, validated inspection procedure explicitly supports acceptance after all its required checks pass. Re-test the deployed artifact and keep the report with the model version.
""", elem_classes="vi-training-copy")
                    status_refresh.click(system_status, outputs=status_json)
                    demo.load(system_status, outputs=status_json)

    return demo


app = build_app()


if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if is_space else "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        max_file_size=f"{config.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)}mb",
        show_error=True,
    )