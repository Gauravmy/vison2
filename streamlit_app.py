"""Streamlit entry point for VisionInspect Pro."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config
from src import ai_vision as ai_vision_provider
from src.database import InspectionDatabase
from src.reporting import export_csv, generate_inspection_report


def _apply_streamlit_secrets() -> None:
    try:
        secrets = st.secrets.to_dict()
    except Exception:
        secrets = {}

    for name in ("AI_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL"):
        if name in secrets:
            setattr(config, name, str(secrets[name]))
    config.AI_PROVIDER = config.AI_PROVIDER.strip().lower()
    if config.GEMINI_API_KEY and not config.AI_PROVIDER:
        config.AI_PROVIDER = "gemini"
    ai_vision_provider.AI_PROVIDER = config.AI_PROVIDER
    ai_vision_provider.GEMINI_API_KEY = config.GEMINI_API_KEY
    ai_vision_provider.GEMINI_MODEL = config.GEMINI_MODEL

    for name in ("PRODUCT_MODEL_VALIDATED", "DEFECT_MODEL_VALIDATED", "AUTOMATED_PASS_ENABLED"):
        if name in secrets:
            setattr(config, name, str(secrets[name]).strip().lower() == "true")
    for secret_name, config_name in (
        ("PRODUCT_MODEL_PATH", "PRODUCT_MODEL_PATH"),
        ("DEFECT_MODEL_PATH", "DEFECT_MODEL_PATH"),
        ("VISIONINSPECT_DATABASE", "DATABASE_PATH"),
        ("VISIONINSPECT_IMAGE_DIR", "INSPECTION_IMAGE_DIR"),
    ):
        if secret_name in secrets:
            setattr(config, config_name, Path(str(secrets[secret_name])))
    if "DEFECT_CLASS_NAMES" in secrets:
        config.DEFECT_CLASS_NAMES = tuple(
            label.strip() for label in str(secrets["DEFECT_CLASS_NAMES"]).split(",") if label.strip()
        )
    for secret_name, config_name in (
        ("MODEL_CONFIDENCE", "MODEL_CONFIDENCE"),
        ("PRODUCT_CATEGORY_CONFIDENCE", "PRODUCT_CATEGORY_CONFIDENCE"),
    ):
        if secret_name in secrets:
            setattr(config, config_name, float(secrets[secret_name]))


_apply_streamlit_secrets()

import app as backend


st.set_page_config(
    page_title="VisionInspect Pro",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_system_status() -> dict[str, Any]:
    return backend.system_status()

st.markdown(
    """
    <style>
    :root {
        --vi-bg: #f6f6f2;
        --vi-surface: #ffffff;
        --vi-text: #202923;
        --vi-muted: #69736d;
        --vi-border: #e3e5df;
        --vi-accent: #285b49;
    }
    .stApp { background: var(--vi-bg); color: var(--vi-text); }
    .block-container { max-width: 1440px; padding-top: 1.1rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: var(--vi-text); letter-spacing: 0; }
    h1 { font-size: 1.45rem; margin-bottom: .1rem; }
    h2 { font-size: 1.05rem; }
    h3 { font-size: .95rem; }
    p, label, .stCaption { color: var(--vi-muted); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stTabs"] button { color: var(--vi-muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--vi-accent); }
    [data-testid="stMetric"] { padding: .5rem .7rem; border-right: 1px solid var(--vi-border); }
    [data-testid="stMetricLabel"] { color: var(--vi-muted); font-size: .75rem; }
    [data-testid="stMetricValue"] { color: var(--vi-text); font-size: 1.4rem; }
    div.stButton > button[kind="primary"] { background: var(--vi-accent); border-color: var(--vi-accent); }
    div.stButton > button { border-radius: 7px; }
    div[data-testid="stFileUploader"] section { background: var(--vi-surface); border-color: var(--vi-border); }
    .vi-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding: .1rem 0 .8rem; margin-bottom:.6rem; border-bottom:1px solid var(--vi-border); }
    .vi-brand { display:flex; align-items:center; gap:.7rem; }
    .vi-mark { display:block; width:28px; height:28px; border:1.5px solid var(--vi-accent); border-radius:7px; position:relative; }
    .vi-mark:before { content:""; position:absolute; inset:6px; border:1.5px solid var(--vi-accent); border-radius:2px; }
    .vi-mark:after { content:""; position:absolute; width:5px; height:5px; top:4px; right:4px; background:var(--vi-accent); border-radius:1px; }
    .vi-title { color:var(--vi-text); font-size:1.05rem; font-weight:650; line-height:1.2; }
    .vi-subtitle, .vi-status-line { color:var(--vi-muted); font-size:.76rem; }
    .vi-status-line { white-space:normal; text-align:right; }
    .vi-empty { min-height:195px; display:flex; align-items:center; justify-content:center; text-align:center; padding:1.4rem; border:1px dashed var(--vi-border); border-radius:8px; background:rgba(255,255,255,.65); }
    .vi-empty strong { display:block; color:var(--vi-text); margin-bottom:.3rem; }
    .vi-empty span { color:var(--vi-muted); font-size:.82rem; }
    @media (max-width: 760px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .vi-header { align-items:flex-start; flex-direction:column; }
        .vi-status-line { text-align:left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _system_line(status: dict[str, Any]) -> str:
    products = status["local_models"]["product_localization"]
    defects = status["local_models"]["defect_inspection"]
    if products["available"] and products.get("validation_enabled") and defects["available"]:
        return "Validated product and defect models loaded"
    if products["available"] and not products.get("validation_enabled"):
        return "Product model loaded · category validation required"
    if defects["available"]:
        return "Defect model loaded · category confirmation may be required"
    return "Local image checks available · product and defect weights not configured"


def _json_or_message(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {"message": value}


def _image_bytes(upload: Any) -> bytes:
    raw = upload.getvalue()
    if len(raw) > config.MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError(f"Image exceeds the {config.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    return raw


def _quality_summary(quality: dict[str, Any]) -> None:
    if quality.get("suitable"):
        st.success("Image quality checks passed for analysis.")
    else:
        st.warning("Image needs to be recaptured.")
    values = [
        ("Dimensions", f"{quality.get('width', 0)} × {quality.get('height', 0)} px"),
        ("Blur variance", f"{quality.get('blur_variance', 0):.1f}"),
        ("Mean brightness", f"{quality.get('mean_brightness', 0):.1f}"),
        ("Clipped pixels", f"{quality.get('clipped_pixel_fraction', 0):.1%}"),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    concerns = quality.get("concerns", [])
    if concerns:
        for concern in concerns:
            st.caption(f"• {concern}")


def _render_inspection(result: tuple[Any, ...]) -> None:
    original, annotated, outcome, explanation, detections, ai_text, method, quality, metadata = result
    st.markdown(backend.render_status_card(outcome, explanation), unsafe_allow_html=True)
    with st.container():
        image_columns = st.columns(2)
        image_columns[0].image(original, caption="Original image", width="stretch")
        image_columns[1].image(annotated, caption="Annotated evidence", width="stretch")
    with st.expander("Evidence and measurements", expanded=True):
        evidence_tabs = st.tabs(["Detections", "AI observations", "Image quality", "Method and audit"])
        with evidence_tabs[0]:
            if detections.empty:
                st.info("No supported model detections were produced.")
            else:
                st.dataframe(detections, width="stretch", hide_index=True)
        with evidence_tabs[1]:
            ai_result = _json_or_message(ai_text)
            if ai_result == {"message": "Not requested."}:
                st.caption("AI-assisted analysis was not requested.")
            else:
                st.json(ai_result)
        with evidence_tabs[2]:
            _quality_summary(quality)
        with evidence_tabs[3]:
            st.json({"inspection method": method, "audit": metadata})


def _render_inspect() -> None:
    left, right = st.columns([0.92, 1.55], gap="large")
    with left:
        st.subheader("Capture")
        st.caption("Capture a product or upload a JPG, PNG, or WEBP image.")
        source = st.radio("Image source", ["Upload", "Camera"], horizontal=True, key="vi_source")
        image = None
        if source == "Upload":
            upload = st.file_uploader(
                "Product photo", type=["jpg", "jpeg", "png", "webp"],
                max_upload_size=config.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024),
                key="vi_upload",
            )
            if upload is not None:
                try:
                    image = backend.load_image(_image_bytes(upload))
                    st.image(image, caption="Image uploaded. Ready to inspect.", width="stretch")
                except ValueError as error:
                    st.error(f"Image could not be loaded: {error}")
        else:
            camera = st.camera_input("Camera frame", key="vi_camera")
            if camera is None:
                st.caption("Camera permission is requested by your browser when you start capture.")
            else:
                try:
                    image = backend.load_image(_image_bytes(camera))
                    st.image(image, caption="Image captured. Ready to inspect.", width="stretch")
                except ValueError as error:
                    st.error(f"Camera image could not be decoded: {error}")

        category_mode = st.radio(
            "Product category", ["Auto-detect", "Manual override"], horizontal=True,
            key="vi_category_mode",
        )
        manual_product = None
        if category_mode == "Manual override":
            manual_product = st.selectbox("Manual category", backend.PRODUCTS, key="vi_manual_product")
            st.caption("The selected category is recorded as a manual override.")
        else:
            st.caption("Uncertain model match? The inspection stays in review.")

        mode = st.selectbox("Inspection method", backend.MODES, key="vi_mode")
        voice = st.checkbox("Voice feedback", value=False, key="vi_voice")
        action_columns = st.columns([2, 1])
        inspect = action_columns[0].button("Inspect product", type="primary", width="stretch")
        clear = action_columns[1].button("Clear", width="stretch")
        if clear:
            st.session_state.pop("vi_last_result", None)
            st.session_state.pop("vi_last_error", None)
            st.rerun()
        if inspect:
            if image is None:
                st.error("Capture an image or upload a product photo before inspection.")
            else:
                with st.spinner("Inspecting image and recording evidence..."):
                    try:
                        st.session_state["vi_last_result"] = backend.run_inspection(
                            image, manual_product, mode, voice,
                        )
                        st.session_state.pop("vi_last_error", None)
                    except Exception as error:
                        st.session_state["vi_last_error"] = str(error)
                        st.session_state.pop("vi_last_result", None)

    with right:
        st.subheader("Inspection result")
        if st.session_state.get("vi_last_error"):
            st.error(st.session_state["vi_last_error"])
        result = st.session_state.get("vi_last_result")
        if result is None:
            st.markdown(
                '<div class="vi-empty"><div><strong>Ready for inspection</strong>'
                '<span>Capture an image or upload a product photo to begin.</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            _render_inspection(result)


def _render_dashboard() -> None:
    st.subheader("Inspection activity")
    filter_columns = st.columns([1, 1, 1, 1, 1])
    product = filter_columns[0].selectbox("Product", backend.PRODUCT_FILTER_OPTIONS, key="vi_dash_product")
    outcome = filter_columns[1].selectbox("Outcome", ["All", *backend.OUTCOMES], key="vi_dash_outcome")
    from_text = filter_columns[2].text_input("From date", placeholder="YYYY-MM-DD", key="vi_dash_from")
    to_text = filter_columns[3].text_input("To date", placeholder="YYYY-MM-DD", key="vi_dash_to")
    apply = filter_columns[4].button("Apply", type="primary", width="stretch")
    if apply:
        st.session_state["vi_dashboard_filters"] = (product, outcome, from_text, to_text)
    selected_filters = st.session_state.get("vi_dashboard_filters", ("All", "All", "", ""))
    product, outcome, from_text, to_text = selected_filters
    try:
        if from_text:
            date.fromisoformat(from_text)
        if to_text:
            date.fromisoformat(to_text)
        data = backend.DATABASE.dashboard_data(
            product=product, outcome=outcome, date_from=from_text, date_to=to_text,
        )
    except ValueError:
        st.error("Enter dates using YYYY-MM-DD.")
        return

    counts = data["counts"]
    metric_columns = st.columns(5)
    metric_values = [
        ("Total inspections", data["total"]),
        ("Accepted", counts[backend.Outcome.PASS.value]),
        ("Rejected", counts[backend.Outcome.FAIL.value]),
        ("Manual review", counts[backend.Outcome.REVIEW.value]),
        ("Recapture", counts[backend.Outcome.RECAPTURE.value]),
    ]
    for column, (label, value) in zip(metric_columns, metric_values):
        column.metric(label, value)
    if outcome != "All":
        st.caption("Acceptance rate is unavailable while filtering by outcome.")
    elif data["total"]:
        st.caption(f"Acceptance rate: {data['accepted_rate']:.1%} of {data['total']} filtered inspections.")
    else:
        st.caption("Acceptance rate is unavailable until an inspection is recorded.")

    if data["total"] == 0:
        st.info("No inspections match these filters. Complete an inspection to populate this dashboard.")
    else:
        _, _, trend_figure, category_figure = backend._dashboard_frames(data)
        charts = st.columns(2)
        charts[0].plotly_chart(trend_figure, width="stretch")
        charts[1].plotly_chart(category_figure, width="stretch")
        tables = st.columns(2)
        tables[0].dataframe(
            pd.DataFrame(data["common_observations"], columns=["observed class", "count"]),
            width="stretch", hide_index=True,
        )
        recent = [
            {key: row.get(key) for key in ("inspection_id", "timestamp", "product", "provider", "outcome")}
            for row in data["recent"]
        ]
        tables[1].dataframe(pd.DataFrame(recent), width="stretch", hide_index=True)


def _render_history() -> None:
    st.subheader("Inspection history")
    with st.form("vi_history_filters"):
        columns = st.columns(5)
        search = columns[0].text_input("Inspection ID", placeholder="Search by ID")
        product = columns[1].selectbox("Product", backend.PRODUCT_FILTER_OPTIONS)
        outcome = columns[2].selectbox("Outcome", ["All", *backend.OUTCOMES])
        date_from = columns[3].text_input("From date", placeholder="YYYY-MM-DD")
        date_to = columns[4].text_input("To date", placeholder="YYYY-MM-DD")
        submitted = st.form_submit_button("Search history", type="primary")
    if submitted:
        st.session_state["vi_history_filters"] = (search, product, outcome, date_from, date_to)
    search, product, outcome, date_from, date_to = st.session_state.get(
        "vi_history_filters", ("", "All", "All", "", ""),
    )
    try:
        records = backend.DATABASE.list_inspections(
            search=search, product=product, outcome=outcome,
            date_from=date_from, date_to=date_to,
        )
    except ValueError:
        st.error("Enter dates using YYYY-MM-DD.")
        return

    st.caption(f"{len(records)} inspection(s)")
    history_fields = ["inspection_id", "timestamp", "product", "outcome", "provider", "explanation"]
    history_table = pd.DataFrame(
        [{key: record.get(key) for key in history_fields} for record in records],
        columns=history_fields,
    )
    st.dataframe(history_table, width="stretch", hide_index=True)

    export_columns = st.columns([1, 3])
    if records:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = export_csv(records, Path(temporary_directory) / "inspection-history.csv")
            csv_bytes = Path(csv_path).read_bytes()
        export_columns[0].download_button(
            "Export CSV", data=csv_bytes, file_name="inspection-history.csv",
            mime="text/csv", width="stretch",
        )
    else:
        export_columns[0].button("Export CSV", disabled=True, width="stretch")

    with st.expander("Open saved inspection evidence"):
        inspection_id = st.text_input("Inspection ID", key="vi_record_id")
        if st.button("Open record", key="vi_open_record"):
            record = backend.DATABASE.get_inspection(inspection_id.strip())
            if record is None:
                st.error("No inspection found for that ID.")
            else:
                st.markdown(backend.render_status_card(record["outcome"], record["explanation"]), unsafe_allow_html=True)
                evidence_columns = st.columns(2)
                for column, label, path_key in (
                    (evidence_columns[0], "Original", "original_image_path"),
                    (evidence_columns[1], "Annotated", "annotated_image_path"),
                ):
                    image_path = record.get(path_key)
                    if image_path and Path(image_path).is_file():
                        column.image(image_path, caption=label, width="stretch")
                st.json({
                    "inspection ID": record["inspection_id"],
                    "timestamp": record["timestamp"],
                    "product": record["product"],
                    "method": record["provider"],
                    "quality": record["quality"],
                    "detections": record["detections"],
                    "AI-assisted observations": record["ai_observations"],
                })
                report_path = generate_inspection_report(record, config.ROOT_DIR / "artifacts" / "reports")
                st.download_button(
                    "Download individual report", data=Path(report_path).read_bytes(),
                    file_name=Path(report_path).name, mime="text/html",
                )


def _render_models() -> None:
    st.subheader("Models and system settings")
    st.caption("Readiness reflects installed models and configured providers, not configuration fields alone.")
    status = _cached_system_status()
    st.info(
        "No product-recognition or defect weights are included by default. Auto-detect reports Unknown; "
        "manual category selection does not establish defect-free quality."
    )
    with st.expander("Runtime status", expanded=True):
        st.json(status)
    with st.expander("Dataset and model training"):
        st.markdown(
            """
**Product category model**

Use `data/product_dataset/` and `data/product_dataset.yaml`. Train and validate separately:

```powershell
python scripts/train_detector.py --data data/product_dataset.yaml --project runs/product-recognition --epochs 100 --imgsz 640
python scripts/validate_model.py --weights runs/product-recognition/run/weights/best.pt --data data/product_dataset.yaml --split val
```

After held-out review, register the weights at `models/product_detector.pt`, choose `PRODUCT_CATEGORY_CONFIDENCE`, and set `PRODUCT_MODEL_VALIDATED=true`. Exact labels `marker`, `plastic_bottle`, and `phone_charger` are required for confident auto-identification. Generic `bottle`/`charger` labels stay uncertain.

**Product defect model**

Use `data/dataset.yaml`. Defect labels must be product-scoped (`marker_`, `bottle_`/`plastic_bottle_`, or `charger_`/`phone_charger_`). Then run:

```powershell
python scripts/train_detector.py --data data/dataset.yaml --epochs 100 --imgsz 640
python scripts/validate_model.py --weights runs/defect-detection/run/weights/best.pt --data data/dataset.yaml --split val
```

Only after documented domain review should you set `DEFECT_CLASS_NAMES` and `DEFECT_MODEL_VALIDATED=true`. Automated acceptance additionally requires a separate validated policy and remains disabled by default.

Images cannot establish electrical safety, internal damage, hidden leakage, material strength, or unseen conditions.
""",
        )


def main() -> None:
    status = _cached_system_status()
    st.markdown(
        f'<header class="vi-header"><div class="vi-brand"><span class="vi-mark"></span>'
        f'<div><div class="vi-title">VisionInspect Pro</div>'
        f'<div class="vi-subtitle">Quality inspection workspace · v{config.APP_VERSION}</div></div></div>'
        f'<div class="vi-status-line">{_system_line(status)}</div></header>',
        unsafe_allow_html=True,
    )
    inspect_tab, dashboard_tab, history_tab, models_tab = st.tabs(
        ["Inspect", "Dashboard", "History", "Models & settings"]
    )
    with inspect_tab:
        _render_inspect()
    with dashboard_tab:
        _render_dashboard()
    with history_tab:
        _render_history()
    with models_tab:
        _render_models()


if __name__ == "__main__":
    main()
