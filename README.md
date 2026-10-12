---
title: VisionInspect Pro
sdk: gradio
app_file: app.py
---

# VisionInspect Pro

A visual quality-inspection workspace for markers, plastic bottles, and phone chargers. It combines OpenCV image suitability checks, optional YOLO localization/defect inference, optional Gemini image observations, SQLite audit records, CSV/HTML reports, and an empty-state-aware dashboard. The shared backend is available through both the legacy Gradio entry point (`app.py`) and the Streamlit entry point (`streamlit_app.py`).

## Decision limitations

The repository contains no trained product-recognition or product-defect weights, private data, or sample inspection records. In the current clean install, automatic category detection is unavailable, so auto mode records **Unknown** and returns **REVIEW REQUIRED**; choose Manual override to run the available image-quality checks for a known category. A suitable image never returns PASS without a validated defect policy. Blurry, undersized, dark, overexposed, or heavily clipped images return **RECAPTURE REQUIRED**. A rejection is only supported by a configured, class-matched model explicitly enabled after domain-specific validation. An AI-only recommendation cannot pass a product. Photographs cannot establish electrical safety, internal charger condition, hidden leakage, strength, or unseen components.

Product recognition supports YOLO detection and classification outputs when real weights are installed. Exact model classes `marker`, `plastic_bottle`, and `phone_charger` can support auto-identification above `PRODUCT_CATEGORY_CONFIDENCE` (default `0.70`) only after `PRODUCT_MODEL_VALIDATED=true` is explicitly set following held-out evaluation. Generic `bottle`/`charger` classes remain uncertain and require manual confirmation; COCO localization is not defect inspection. Defect classes must be product-scoped with `marker_`, `bottle_`/`plastic_bottle_`, or `charger_`/`phone_charger_` prefixes. Model validation metrics do not certify an artifact. `DEFECT_MODEL_VALIDATED=true` is an explicit operator attestation after independent evaluation, not something the app sets automatically.

## Requirements

- Windows 10/11 or Linux; Python 3.11 recommended.
- Camera mode requires a browser with webcam permission. The browser captures the image; the server-side OpenCV camera probe is only a separate diagnostic. A hosted Space usually cannot access the visitor's local camera through server-side OpenCV, but browser webcam capture can work when the Space page is opened over HTTPS and permission is granted.
- Gemini analysis is optional and requires a Google AI API key. Local OpenCV inspection remains available without it.
- YOLO weights and inspection datasets are not included.

## Windows setup (PowerShell)

From the project directory:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run streamlit_app.py
```

To run the preserved Gradio interface instead, use `python app.py`. The default install omits optional Ultralytics/PyTorch so Streamlit Cloud can start without the large inference runtime. To train or run local YOLO models, install CPU PyTorch and the optional model package:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-models.txt
```

For CUDA, choose the matching PyTorch build from the official selector. Training may download initialization weights; no model weights are fetched automatically by the inspection app.

Run tests from the activated environment:

```powershell
python -m pytest -q
```

For a quick standard-library-only decision check after installing the app dependencies:

```powershell
python -m unittest tests.test_decision_engine
```

## Optional Gemini configuration

Set `AI_PROVIDER=gemini`, `GEMINI_MODEL` (optional), and `GEMINI_API_KEY` in `.env` for local use. Keep `.env` private; it is ignored by Git. In PowerShell, you can enter the key without placing it in command history:

```powershell
$env:AI_PROVIDER = "gemini"
$secret = Read-Host "Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
python app.py
```

For hosted deployment, configure `AI_PROVIDER`, `GEMINI_MODEL`, and `GEMINI_API_KEY` in the Space's Settings > Variables and secrets. The key is never displayed in the status tab. The app uses Google's official `google-genai` SDK and validates structured responses. Provider errors are reported; an unavailable provider is never represented as a completed AI analysis.

## Add and validate a defect model

1. Build a documented, product-specific dataset using the structure in [data/README.md](data/README.md) and the class list in `data/dataset.yaml`. Use images you have rights to use, carefully reviewed bounding boxes, representative normal examples, and item/session-separated validation and holdout splits.
2. Train from the project root:

   ```powershell
   python scripts/train_detector.py --data data/dataset.yaml --epochs 100 --imgsz 640
   ```

3. Evaluate on a held-out split:

   ```powershell
   python scripts/validate_model.py --weights runs/defect-detection/run/weights/best.pt --data data/dataset.yaml --split val
   ```

4. Prefix every defect class with its supported product scope (for example, `marker_barrel_crack`, `bottle_body_crack`, or `charger_casing_crack`). Have a qualified reviewer determine whether the intended classes, capture conditions, and operating thresholds are supported. Preserve the evaluation report. Copy approved weights to `models/defect_detector.pt` (or set `DEFECT_MODEL_PATH`), set `DEFECT_CLASS_NAMES` to the exact model class names, and only then set `DEFECT_MODEL_VALIDATED=true`. Configure `MODEL_CONFIDENCE` using evaluated operating characteristics. Restart the app and confirm the classes and enabled status in Model and System Status.

Weights are excluded from Git. `DEFECT_MODEL_VALIDATED=true` enables supported defect decisions only when the configured classes exactly match. Automated acceptance is disabled by default. Set `AUTOMATED_PASS_ENABLED=true` only when a documented, independently evaluated inspection policy establishes that this model, image-quality gates, confidence threshold, and capture procedure support acceptance for the stated product scope. The application still cannot assess hidden or non-visual properties.

### Product category weights

Keep category data separate under `data/product_dataset/`, with object boxes for the classes declared in `data/product_dataset.yaml`. Train and validate it independently:

```powershell
python scripts/train_detector.py --data data/product_dataset.yaml --project runs/product-recognition --epochs 100 --imgsz 640
python scripts/validate_model.py --weights runs/product-recognition/run/weights/best.pt --data data/product_dataset.yaml --split val
```

After reviewing a held-out set, copy the artifact to `models/product_detector.pt`, set `PRODUCT_CATEGORY_CONFIDENCE` using measured results, and only then set `PRODUCT_MODEL_VALIDATED=true`. Auto-detect remains manual-review-only until supported class names, threshold, and validation attestation are present.

## Data and deployment

SQLite records and original/annotated evidence are written to `artifacts/` by default. They are intentionally ignored by Git. Configure `VISIONINSPECT_DATABASE` and `VISIONINSPECT_IMAGE_DIR` to use persistent storage. Hugging Face Spaces local disk is ephemeral unless a persistent storage option is attached; arrange protected, durable storage before using the app for operational audit records. Do not upload sensitive inspection images or private datasets without an approved data-handling plan.

Deploy this repository as a Gradio Space using the metadata at the top of this README. The root `app.py` is the entry point and `requirements.txt` installs runtime packages. Set secrets through Space settings; optionally add validated weights through private storage or a controlled artifact download mechanism. Check camera access over HTTPS, persistent database/image storage, CPU/RAM and startup time, model licensing, API quota, and output retention before operational use. This repository is deployment-ready but has not been deployed or verified on Hugging Face Spaces.

### Streamlit Community Cloud

Push this project to a GitHub repository you control, create a Streamlit Community Cloud app from that repository, and set the app file path to `streamlit_app.py`. The root `requirements.txt` contains the Streamlit runtime and does not install PyTorch/Ultralytics by default. Add `GEMINI_API_KEY`, `AI_PROVIDER`, and any model/storage configuration through the deployment's Secrets panel; never commit `.env` or weights. Streamlit Community Cloud disk is ephemeral, so attach an approved persistent storage/database solution before operational use. Camera capture requires browser camera permission over HTTPS. Local deployment was verified here; publishing to Community Cloud requires a GitHub remote/account and was not performed in this workspace.

## Docker

```powershell
docker build -t visioninspect-pro .
docker run --rm -p 7860:7860 -v "${PWD}/artifacts:/app/artifacts" visioninspect-pro
```

The app is then available at `http://localhost:7860`. Supply secrets using your container platform's secret mechanism, not a committed Dockerfile or image layer.

## Implemented UI

- **Inspection**: webcam/upload image input, auto-detect or explicit manual category override, measured quality output, actual model detections when enabled, separately labeled AI observations, annotated evidence, timing, voice toggle, and persisted inspection metadata.
- **Dashboard**: counts, rates, categories, observed defect labels, trend, and recent records calculated from SQLite. Empty state starts at zero with no sample records.
- **History and Reports**: filters, CSV export, and individual HTML evidence report.
- **Model and System Status**: camera probe, weight paths/classes, provider setup, database, voice dependency, and known warnings.
- **Dataset and Model Training**: directory and annotation instructions with training/validation commands.

Voice feedback uses optional local text-to-speech on a daemon thread. It can fail if the host has no compatible speech engine or audio device; inspection processing does not wait for it.

## Tests and remaining limits

Uploads are limited to 25 MB and 40 megapixels; formats are JPEG, PNG, and WEBP. Run `python -m pytest -q`. Unit tests use temporary databases and explicitly named mocked-inference/provider tests; they do not validate real model accuracy or call Gemini. A clean checkout cannot validate category recognition or defect quality without real, reviewed weights and data. Hosted webcam behavior, third-party API availability, OS speech-engine behavior, and deployment persistence depend on the target environment and are not claimed as verified here.