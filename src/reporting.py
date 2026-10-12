"""CSV history export and individual HTML inspection reports."""

from __future__ import annotations

import csv
import base64
import html
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


CSV_FIELDS = [
    "inspection_id", "timestamp", "product", "mode", "provider", "outcome", "explanation",
    "detections", "ai_observations", "quality", "model_version", "processing_ms",
]


def export_csv(records: list[dict[str, Any]], destination: str | Path) -> str:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({
                **record,
                "detections": json.dumps(record.get("detections", []), ensure_ascii=True),
                "ai_observations": json.dumps(record.get("ai_observations", []), ensure_ascii=True),
                "quality": json.dumps(record.get("quality", {}), ensure_ascii=True),
            })
    return str(path)


def _embedded_image(record: dict[str, Any], field: str, label: str) -> str:
    raw_path = record.get(field)
    if not raw_path:
        return f"<p>{html.escape(label)} evidence was not stored.</p>"
    try:
        image_path = Path(raw_path).resolve(strict=True)
        image_root = Path(config.INSPECTION_IMAGE_DIR).resolve()
        if not image_path.is_relative_to(image_root) or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return f"<p>{html.escape(label)} evidence is unavailable.</p>"
        if image_path.stat().st_size > 15 * 1024 * 1024:
            return f"<p>{html.escape(label)} evidence exceeds the report size limit.</p>"
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError:
        return f"<p>{html.escape(label)} evidence is unavailable.</p>"
    return f'<figure><figcaption>{html.escape(label)}</figcaption><img alt="{html.escape(label)}" src="data:{mime_type};base64,{encoded}"></figure>'


def generate_inspection_report(record: dict[str, Any], destination_dir: str | Path) -> str:
    folder = Path(destination_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"inspection-{record['inspection_id']}.html"
    fields = ["timestamp", "product", "mode", "provider", "outcome", "explanation", "model_version", "processing_ms"]
    rows = "".join(
        f"<tr><th>{html.escape(field.replace('_', ' ').title())}</th><td>{html.escape(str(record.get(field) or ''))}</td></tr>"
        for field in fields
    )
    evidence = html.escape(json.dumps({
        "quality": record.get("quality", {}),
        "detections": record.get("detections", []),
        "AI-assisted observations": record.get("ai_observations", []),
    }, indent=2, ensure_ascii=True))
    images = "".join((
        _embedded_image(record, "original_image_path", "Original image"),
        _embedded_image(record, "annotated_image_path", "Annotated image"),
    ))
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Inspection {html.escape(record['inspection_id'])}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#202b2a}}h1{{font-size:24px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #cbd3d0;padding:10px}}th{{width:190px}}pre{{white-space:pre-wrap;background:#f2f5f3;padding:16px;border-radius:4px}}figure{{margin:20px 0}}img{{max-width:100%;height:auto;border:1px solid #cbd3d0}}</style>
<h1>Inspection Report</h1><p>ID: {html.escape(record['inspection_id'])}</p>
<table>{rows}</table><h2>Image evidence</h2>{images}<h2>Recorded measurements and observations</h2><pre>{evidence}</pre>
<p>Visual inspection cannot establish hidden or internal product conditions. AI observations are not verified physical measurements.</p>
</html>"""
    path.write_text(document, encoding="utf-8")
    return str(path)


def timestamped_csv_path(directory: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(directory) / f"inspection-history-{stamp}.csv"