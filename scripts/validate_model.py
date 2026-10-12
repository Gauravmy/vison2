"""Run a YOLO validation split and report metrics and class compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-validation.json"))
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Model weights not found: {args.weights}")
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error(f"Install requirements.txt to validate: {error}")
    dataset = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    expected = dataset.get("names", {})
    if isinstance(expected, list):
        expected = dict(enumerate(expected))
    model = YOLO(str(args.weights))
    actual = {int(key): str(value) for key, value in model.names.items()}
    options = {"data": str(args.data), "split": args.split, "imgsz": args.imgsz, "plots": True}
    if args.device is not None:
        options["device"] = args.device
    metrics = model.val(**options)
    report = {
        "weights": str(args.weights.resolve()),
        "dataset": str(args.data.resolve()),
        "split": args.split,
        "expected_classes": {str(key): str(value) for key, value in expected.items()},
        "model_classes": {str(key): value for key, value in actual.items()},
        "classes_match": {str(key): str(value) for key, value in expected.items()} == {str(key): value for key, value in actual.items()},
        "metrics": {str(key): float(value) for key, value in metrics.results_dict.items()},
        "certified_or_approved": False,
        "notice": "Metrics are evidence for expert review, not a certification or automatic acceptance policy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Validation report saved to {args.output}")


if __name__ == "__main__":
    main()