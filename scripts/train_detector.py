"""Train a category-specific YOLO detector; training does not certify it."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt", help="Pretrained initialization or local model path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--project", type=Path, default=Path("runs/defect-detection"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")
    if args.epochs < 1 or args.imgsz < 32:
        parser.error("epochs must be positive and imgsz must be at least 32")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error(f"Install requirements.txt to train: {error}")
    model = YOLO(args.weights)
    options = {
        "data": str(args.data), "epochs": args.epochs, "imgsz": args.imgsz,
        "project": str(args.project), "name": "run", "exist_ok": True,
    }
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)
    print(f"Training output: {args.project / 'run' / 'weights' / 'best.pt'}")
    print("This artifact is not approved for production decisions until independent holdout evaluation is reviewed.")


if __name__ == "__main__":
    main()