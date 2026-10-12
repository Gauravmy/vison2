# Defect Dataset

Keep private or licensed data out of Git. The repository contains no sample inspection images and no pretrained defect weights.

Expected layout:

```text
data/dataset/
  images/train/
  images/val/
  images/test/       # optional, held out until final evaluation
  labels/train/
  labels/val/
  labels/test/       # optional
```

Use one YOLO-format `.txt` label file per image. Each line is `class_id x_center y_center width height`, with normalized coordinates. Define stable, visually verifiable external-defect classes in `dataset.yaml`; keep normal images without boxes in the dataset. Split by physical unit or capture session rather than near-identical frames. Review annotation quality and class balance, document capture conditions, and maintain a separate final holdout set. See the app's Dataset and Model Training tab before using any model for decisions.

Names for defect classes must carry an explicit product prefix so the runtime can prevent cross-category defects (for example `marker_barrel_crack`, `bottle_body_crack`, or `charger_casing_crack`). Product-category classifier/detector weights are a separate artifact at `models/product_detector.pt`; supported exact category classes are documented in `models/README.md`.

For product-category detection, use a separate `data/product_dataset/` tree with the same images/labels layout and classes `marker`, `plastic_bottle`, and `phone_charger` from `product_dataset.yaml`. Box the whole product and split by physical item/session. Keep category data separate from defect data so category labels are never mistaken for defects.