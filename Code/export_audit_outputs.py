"""Export a reproducible dataset inventory summary and figure (not model results)."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "data" / "manifest.csv"
OUT = ROOT / "docs" / "outputs"
EVALUATION = ROOT / "outputs" / "evaluation"
CLASSIFIERS = ROOT / "outputs" / "classifiers"
SEGMENTERS = ROOT / "outputs" / "segmenters"
SHORTCUT_CHECK = ROOT / "outputs" / "data" / "shortcut_check.csv"


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Run Code/prepare_data.py first to create outputs/data/manifest.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    records = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline="")))
    counts = Counter((row["source"], row["split"], row["label"]) for row in records)
    unique = defaultdict(set)
    for row in records:
        unique[(row["source"], row["split"], row["label"])].add(row["sha256"])

    count_path = OUT / "dataset_split_counts.csv"
    with count_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "split", "label", "image_count", "unique_sha256_count"])
        for key in sorted(counts):
            writer.writerow([*key, counts[key], len(unique[key])])

    hashes = {r["sha256"] for r in records}
    groups = defaultdict(set)
    conflicts = 0
    for row in records:
        groups[row["sha256"]].add((row["split"], row["label"]))
        conflicts += int(row["label_conflict"].lower() == "true")
    test_summary_path = EVALUATION / "resnet50_224_s0_test_summary.json"
    if not test_summary_path.exists():
        test_summary_path = OUT / "classifier_test_summary.json"
    test_summary = json.loads(test_summary_path.read_text(encoding="utf-8")) if test_summary_path.exists() else None
    run_tag = "resnet50_224_s0"
    run_dir = CLASSIFIERS / run_tag
    if test_summary is not None:
        published_summary = OUT / "classifier_test_summary.json"
        if test_summary_path.resolve() != published_summary.resolve():
            shutil.copy2(test_summary_path, published_summary)
        history_path = run_dir / "history.csv"
        if history_path.exists():
            shutil.copy2(history_path, OUT / "classifier_training_history.csv")
        classifier_status = "trained and evaluated on held-out test split"
        classifier_reason = f"seed {test_summary['training_seed']}; epoch {test_summary['checkpoint_epoch']}; see classifier_test_summary.json"
        models_run = [{
            "model": "ResNet-50",
            "training_seed": test_summary["training_seed"],
            "checkpoint_epoch": test_summary["checkpoint_epoch"],
            "test_metrics": test_summary["metrics"],
        }]
        run_status = "One ResNet-50 classifier was trained and evaluated on the held-out test split."
    elif (run_dir / "best.pt").exists():
        classifier_status = "trained; test evaluation not found"
        classifier_reason = "checkpoint exists, but no held-out test summary is available"
        models_run = [{"model": "ResNet-50", "status": "trained; not evaluated"}]
        run_status = "A ResNet-50 checkpoint exists, but test evaluation has not been recorded."
    else:
        classifier_status = "not run"
        classifier_reason = "no trained checkpoint or held-out test summary found"
        models_run = []
        run_status = "No trained classifier or held-out test summary was found."

    segmenter_run = SEGMENTERS / "unet_128_s42"
    segmenter_history = segmenter_run / "history.csv"
    segmenter_checkpoint = segmenter_run / "best.pt"
    segmenter_manifest = ROOT / "outputs" / "data" / "segmentation_train_manifest.csv"
    segmenter_artifacts = (segmenter_history, segmenter_checkpoint, segmenter_manifest)
    if any(path.exists() for path in segmenter_artifacts):
        missing = [str(path) for path in segmenter_artifacts if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete U-Net run artifacts: {', '.join(missing)}")
        import pandas as pd
        history = pd.read_csv(segmenter_history)
        best = history.loc[history["val_dice_against_training_labels"].idxmax()]
        split_counts = pd.read_csv(segmenter_manifest)["split"].value_counts().to_dict()
        segmenter_status = "trained on train-only CAM pseudo-masks; no expert-mask evaluation"
        segmenter_reason = (
            f"seed 42; epoch {int(best['epoch'])}; "
            f"validation Dice against pseudo-labels {best['val_dice_against_training_labels']:.4f}"
        )
        models_run.append({
            "model": "Compact U-Net",
            "training_seed": 42,
            "checkpoint_epoch": int(best["epoch"]),
            "training_images": int(split_counts.get("train", 0)),
            "validation_images": int(split_counts.get("val", 0)),
            "validation_metrics_against_pseudo_labels": {
                "dice": float(best["val_dice_against_training_labels"]),
                "iou": float(best["val_iou_against_training_labels"]),
            },
            "test_evaluated": False,
        })
        run_status += " A compact U-Net was trained using train-only CAM pseudo-masks; expert-mask evaluation was not possible."
    else:
        segmenter_status = "not run"
        segmenter_reason = "no trained U-Net run artifacts found"
        run_status += " No segmentation model run artifacts were found."
    summary = {
        "images_indexed": len(records),
        "unique_exact_sha256": len(hashes),
        "exact_duplicate_groups_crossing_splits": sum(len({x[0] for x in values}) > 1 for values in groups.values()),
        "duplicate_label_conflict_rows": conflicts,
        "candidate_mask_pairs": sum(bool(row["mask_path"]) for row in records),
        "deep_learning_models_run": models_run,
        "model_run_status": run_status,
    }
    if SHORTCUT_CHECK.exists():
        shutil.copy2(SHORTCUT_CHECK, OUT / "shortcut_check.csv")
    (OUT / "run_and_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "model_run_status.csv").write_text(
        "model,status,reason\n"
        f'ResNet-50 classifier,{classifier_status},"{classifier_reason}"\n'
        'DenseNet-121 classifier,not run,"not included in the single-configuration experiment"\n'
        f'Compact U-Net segmenter,{segmenter_status},"{segmenter_reason}"\n',
        encoding="utf-8",
    )

    # Horizontal grouped bars show the observed split/class inventory only.
    width, height = 1500, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((100, 42), "Bone fracture dataset inventory", fill="#172B4D", font=font(42, True))
    draw.text((100, 100), "Indexed image files by existing split and class (not model performance)",
              fill="#52657A", font=font(24))
    splits = ["train", "val", "test"]
    classes = [("fractured", "#2F6B9A"), ("not_fractured", "#E69F24")]
    x0, x1 = 310, 1300
    top, bottom = 205, 665
    max_value = max(counts.values())
    for tick in range(0, 701, 100):
        x = x0 + (x1 - x0) * tick / max_value
        draw.line((x, top, x, bottom), fill="#DDE4EC", width=2)
        draw.text((x - 15, bottom + 18), str(tick), fill="#52657A", font=font(19))
    for i, split in enumerate(splits):
        center = top + 75 + i * 145
        draw.text((100, center - 14), split.capitalize(), fill="#172B4D", font=font(28, True))
        for j, (label, color) in enumerate(classes):
            value = counts[("bone-fracture-dataset", split, label)]
            y = center + j * 42
            bar_end = x0 + (x1 - x0) * value / max_value
            draw.rounded_rectangle((x0, y, bar_end, y + 28), radius=5, fill=color)
            draw.text((bar_end + 12, y + 1), f"{value:,}", fill="#172B4D", font=font(20, True))
    draw.text((x0, 725), "Image count", fill="#52657A", font=font(20))
    for i, (label, color) in enumerate(classes):
        x = 900 + i * 280
        draw.rounded_rectangle((x, 705, x + 24, 729), radius=4, fill=color)
        draw.text((x + 35, 703), label.replace("_", " "), fill="#172B4D", font=font(18))
    image.save(OUT / "dataset_distribution.png", dpi=(180, 180))
    print(f"Wrote audit-only outputs to {OUT}")


if __name__ == "__main__":
    main()
