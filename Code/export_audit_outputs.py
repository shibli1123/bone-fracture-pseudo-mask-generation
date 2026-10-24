"""Export a reproducible dataset inventory summary and figure (not model results)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "data" / "manifest.csv"
OUT = ROOT / "docs" / "outputs"


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
    summary = {
        "images_indexed": len(records),
        "unique_exact_sha256": len(hashes),
        "exact_duplicate_groups_crossing_splits": sum(len({x[0] for x in values}) > 1 for values in groups.values()),
        "duplicate_label_conflict_rows": conflicts,
        "candidate_mask_pairs": sum(bool(row["mask_path"]) for row in records),
        "deep_learning_models_run": [],
        "model_run_status": "No model was run: PyTorch/torchvision are unavailable in the active runtime; no expert masks are present.",
    }
    (OUT / "run_and_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "model_run_status.csv").write_text(
        "model,status,reason\n"
        'ResNet-50 classifier,not run,"PyTorch/torchvision unavailable; no trained checkpoint or predictions"\n'
        'DenseNet-121 classifier,not run,"PyTorch/torchvision unavailable; no trained checkpoint or predictions"\n'
        'Compact U-Net segmenter,not run,"PyTorch/torchvision unavailable and no expert pixel masks"\n',
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
