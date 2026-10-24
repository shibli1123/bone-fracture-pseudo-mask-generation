from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from wsss.masks import clean_mask, threshold_cam


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold CAM files into binary pseudo-masks.")
    parser.add_argument("--index", type=Path, required=True, help="Index CSV produced by cam_generate.py")
    parser.add_argument("--threshold", type=float, required=True, help="Choose this on validation only before test evaluation.")
    parser.add_argument("--mode", choices=("global", "otsu", "percentile"), default="global")
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/pseudo_masks")
    args = parser.parse_args()
    if args.mode == "global" and not 0 <= args.threshold <= 1:
        parser.error("Global threshold must be between 0 and 1.")
    rows = []
    with args.index.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            cam = np.load(row["cam_path"])
            binary = clean_mask(threshold_cam(cam, args.mode, args.threshold), args.kernel)
            relative = Path(row["relative_path"])
            dest = args.output / row["split"] / relative.parent / f"{relative.stem}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary * 255).save(dest)
            row["mask_path"] = str(dest.resolve())
            rows.append(row)
    index_path = args.output / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8-sig") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} masks under {args.output}; index: {index_path}")


if __name__ == "__main__":
    main()
