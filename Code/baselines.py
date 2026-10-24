from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate import score_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trivial localization baselines on a masked split.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/manifest.csv")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/evaluation/baselines.csv")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    frame = pd.read_csv(args.manifest)
    frame = frame[(frame.split == args.split) & frame.mask_path.fillna("").astype(str).str.len().gt(0)]
    records = []
    for row in frame.itertuples(index=False):
        truth_gray = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if truth_gray is None:
            continue
        truth = cv2.resize(truth_gray, (224, 224), interpolation=cv2.INTER_NEAREST) > 0
        empty = np.zeros_like(truth)
        yy, xx = np.ogrid[:224, :224]
        central = ((xx - 112) / 45) ** 2 + ((yy - 112) / 70) ** 2 <= 1
        random_blob = rng.random((224, 224)) < 0.10
        for name, pred in (("empty", empty), ("central_ellipse", central), ("random_10pct", random_blob)):
            records.append({"source": row.source, "relative_path": row.relative_path, "split": args.split,
                            "baseline": name, **score_mask(pred, truth)})
    if not records:
        parser.error("No readable expert-mask pairs found. Baseline results cannot be computed.")
    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby("baseline")[["dice", "iou", "precision", "recall"]].mean().to_string())
    print(f"Saved per-image baseline results to {args.output}")


if __name__ == "__main__":
    main()

