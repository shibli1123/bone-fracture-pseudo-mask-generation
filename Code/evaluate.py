from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wsss.masks import clean_mask, threshold_cam


def score_mask(pred: np.ndarray, truth: np.ndarray, cam: np.ndarray | None = None) -> dict[str, float]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    union = tp + fp + fn
    both_empty = union == 0
    dice = 1.0 if both_empty else 2 * tp / max(2 * tp + fp + fn, 1)
    iou = 1.0 if both_empty else tp / union
    precision = 1.0 if both_empty else tp / max(tp + fp, 1)
    recall = 1.0 if both_empty else tp / max(tp + fn, 1)
    result = {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
    if cam is not None:
        y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        result["pointing_game"] = float(truth[y, x])
    return result


def bootstrap_ci(values: np.ndarray, seed: int, groups=None, samples: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if groups is not None:
        grouped = pd.DataFrame({"value": values, "group": groups}).groupby("group").value.mean()
        values = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def evaluate_at(frame: pd.DataFrame, threshold: float, mode: str, kernel: int, seed: int) -> pd.DataFrame:
    results = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.mask_path) or not str(row.mask_path) or not Path(str(row.mask_path)).is_file():
            continue
        cam = np.load(row.cam_path).astype(np.float32)
        truth_gray = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if truth_gray is None:
            continue
        truth = cv2.resize(truth_gray, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        pred = clean_mask(threshold_cam(cam, mode, threshold), kernel)
        metrics = score_mask(pred, truth, cam)
        results.append({"source": row.source, "relative_path": row.relative_path, "split": row.split,
                        "group_id": getattr(row, "group_id", row.sha256),
                        "threshold": threshold, "threshold_mode": mode, **metrics})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAM localization against expert masks.")
    parser.add_argument("--index", type=Path, required=True, help="Index CSV from cam_generate.py")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--threshold", type=float, help="Required for test; select from validation first.")
    parser.add_argument("--sweep", action="store_true", help="Validation-only global threshold sweep.")
    parser.add_argument("--mode", choices=("global", "otsu", "percentile"), default="global")
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/evaluation")
    args = parser.parse_args()
    if args.split == "test" and (args.threshold is None or args.sweep):
        parser.error("Test evaluation requires one fixed --threshold and does not permit --sweep.")
    if args.split == "val" and args.sweep and args.mode != "global":
        parser.error("Threshold sweep is currently defined for global CAM thresholds only.")
    frame = pd.read_csv(args.index)
    frame = frame[frame.split == args.split]
    if frame.empty:
        parser.error(f"No rows for split {args.split}.")
    if not frame.mask_path.fillna("").astype(str).str.len().gt(0).any():
        parser.error("No expert/candidate masks are paired to this split. Check data pairing before evaluation.")
    thresholds = np.round(np.arange(0.1, 1.0, 0.05), 2) if args.sweep else [args.threshold if args.threshold is not None else 0.4]
    all_results = []
    for threshold in thresholds:
        result = evaluate_at(frame, float(threshold), args.mode, args.kernel, args.seed)
        if result.empty:
            parser.error("No readable image-mask pairs were evaluated.")
        result["seed"] = args.seed
        result["cam_tag"] = args.index.parent.name
        all_results.append(result)
    metrics = pd.concat(all_results, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{args.mode}_{'sweep' if args.sweep else str(args.threshold)}_s{args.seed}"
    metrics_path = args.output / f"{tag}_per_image.csv"
    metrics.to_csv(metrics_path, index=False)
    summary_rows = []
    for keys, group in metrics.groupby(["threshold", "source"], dropna=False):
        threshold, source = keys
        item = {"cam_tag": args.index.parent.name, "seed": args.seed,
                "split": args.split, "threshold": threshold, "source": source, "n": len(group)}
        for metric in ("dice", "iou", "precision", "recall", "pointing_game"):
            values = group[metric].to_numpy(float)
            lo, hi = bootstrap_ci(values, args.seed, groups=group.group_id.to_numpy())
            item.update({f"{metric}_mean": values.mean(), f"{metric}_std": values.std(ddof=1) if len(values) > 1 else 0.0,
                         f"{metric}_ci95_low": lo, f"{metric}_ci95_high": hi})
        summary_rows.append(item)
    summary_path = args.output / f"{tag}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Saved {metrics_path} and {summary_path}")
    if args.split == "val" and args.sweep:
        dice_by_threshold = metrics.groupby("threshold").dice.mean()
        print(f"Validation-selected global threshold: {dice_by_threshold.idxmax():.2f}")


if __name__ == "__main__":
    main()
