from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from train_classifier import ManifestDataset
from wsss.models import CLASS_NAMES, build_classifier


def classification_metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_fractured": float(precision_score(truth, predicted, zero_division=0)),
        "recall_fractured": float(recall_score(truth, predicted, zero_division=0)),
        "f1_fractured": float(f1_score(truth, predicted, zero_division=0)),
        "auc_fractured": float(roc_auc_score(truth, probabilities)) if len(np.unique(truth)) == 2 else float("nan"),
    }


def bootstrap_intervals(
    truth: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, seed: int, samples: int
) -> dict[str, list[float]]:
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in classification_metrics(truth, probabilities)}
    for _ in range(samples):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in selected])
        metrics = classification_metrics(truth[indices], probabilities[indices])
        for name, value in metrics.items():
            if np.isfinite(value):
                values[name].append(value)
    return {
        name: np.quantile(metric_values, [0.025, 0.975]).tolist()
        if metric_values else [float("nan"), float("nan")]
        for name, metric_values in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one saved classifier on the held-out test split.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/manifest.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/evaluation")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive.")

    frame = pd.read_csv(args.manifest)
    required = {"image_path", "label", "split"}
    if not required.issubset(frame.columns):
        parser.error(f"Manifest must contain {', '.join(sorted(required))}.")
    if "label_conflict" in frame:
        frame = frame[~frame.label_conflict.astype(str).str.casefold().isin({"true", "1"})]
    if "sha256" in frame:
        frame = frame.drop_duplicates("sha256", keep="first")
    frame = frame[frame.split == "test"].reset_index(drop=True)
    if frame.empty:
        parser.error("The manifest has no test images.")
    if set(frame.label) != set(CLASS_NAMES):
        parser.error("The held-out test split must contain both classes for evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_classifier(checkpoint["model_name"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(
        ManifestDataset(frame, int(checkpoint["image_size"]), train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.softmax(logits, dim=1)[:, CLASS_NAMES.index("fractured")].cpu().tolist())

    truth = (frame.label.to_numpy() == "fractured")
    probabilities = np.asarray(probabilities, dtype=float)
    groups = frame["group_id"].fillna(frame.get("sha256", frame.relative_path)).astype(str).to_numpy()
    metrics = classification_metrics(truth, probabilities)
    intervals = bootstrap_intervals(truth, probabilities, groups, args.seed, args.bootstrap_samples)
    for name, (low, high) in intervals.items():
        metrics[f"{name}_ci95_low"] = float(low)
        metrics[f"{name}_ci95_high"] = float(high)

    frame["true_label"] = frame.label
    frame["predicted_probability_fractured"] = probabilities
    frame["predicted_label"] = np.where(probabilities >= 0.5, "fractured", "not_fractured")
    frame["correct"] = frame.true_label == frame.predicted_label
    model_tag = f"{checkpoint['model_name']}_{checkpoint['image_size']}_s{checkpoint.get('seed', 0)}"
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / f"{model_tag}_test_predictions.csv"
    summary_path = args.output / f"{model_tag}_test_summary.json"
    frame[["source", "relative_path", "group_id", "true_label", "predicted_label",
           "predicted_probability_fractured", "correct"]].to_csv(predictions_path, index=False)
    summary = {
        "model": checkpoint["model_name"],
        "image_size": checkpoint["image_size"],
        "training_seed": checkpoint.get("seed", 0),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_auc_fractured": checkpoint["best_validation_metrics"]["auc_fractured"],
        "test_split": "test",
        "n": len(frame),
        "class_counts": frame.label.value_counts().to_dict(),
        "bootstrap_unit": "exact-image SHA-256 group",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": torchvision.__version__,
            "device": str(device),
        },
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved test summary: {summary_path}")


if __name__ == "__main__":
    main()
