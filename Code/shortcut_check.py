from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(path: str) -> list[float]:
    with Image.open(path) as image:
        gray = np.asarray(ImageOps.grayscale(image).resize((128, 128)), dtype=np.float32) / 255.0
    hist, _ = np.histogram(gray, bins=16, range=(0, 1), density=False)
    hist = hist.astype(np.float64) / max(hist.sum(), 1)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log(hist)).sum())
    return [float(gray.mean()), float(gray.std()), entropy,
            float((gray < 0.05).mean()), float((gray > 0.95).mean())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether simple image statistics predict the fracture label.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/manifest.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/shortcut_check.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    if "label_conflict" in frame:
        frame = frame[~frame.label_conflict.astype(str).str.casefold().isin({"true", "1"})]
    if "sha256" in frame:
        frame = frame.drop_duplicates("sha256", keep="first")
    train = frame[frame.split == "train"].copy()
    val = frame[frame.split == "val"].copy()
    if train.empty or val.empty or set(train.label) != {"fractured", "not_fractured"}:
        parser.error("Need both classes in train and a nonempty validation split.")
    x_train = np.asarray([features(p) for p in train.image_path])
    x_val = np.asarray([features(p) for p in val.image_path])
    y_train = (train.label == "fractured").astype(int).to_numpy()
    y_val = (val.label == "fractured").astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, probabilities) if len(np.unique(y_val)) == 2 else float("nan")
    result = pd.DataFrame([{"n_train": len(train), "n_val": len(val), "val_auc_image_statistics": auc,
                            "warning_auc_above_0_7": bool(np.isfinite(auc) and auc > 0.7),
                            "features": "mean, standard deviation, entropy, near-black fraction, near-white fraction"}])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print("This is a diagnostic baseline, not proof that model predictions are free from shortcuts.")


if __name__ == "__main__":
    main()

