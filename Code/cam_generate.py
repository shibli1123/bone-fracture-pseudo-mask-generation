from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from wsss.cam import CAMGenerator
from wsss.models import CLASS_NAMES, build_classifier


def load_image(path: str, size: int, device):
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with Image.open(path) as source:
        return transform(ImageOps.grayscale(source).convert("RGB")).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class activation maps for a manifest split.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/manifest.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("gradcam", "gradcam++", "layercam"), default="layercam")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--tta-flip", action="store_true", help="Average CAMs from original and horizontally flipped image.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/cams")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    size = int(checkpoint["image_size"])
    model = build_classifier(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    generator = CAMGenerator(model, model_name, args.method)
    frame = pd.read_csv(args.manifest)
    if "label_conflict" in frame:
        frame = frame[~frame.label_conflict.astype(str).str.casefold().isin({"true", "1"})]
    if "sha256" in frame:
        frame = frame.drop_duplicates("sha256", keep="first")
    if args.split != "all":
        frame = frame[frame.split == args.split]
    tag = f"{model_name}_{size}_s{checkpoint.get('seed', 0)}_{args.method}" + ("_tta" if args.tta_flip else "")
    out_dir = args.output / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for row in frame.itertuples(index=False):
            x = load_image(row.image_path, size, device)
            cam, logits = generator(x, class_index=CLASS_NAMES.index("fractured"))
            if args.tta_flip:
                flipped, _ = generator(x.flip(-1), class_index=CLASS_NAMES.index("fractured"))
                cam = (cam + flipped.flip(-1)) / 2.0
                cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-8)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            relative = Path(row.relative_path)
            safe_name = hashlib.sha256(row.relative_path.encode("utf-8")).hexdigest()[:16]
            cam_path = out_dir / row.split / relative.parent / f"{safe_name}.npy"
            cam_path.parent.mkdir(parents=True, exist_ok=True)
            array = cam[0].cpu().numpy().astype(np.float32)
            np.save(cam_path, array)
            preview = out_dir / row.split / relative.parent / f"{safe_name}.png"
            Image.fromarray(np.round(array * 255).astype(np.uint8)).save(preview)
            rows.append({"source": row.source, "relative_path": row.relative_path, "image_path": row.image_path,
                         "label": row.label, "split": row.split, "sha256": row.sha256,
                         "group_id": getattr(row, "group_id", row.sha256),
                         "cam_tag": tag,
                         "mask_path": row.mask_path, "cam_path": str(cam_path.resolve()),
                         "predicted_fracture_probability": float(probabilities[0])})
    finally:
        generator.close()
    index = out_dir / "index.csv"
    with index.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} CAMs and index {index}")


if __name__ == "__main__":
    main()
