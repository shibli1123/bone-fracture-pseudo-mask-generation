from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from wsss.models import CLASS_NAMES, build_classifier


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.size, self.train = image_size, train
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            image = ImageOps.grayscale(image).convert("RGB")
            if self.train:
                if random.random() < 0.5:
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
                if random.random() < 0.5:
                    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.10))
            x = self.to_tensor(image)
        return x, CLASS_NAMES.index(row.label)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    ys, probs = [], []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_total += criterion(logits, y).item() * len(y)
            ys.extend(y.cpu().tolist())
            probs.extend(torch.softmax(logits, dim=1)[:, 0].cpu().tolist())
    y_fractured = np.asarray(ys) == CLASS_NAMES.index("fractured")
    pred_fractured = np.asarray(probs) >= 0.5
    metrics = {
        "loss": loss_total / max(len(ys), 1),
        "accuracy": accuracy_score(y_fractured, pred_fractured),
        "precision_fractured": precision_score(y_fractured, pred_fractured, zero_division=0),
        "recall_fractured": recall_score(y_fractured, pred_fractured, zero_division=0),
        "f1_fractured": f1_score(y_fractured, pred_fractured, zero_division=0),
        "auc_fractured": roc_auc_score(y_fractured, probs) if len(set(ys)) == 2 else float("nan"),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transfer-learning fracture classifier.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/manifest.csv")
    parser.add_argument("--model", choices=("resnet50", "densenet121"), default="resnet50")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/classifiers")
    parser.add_argument("--no-pretrained", action="store_true", help="Use random initialization; not the default research setting.")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("epochs must be positive and warmup-epochs must be in [0, epochs).")
    seed_everything(args.seed)
    frame = pd.read_csv(args.manifest)
    if not {"image_path", "label", "split"}.issubset(frame.columns):
        parser.error("Manifest must contain image_path, label and split columns. Run prepare_data.py first.")
    if "label_conflict" in frame:
        frame = frame[~frame.label_conflict.astype(str).str.casefold().isin({"true", "1"})]
    if "sha256" in frame:
        frame = frame.drop_duplicates("sha256", keep="first")
    train_frame = frame[frame.split == "train"].copy()
    val_frame = frame[frame.split == "val"].copy()
    if train_frame.empty or val_frame.empty:
        parser.error("Both train and val splits need images.")
    if set(train_frame.label) != set(CLASS_NAMES):
        parser.error("Training split must contain both fractured and not_fractured classes.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ManifestDataset(train_frame, args.image_size, True)
    val_ds = ManifestDataset(val_frame, args.image_size, False)
    class_counts = Counter(train_frame.label)
    weights = [1.0 / class_counts[label] for label in train_frame.label]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

    model = build_classifier(args.model, pretrained=not args.no_pretrained).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if args.model == "resnet50" else model.classifier
    for parameter in head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_head, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    run = args.output / f"{args.model}_{args.image_size}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -float("inf")
    best_path = run / "best.pt"

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True
            head_ids = {id(parameter) for parameter in head.parameters()}
            backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
            optimizer = torch.optim.AdamW([
                {"params": backbone, "lr": args.lr_backbone},
                {"params": head.parameters(), "lr": args.lr_head},
            ], weight_decay=1e-4)
        model.train()
        total_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            seen += len(y)
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1), **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row)
        score = metrics["auc_fractured"]
        if np.isnan(score):
            score = metrics["f1_fractured"]
        if score > best_auc:
            best_auc = score
            torch.save({"model_name": args.model, "image_size": args.image_size, "seed": args.seed,
                        "class_names": CLASS_NAMES, "model_state_dict": model.state_dict(),
                        "best_validation_metrics": metrics, "epoch": epoch + 1}, best_path)
        print(json.dumps(row))

    with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    (run / "run.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path}")
    print(f"Device: {device}; no test split was evaluated.")


if __name__ == "__main__":
    main()
