from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.enc1, self.enc2 = ConvBlock(1, width), ConvBlock(width, width * 2)
        self.enc3, self.enc4 = ConvBlock(width * 2, width * 4), ConvBlock(width * 4, width * 8)
        self.center = ConvBlock(width * 8, width * 16)
        self.dec4, self.dec3 = ConvBlock(width * 24, width * 8), ConvBlock(width * 12, width * 4)
        self.dec2, self.dec1 = ConvBlock(width * 6, width * 2), ConvBlock(width * 3, width)
        self.pool = nn.MaxPool2d(2)
        self.output = nn.Conv2d(width, 1, 1)

    @staticmethod
    def up(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, x), dim=1)

    def forward(self, x):
        a = self.enc1(x); b = self.enc2(self.pool(a)); c = self.enc3(self.pool(b))
        d = self.enc4(self.pool(c)); center = self.center(self.pool(d))
        x = self.dec4(self.up(center, d)); x = self.dec3(self.up(x, c))
        x = self.dec2(self.up(x, b)); x = self.dec1(self.up(x, a))
        return self.output(x)


class PairDataset(Dataset):
    def __init__(self, rows, size: int, augment: bool):
        self.rows, self.size, self.augment = rows.reset_index(drop=True), size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as image:
            image = ImageOps.grayscale(image).resize((self.size, self.size), Image.Resampling.BILINEAR)
            x = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(row.mask_path) as image:
            image = ImageOps.grayscale(image).resize((self.size, self.size), Image.Resampling.NEAREST)
            y = (np.asarray(image, dtype=np.uint8) > 0).astype(np.float32)
        if self.augment and random.random() < 0.5:
            x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    dice = (2 * (prob * target).sum(dims) + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return bce + (1 - dice).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); totals = np.zeros(3, dtype=float); count = 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        logits = model(image); loss = loss_fn(logits, target)
        pred = torch.sigmoid(logits) >= 0.5; truth = target >= 0.5
        dims = (1, 2, 3); intersection = (pred & truth).sum(dims).float()
        dice = (2 * intersection + 1e-7) / ((pred.sum(dims) + truth.sum(dims)).float() + 1e-7)
        iou = (intersection + 1e-7) / ((pred | truth).sum(dims).float() + 1e-7)
        n = image.size(0); totals += n * np.array([loss.item(), dice.mean().item(), iou.mean().item()]); count += n
    return totals / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small U-Net on paired masks; never reads the test split.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/data/manifest.csv")
    parser.add_argument("--mask-index", type=Path, required=True, help="make_masks.py index or a manifest with mask_path values")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/segmenters")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    base = pd.read_csv(args.manifest)
    if "label_conflict" in base:
        base = base[~base.label_conflict.astype(str).str.casefold().isin({"true", "1"})]
    if "sha256" in base:
        base = base.drop_duplicates("sha256", keep="first")
    masks = pd.read_csv(args.mask_index)
    if not {"relative_path", "split", "mask_path"}.issubset(masks.columns):
        parser.error("Mask index needs relative_path, split and mask_path columns.")
    key = ["relative_path", "split"]
    paired = base.merge(masks[key + ["mask_path"]].drop_duplicates(key), on=key, how="left", suffixes=("", "_mask"))
    if "mask_path_mask" in paired:
        paired["mask_path"] = paired["mask_path_mask"].fillna(paired["mask_path"])
        paired.drop(columns=["mask_path_mask"], inplace=True)
    paired = paired[paired.mask_path.fillna("").astype(str).map(lambda p: bool(p) and Path(p).is_file())]
    train = paired[paired.split == "train"].copy()
    val = paired[paired.split == "val"].copy()
    if len(train) < 2 or val.empty:
        parser.error("Need at least two train pairs and one validation pair. Test split is intentionally excluded.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(PairDataset(train, args.image_size, True), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(PairDataset(val, args.image_size, False), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    run = args.output / f"unet_{args.image_size}_s{args.seed}"; run.mkdir(parents=True, exist_ok=True)
    history = []; best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total, count = 0.0, 0
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(image), target)
            loss.backward(); optimizer.step(); total += loss.item() * len(image); count += len(image)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": val_loss,
               "val_dice_against_training_labels": val_dice, "val_iou_against_training_labels": val_iou}
        history.append(row); print(row)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model_name": "unet", "image_size": args.image_size, "seed": args.seed,
                        "model_state_dict": model.state_dict(), "validation_metrics": row}, run / "best.pt")
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    print(f"Saved best checkpoint to {run / 'best.pt'} on {device}; test split was not evaluated.")


if __name__ == "__main__":
    main()

