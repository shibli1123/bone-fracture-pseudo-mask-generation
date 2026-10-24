from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_ALIASES = {
    "fracture": "fractured", "fractured": "fractured", "positive": "fractured",
    "fracture present": "fractured", "not fractured": "not_fractured",
    "not_fractured": "not_fractured", "non-fractured": "not_fractured",
    "nonfractured": "not_fractured", "non fractured": "not_fractured",
    "normal": "not_fractured", "negative": "not_fractured",
}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}


def normalize_label(value: str) -> str | None:
    return LABEL_ALIASES.get(" ".join(value.casefold().replace("-", " ").split()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_source(path: Path, root: Path) -> str:
    parts = [part.casefold().replace("_", "-") for part in path.relative_to(root).parts]
    known = ("fracatlas", "grazpedwri", "xray-datasets", "xray-bone-fracture")
    for part in parts:
        if any(name in part for name in known):
            return next(name for name in known if name in part)
    for part in parts:
        if "bone-fracture-dataset" in part or part == "cropped-dataset":
            return "bone-fracture-dataset"
    return parts[0] if len(parts) > 1 else "unspecified"


def infer_split(path: Path) -> str | None:
    for part in path.parts:
        if part.casefold() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.casefold()]
    return None


def _mask_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        if not any(part.casefold() in {"mask", "masks", "annotations", "ground-truth", "ground_truth"}
                   for part in path.parts):
            continue
        stem = path.stem.casefold()
        for suffix in ("_mask", "-mask", "_segmentation", "_annotation"):
            stem = stem.removesuffix(suffix)
        index.setdefault(stem, path)
    return index


def _assign_group_splits(records: list[dict], seed: int, ratios: tuple[float, float, float]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[row["sha256"]].append(row)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[tuple[str, list[dict]]]] = defaultdict(list)
    for group_id, members in groups.items():
        labels = {row["label"] for row in members}
        sources = {row["source"] for row in members}
        splits = {row["source_split"] for row in members if row["source_split"]}
        key = ("mixed" if len(sources) > 1 else next(iter(sources)),
               "conflict" if len(labels) > 1 else next(iter(labels)))
        for row in members:
            row["group_id"] = group_id
            row["duplicate_count"] = len(members)
            row["label_conflict"] = len(labels) > 1
        if len(splits) == 1:
            assigned = next(iter(splits))
        else:
            buckets[key].append((group_id, members))
            continue
        for row in members:
            row["split"] = assigned
    for group_list in buckets.values():
        rng.shuffle(group_list)
        n = len(group_list)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        for i, (_, members) in enumerate(group_list):
            split = "train" if i < train_end else "val" if i < val_end else "test"
            for row in members:
                row["split"] = split


def collect_manifest(data_root: Path, seed: int = 42,
                     ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> list[dict]:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")
    ratio_sum = sum(ratios)
    if any(value <= 0 for value in ratios) or abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("Split ratios must be positive and sum to 1.")

    mask_index = _mask_index(data_root)
    records: list[dict] = []
    ignored = {"mask", "masks", "annotations", "ground-truth", "ground_truth"}
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        if any(part.casefold() in ignored for part in path.parts):
            continue
        labels = [normalize_label(part) for part in path.parts[:-1]]
        labels = [label for label in labels if label]
        if not labels:
            continue
        label = labels[-1]
        stem = path.stem.casefold()
        mask_path = mask_index.get(stem)
        if mask_path is None:
            for suffix in ("_yolo", "_image", "_img"):
                mask_path = mask_index.get(stem.removesuffix(suffix))
                if mask_path:
                    break
        records.append({
            "source": infer_source(path, data_root),
            "relative_path": path.relative_to(data_root).as_posix(),
            "image_path": str(path.resolve()),
            "label": label,
            "source_split": infer_split(path),
            "sha256": sha256_file(path),
            "mask_path": str(mask_path.resolve()) if mask_path else "",
        })
    if not records:
        raise ValueError(f"No images in class folders named fractured/not fractured under {data_root}")
    _assign_group_splits(records, seed, ratios)
    for row in records:
        row.pop("source_split", None)
    return records

