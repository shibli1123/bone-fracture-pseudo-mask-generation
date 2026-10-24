from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from wsss.data import collect_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image folders and create a hash-grouped manifest.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "Datasets")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = collect_manifest(args.data, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    columns = ["source", "relative_path", "image_path", "label", "split", "sha256", "group_id",
               "duplicate_count", "label_conflict", "mask_path"]
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["source"], r["split"], r["label"]) for r in rows)
    collisions = defaultdict(set)
    for row in rows:
        collisions[row["group_id"]].add(row["split"])
    cross_split = sum(len(splits) > 1 for splits in collisions.values())
    conflicts = sum(bool(r["label_conflict"]) for r in rows)
    masks = sum(bool(r["mask_path"]) for r in rows)
    lines = ["# Dataset audit", "", f"Images indexed: {len(rows)}",
             f"Unique exact-image hashes: {len(collisions)}",
             f"Exact duplicate groups crossing splits: {cross_split}",
             f"Images in label-conflict duplicate groups: {conflicts}",
             f"Images paired with a candidate mask file: {masks}", "",
             "| Source | Split | Label | Images |", "|---|---|---|---:|"]
    for (source, split, label), count in sorted(counts.items()):
        lines.append(f"| {source} | {split} | {label} | {count} |")
    lines += ["", "Mask pairing is filename based and must be reviewed before mask metrics are reported.",
              "Patient identifiers were not available; SHA-256 grouping only removes exact duplicates.",
              "Near-duplicate images may still cross splits."]
    (args.out / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest} ({len(rows)} images); audit: {args.out / 'audit.md'}")


if __name__ == "__main__":
    main()
