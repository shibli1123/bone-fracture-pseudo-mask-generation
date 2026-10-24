from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run CAM summary CSVs over seeds.")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/evaluation")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/results")
    args = parser.parse_args()
    files = [path for path in args.input.glob("*_summary.csv") if path.name != "aggregate_summary.csv"]
    if not files:
        parser.error(f"No per-run *_summary.csv files found under {args.input}.")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["run_file"] = path.name
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if "seed" not in merged:
        merged["seed"] = merged.run_file.str.extract(r"_s(\d+)")[0]
    metric_columns = [column for column in merged if column.endswith("_mean") and column.rsplit("_", 1)[0] in {"dice", "iou", "precision", "recall", "pointing_game"}]
    if not metric_columns:
        parser.error("Input files do not contain supported metric mean columns.")
    dimensions = [name for name in ("cam_tag", "split", "threshold", "source") if name in merged]
    summary = merged.groupby(dimensions, dropna=False)[metric_columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    summary = summary.reset_index()
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "aggregate_summary.csv", index=False)
    merged.to_csv(args.output / "all_run_summaries.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Aggregated {len(files)} run summaries into {args.output}")


if __name__ == "__main__":
    main()

