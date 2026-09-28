from __future__ import annotations
import os
import pickle
import yaml

from adas.data.loader import load_ngsim
from adas.data.preprocess import extract_pairs, smooth_pairs
from adas.data.splits import split_by_vehicle


def main() -> None:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    print("Loading NGSIM data...")
    df = load_ngsim(dc["raw_file"])
    print(f"  Loaded {len(df)} rows, {df['vehicle_id'].nunique()} vehicles")

    print("Extracting leader-follower pairs...")
    min_frames = int(dc["min_duration_s"] / 0.1)
    pairs = extract_pairs(df, min_frames=min_frames, min_speed=dc["min_speed_ms"])
    print(f"  Extracted {len(pairs)} pairs")

    total_frames = sum(len(p) for p in pairs)
    follower_ids = set(p["follower_id"].iloc[0] for p in pairs)
    median_gap = float(
        __import__("numpy").median(
            __import__("numpy").concatenate([p["gap"].values for p in pairs])
        )
    )
    print(f"  Total frames: {total_frames}")
    print(f"  Unique followers: {len(follower_ids)}")
    print(f"  Median gap: {median_gap:.1f} m")

    print("Smoothing pairs...")
    pairs = smooth_pairs(pairs, window=dc["savgol_window"], polyorder=dc["savgol_polyorder"])

    print("Splitting by vehicle...")
    train, cal, test = split_by_vehicle(pairs, dc["split_ratios"], dc["split_seed"])
    train_ids = set(p["follower_id"].iloc[0] for p in train)
    cal_ids = set(p["follower_id"].iloc[0] for p in cal)
    test_ids = set(p["follower_id"].iloc[0] for p in test)
    overlap = train_ids & cal_ids | train_ids & test_ids | cal_ids & test_ids
    print(f"  Train: {len(train)} pairs ({len(train_ids)} vehicles)")
    print(f"  Cal:   {len(cal)} pairs ({len(cal_ids)} vehicles)")
    print(f"  Test:  {len(test)} pairs ({len(test_ids)} vehicles)")
    print(f"  Vehicle overlap: {len(overlap)}")

    os.makedirs(dc["processed_dir"], exist_ok=True)
    out_path = os.path.join(dc["processed_dir"], "splits.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"train": train, "cal": cal, "test": test}, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
