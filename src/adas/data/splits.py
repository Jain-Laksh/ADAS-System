from __future__ import annotations
import numpy as np
import pandas as pd


def split_by_vehicle(
    pairs: list[pd.DataFrame],
    ratios: list[float] = [0.6, 0.2, 0.2],
    seed: int = 42,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    vehicle_ids = sorted(set(p["follower_id"].iloc[0] for p in pairs))
    rng = np.random.RandomState(seed)
    rng.shuffle(vehicle_ids)
    n = len(vehicle_ids)
    n_train = int(n * ratios[0])
    n_cal = int(n * (ratios[0] + ratios[1]))
    train_ids = set(vehicle_ids[:n_train])
    cal_ids = set(vehicle_ids[n_train:n_cal])
    test_ids = set(vehicle_ids[n_cal:])
    train = [p for p in pairs if p["follower_id"].iloc[0] in train_ids]
    cal = [p for p in pairs if p["follower_id"].iloc[0] in cal_ids]
    test = [p for p in pairs if p["follower_id"].iloc[0] in test_ids]
    return train, cal, test
