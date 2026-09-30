from __future__ import annotations
import numpy as np
import pandas as pd


def compute_labels(pair_df: pd.DataFrame, horizon_s: float, ttc_threshold: float, dt: float) -> np.ndarray:
    gap = pair_df["gap"].values
    v_rel = pair_df["lead_speed"].values - pair_df["ego_speed"].values
    
    # Compute instantaneous TTC for all frames
    ttc = np.full_like(gap, 10.0)
    closing = v_rel < 0
    ttc[closing] = gap[closing] / -v_rel[closing]
    
    n = len(pair_df)
    horizon_steps = int(horizon_s / dt)
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        end_idx = min(i + horizon_steps, n)
        if end_idx > i:
            min_ttc = np.min(ttc[i:end_idx])
            if min_ttc < ttc_threshold:
                labels[i] = 1
                
    return labels
