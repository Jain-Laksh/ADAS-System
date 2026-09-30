from __future__ import annotations
import numpy as np
import pandas as pd


def compute_features(pair_df: pd.DataFrame, window_s: float, dt: float, ttc_cap: float = 10.0) -> pd.DataFrame:
    eps = 1e-6
    n = len(pair_df)
    window_steps = int(window_s / dt)
    
    f_gap = pair_df["filtered_gap"].values
    f_vrel = pair_df["filtered_vrel"].values
    ego_speed = pair_df["ego_speed"].values
    lead_speed = pair_df["filtered_lead_speed"].values
    f_alead = pair_df["filtered_a_lead"].values
    ego_acc = pair_df["ego_acc"].values
    
    features = pd.DataFrame(index=pair_df.index)
    features["gap"] = f_gap
    features["v_rel"] = f_vrel
    features["ego_speed"] = ego_speed
    features["lead_speed"] = lead_speed
    features["lead_acc"] = f_alead
    features["ego_acc"] = ego_acc
    
    features["time_headway"] = f_gap / np.maximum(ego_speed, eps)
    
    ttc = np.full(n, ttc_cap)
    closing = f_vrel < 0
    ttc[closing] = f_gap[closing] / np.maximum(-f_vrel[closing], eps)
    features["ttc"] = np.clip(ttc, 0, ttc_cap)
    
    req_decel = np.zeros(n)
    req_decel[closing] = (f_vrel[closing] ** 2) / (2.0 * np.maximum(f_gap[closing], eps))
    features["req_decel"] = np.clip(req_decel, 0, 20.0)
    
    # Rolling features
    df_vrel = pd.Series(f_vrel)
    features["v_rel_mean"] = df_vrel.rolling(window=window_steps, min_periods=1).mean()
    features["v_rel_min"] = df_vrel.rolling(window=window_steps, min_periods=1).min()
    features["v_rel_max"] = df_vrel.rolling(window=window_steps, min_periods=1).max()
    
    # Gap slope (using simple diff / time over window for speed, or just use v_rel directly)
    # The spec asks for gap slope:
    gap_diff = pd.Series(f_gap).diff(window_steps)
    time_diff = pd.Series(pair_df["time"]).diff(window_steps)
    slope = gap_diff / time_diff
    # fallback to immediate derivative (v_rel) where window is incomplete
    features["gap_slope"] = slope.fillna(pd.Series(f_vrel, index=pair_df.index))
    
    df_alead = pd.Series(f_alead)
    steps_1s = int(1.0 / dt)
    features["lead_acc_mean_1s"] = df_alead.rolling(window=steps_1s, min_periods=1).mean()
    features["lead_acc_min_window"] = df_alead.rolling(window=window_steps, min_periods=1).min()
    
    return features
