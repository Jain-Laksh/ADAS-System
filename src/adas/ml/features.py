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


def compute_step_features(log, i: int, window_s: float, dt: float, ttc_cap: float = 10.0) -> pd.DataFrame:
    # Computes features for step i using log history
    eps = 1e-6
    window_steps = int(window_s / dt)
    start_idx = max(0, i - window_steps + 1)
    
    f_gap_hist = log.filtered_gap[start_idx:i+1]
    f_vrel_hist = log.filtered_vrel[start_idx:i+1]
    f_alead_hist = log.filtered_a_lead[start_idx:i+1]
    time_hist = log.time[start_idx:i+1]
    
    f_gap = f_gap_hist[-1]
    f_vrel = f_vrel_hist[-1]
    ego_speed = log.ego_v[i]
    ego_acc = log.ego_a[i]
    f_alead = f_alead_hist[-1]
    lead_speed = ego_speed + f_vrel
    
    time_headway = f_gap / max(ego_speed, eps)
    
    ttc = ttc_cap
    if f_vrel < 0:
        ttc = min(f_gap / max(-f_vrel, eps), ttc_cap)
        
    req_decel = 0.0
    if f_vrel < 0:
        req_decel = min((f_vrel ** 2) / (2.0 * max(f_gap, eps)), 20.0)
        
    v_rel_mean = np.mean(f_vrel_hist)
    v_rel_min = np.min(f_vrel_hist)
    v_rel_max = np.max(f_vrel_hist)
    
    if len(f_gap_hist) > 1 and time_hist[-1] > time_hist[0]:
        gap_slope = (f_gap_hist[-1] - f_gap_hist[0]) / (time_hist[-1] - time_hist[0])
    else:
        gap_slope = f_vrel
        
    steps_1s = int(1.0 / dt)
    start_1s = max(0, i - steps_1s + 1)
    lead_acc_mean_1s = np.mean(log.filtered_a_lead[start_1s:i+1])
    lead_acc_min_window = np.min(f_alead_hist)
    
    feat_dict = {
        "gap": [f_gap],
        "v_rel": [f_vrel],
        "ego_speed": [ego_speed],
        "lead_speed": [lead_speed],
        "lead_acc": [f_alead],
        "ego_acc": [ego_acc],
        "time_headway": [time_headway],
        "ttc": [ttc],
        "req_decel": [req_decel],
        "v_rel_mean": [v_rel_mean],
        "v_rel_min": [v_rel_min],
        "v_rel_max": [v_rel_max],
        "gap_slope": [gap_slope],
        "lead_acc_mean_1s": [lead_acc_mean_1s],
        "lead_acc_min_window": [lead_acc_min_window],
    }
    return pd.DataFrame(feat_dict)
