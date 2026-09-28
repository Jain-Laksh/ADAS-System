from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def extract_pairs(df: pd.DataFrame, min_frames: int = 200, min_speed: float = 1.0) -> list[pd.DataFrame]:
    df_with_leader = df[df["preceding"] > 0].copy()
    leader_cols = ["vehicle_id", "frame_id", "local_y", "v_vel", "v_acc", "v_length", "lane_id"]
    leader = df[leader_cols].rename(columns={
        "vehicle_id": "lead_id",
        "local_y": "lead_y",
        "v_vel": "lead_vel",
        "v_acc": "lead_acc",
        "v_length": "lead_length",
        "lane_id": "lead_lane",
    })
    merged = df_with_leader.merge(
        leader,
        left_on=["preceding", "frame_id"],
        right_on=["lead_id", "frame_id"],
        how="inner",
    )
    merged["gap"] = merged["lead_y"] - merged["local_y"] - merged["lead_length"]
    pairs = []
    for (veh_id, lead_id, lane_id), grp in merged.groupby(["vehicle_id", "lead_id", "lane_id"]):
        grp = grp.sort_values("frame_id").reset_index(drop=True)
        if grp["lead_lane"].nunique() > 1:
            continue
        segments = _split_contiguous(grp, min_frames, min_speed)
        for seg in segments:
            pairs.append(_build_pair_df(seg, int(veh_id), int(lead_id)))
    return pairs


def _split_contiguous(grp: pd.DataFrame, min_frames: int, min_speed: float) -> list[pd.DataFrame]:
    frames = grp["frame_id"].values
    diffs = np.diff(frames)
    breaks = np.where(diffs != 1)[0] + 1
    segments_idx = np.split(np.arange(len(grp)), breaks)
    results = []
    for idx in segments_idx:
        if len(idx) < min_frames:
            continue
        seg = grp.iloc[idx]
        if (seg["gap"] <= 0).any():
            continue
        if (seg["v_vel"] < min_speed).any():
            continue
        results.append(seg)
    return results


def _build_pair_df(seg: pd.DataFrame, veh_id: int, lead_id: int) -> pd.DataFrame:
    dt = 0.1
    out = pd.DataFrame({
        "time": np.arange(len(seg)) * dt,
        "frame_id": seg["frame_id"].values,
        "follower_id": veh_id,
        "leader_id": lead_id,
        "lead_pos": seg["lead_y"].values,
        "lead_speed": seg["lead_vel"].values,
        "lead_acc": seg["lead_acc"].values,
        "ego_pos": seg["local_y"].values,
        "ego_speed": seg["v_vel"].values,
        "gap": seg["gap"].values,
    })
    return out


def smooth_pair(pair_df: pd.DataFrame, window: int = 51, polyorder: int = 3) -> pd.DataFrame:
    dt = 0.1
    out = pair_df.copy()
    if len(out) <= window:
        return out
    lead_length = (pair_df["lead_pos"].values - pair_df["ego_pos"].values - pair_df["gap"].values)
    lead_length_mean = np.median(lead_length)
    for col in ["lead_pos", "lead_speed", "ego_pos", "ego_speed"]:
        out[col] = savgol_filter(out[col].values, window, polyorder)
    out["lead_acc"] = np.gradient(out["lead_speed"].values, dt)
    out["ego_acc"] = np.gradient(out["ego_speed"].values, dt)
    out["gap"] = out["lead_pos"] - out["ego_pos"] - lead_length_mean
    return out


def smooth_pairs(pairs: list[pd.DataFrame], window: int = 51, polyorder: int = 3) -> list[pd.DataFrame]:
    return [smooth_pair(p, window, polyorder) for p in pairs]
