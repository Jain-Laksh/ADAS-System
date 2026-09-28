from __future__ import annotations
import pandas as pd


COLUMNS = [
    "vehicle_id", "frame_id", "total_frames", "global_time",
    "local_x", "local_y", "global_x", "global_y",
    "v_length", "v_width", "v_class", "v_vel", "v_acc",
    "lane_id", "preceding", "following", "space_headway", "time_headway",
]

FT_TO_M = 0.3048


def load_ngsim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df["local_x"] = df["local_x"] * FT_TO_M
    df["local_y"] = df["local_y"] * FT_TO_M
    df["v_length"] = df["v_length"] * FT_TO_M
    df["v_width"] = df["v_width"] * FT_TO_M
    df["v_vel"] = df["v_vel"] * FT_TO_M
    df["v_acc"] = df["v_acc"] * FT_TO_M
    df["space_headway"] = df["space_headway"] * FT_TO_M
    frame_min = df["frame_id"].min()
    df["time_s"] = (df["frame_id"] - frame_min) * 0.1
    return df
