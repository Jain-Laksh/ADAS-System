from __future__ import annotations
import numpy as np
import pandas as pd


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    # Keep only non-critical samples
    cal_scores = scores[labels == 0]
    n = len(cal_scores)
    if n == 0:
        return 1.0  # Fallback if no negative samples
        
    sorted_scores = np.sort(cal_scores)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = min(max(idx, 0), n - 1)
    
    return sorted_scores[idx]


def compute_far(scores: np.ndarray, labels: np.ndarray, threshold: float, episodes: np.ndarray | None = None) -> dict:
    non_critical_mask = (labels == 0)
    if not np.any(non_critical_mask):
        return {"sample_far": 0.0, "episode_far": 0.0}
        
    nc_scores = scores[non_critical_mask]
    sample_far = np.mean(nc_scores > threshold)
    
    episode_far = 0.0
    if episodes is not None:
        nc_episodes = episodes[non_critical_mask]
        unique_eps = np.unique(nc_episodes)
        if len(unique_eps) > 0:
            ep_triggers = 0
            for ep in unique_eps:
                ep_mask = (nc_episodes == ep)
                if np.any(nc_scores[ep_mask] > threshold):
                    ep_triggers += 1
            episode_far = ep_triggers / len(unique_eps)
            
    return {"sample_far": float(sample_far), "episode_far": float(episode_far)}


def sweep_alphas(scores: np.ndarray, labels: np.ndarray, alphas: list[float], episodes: np.ndarray | None = None) -> pd.DataFrame:
    results = []
    for a in alphas:
        thresh = calibrate_threshold(scores, labels, a)
        far_metrics = compute_far(scores, labels, thresh, episodes)
        results.append({
            "alpha": a,
            "threshold": thresh,
            "sample_far": far_metrics["sample_far"],
            "episode_far": far_metrics["episode_far"],
        })
    return pd.DataFrame(results)
