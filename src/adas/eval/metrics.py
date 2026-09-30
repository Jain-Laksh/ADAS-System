from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from adas.sim.replay import ReplayLog


def compute_replay_metrics(log: ReplayLog, cfg: dict) -> dict:
    gap = np.array(log.gap)
    ttc = np.array(log.ttc)
    aeb_state = np.array(log.aeb_state)
    ego_v = np.array(log.ego_v)
    ego_a = np.array(log.ego_a)
    
    collisions = np.sum(gap <= 0.0)
    min_gap = np.min(gap)
    min_ttc = np.min(ttc)
    
    aeb_activations = 0
    in_aeb = False
    for s in aeb_state:
        if s in ["partial_brake", "full_brake"]:
            if not in_aeb:
                aeb_activations += 1
                in_aeb = True
        else:
            in_aeb = False
            
    rms_accel = np.sqrt(np.mean(ego_a ** 2))
    jerk = np.diff(ego_a) / cfg["sim"]["dt"]
    peak_jerk = np.max(np.abs(jerk)) if len(jerk) > 0 else 0.0
    
    # ACC time gap error
    des_gap = cfg["control"]["acc"]["standstill_gap"] + cfg["control"]["acc"]["time_gap"] * ego_v
    rms_gap_error = np.sqrt(np.mean((gap - des_gap) ** 2))
    
    return {
        "collisions": int(collisions),
        "min_gap": float(min_gap),
        "min_ttc": float(min_ttc),
        "aeb_activations": int(aeb_activations),
        "rms_accel": float(rms_accel),
        "peak_jerk": float(peak_jerk),
        "rms_gap_error": float(rms_gap_error),
    }


def compute_ml_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "miss_rate": np.nan, "far": np.nan}
        
    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    preds = y_score > threshold
    
    tp = np.sum((preds == 1) & (y_true == 1))
    fn = np.sum((preds == 0) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    
    miss_rate = fn / max(1, (tp + fn))
    far = fp / max(1, (fp + tn))
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "miss_rate": float(miss_rate),
        "far": float(far),
    }
