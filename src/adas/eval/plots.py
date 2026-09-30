from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: str) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    
    ax2.plot(recall, precision, label=f"LightGBM (AUC = {pr_auc:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_reliability(y_true: np.ndarray, y_score: np.ndarray, out_path: str) -> None:
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "s-", label="LightGBM")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_conformal(alphas: list[float], fars: list[float], out_path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(alphas, fars, "o-", label="Empirical FAR")
    plt.plot([0, max(alphas)], [0, max(alphas)], "k--", label="Target Alpha")
    plt.xlabel("Target Alpha")
    plt.ylabel("Held-out Sample FAR")
    plt.title("Conformal Coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trace(log, out_path: str) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    t = log.time
    
    ax1.plot(t, log.gap, label="Gap")
    ax1.plot(t, log.filtered_gap, "--", label="Filtered Gap")
    ax1.set_ylabel("Distance (m)")
    ax1.legend()
    
    ax2.plot(t, log.ego_v, label="Ego Speed")
    ax2.plot(t, log.lead_v, label="Lead Speed")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend()
    
    ax3.plot(t, log.ego_a, label="Ego Accel (actual)")
    ax3.plot(t, log.acc_cmd, "--", label="ACC Cmd")
    ax3.plot(t, log.aeb_cmd, ":", label="AEB Cmd")
    ax3.set_ylabel("Accel (m/s^2)")
    ax3.set_xlabel("Time (s)")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_latency(latencies: list[float], out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.hist(latencies, bins=30, edgecolor="k")
    plt.axvline(np.percentile(latencies, 95), color="r", linestyle="dashed", label="95th Pctl")
    plt.xlabel("MPC Solve Time (ms)")
    plt.ylabel("Count")
    plt.title("Per-cycle Compute Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
