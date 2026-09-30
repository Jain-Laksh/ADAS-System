from __future__ import annotations
import os
import pickle
import yaml
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_risk_model import process_dataset
from adas.ml.model import load_model
from adas.ml.conformal import sweep_alphas
from adas.eval.metrics import compute_ml_metrics, compute_replay_metrics
from adas.eval.plots import plot_roc_pr, plot_reliability, plot_conformal, plot_trace, plot_latency
from adas.sim.replay import run_replay
from adas.control.acc_mpc import AccMPC
from adas.control.aeb_fsm import AEBFSM
from adas.control.supervisor import Supervisor
from adas.control.arbiter import Arbiter
from adas.eval.ablation import run_ablations


def main() -> None:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    with open("results/eval_data.pkl", "rb") as f:
        eval_data = pickle.load(f)
        
    test_pairs = eval_data["test_pairs"]
    stress_scenarios = eval_data["stress_scenarios"]
    
    model = load_model("results/risk_model.lgb")
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    print("--- Experiment 1: Risk Predictor (Open Loop) ---")
    X_test, y_test = process_dataset(test_pairs, cfg)
    scores = model.predict(X_test)
    
    # Fixed TTC baseline
    ttc_baseline = np.zeros_like(y_test)
    ttc_baseline[X_test["ttc"] < 2.0] = 1.0
    
    ttc_metrics = compute_ml_metrics(y_test, ttc_baseline, 0.5)
    ml_metrics = compute_ml_metrics(y_test, scores, 0.5)
    print(f"ML ROC AUC: {ml_metrics['roc_auc']:.3f}, PR AUC: {ml_metrics['pr_auc']:.3f}")
    
    plot_roc_pr(y_test, scores, "results/figures/roc_pr.png")
    plot_reliability(y_test, scores, "results/figures/reliability.png")
    
    print("\n--- Experiment 2: Conformal Coverage ---")
    alphas = cfg["ml"]["conformal"]["alphas"]
    # Provide dummy episodes for FAR (assuming each 100 frames is an episode for simplicity in open loop)
    episodes = np.arange(len(y_test)) // 100 
    
    with open("configs/default.yaml") as f:
        cal_cfg = yaml.safe_load(f)
    
    # Needs calibration set
    with open(os.path.join(cfg["data"]["processed_dir"], "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    X_cal, y_cal = process_dataset(splits["cal"], cfg)
    cal_scores = model.predict(X_cal)
    
    conformal_results = sweep_alphas(cal_scores, y_cal, alphas, np.arange(len(y_cal)) // 100)
    
    # Test coverage on test set
    test_fars = []
    for t in conformal_results["threshold"]:
        far = np.mean(scores[y_test == 0] > t)
        test_fars.append(far)
        
    plot_conformal(alphas, test_fars, "results/figures/conformal.png")
    
    thresh_01 = float(conformal_results[conformal_results["alpha"] == 0.01]["threshold"].iloc[0])
    print(f"Alpha=1% Threshold: {thresh_01:.4f}")
    
    print("\n--- Experiment 3 & 4 & 5: Closed-loop AEB & ACC ---")
    class RiskFn:
        def __call__(self, log, i): 
            from adas.ml.features import compute_step_features
            f = compute_step_features(log, i, cfg["ml"]["feature_window_s"], cfg["sim"]["dt"])
            return model.predict(f)[0]
        
    # Run one stress scenario for tracing
    scen = stress_scenarios[0]
    risk_fn = RiskFn()
    
    acc = AccMPC(cfg)
    aeb = AEBFSM(cfg)
    sup = Supervisor(cfg)
    arb = Arbiter(cfg)
    
    log = run_replay(scen, cfg, acc, aeb, sup, arb, risk_fn, thresh_01)
    plot_trace(log, "results/figures/trace.png")
    plot_latency(log.mpc_solve_ms, "results/figures/latency.png")
    
    lat = np.array(log.mpc_solve_ms)
    print(f"MPC Latency: Mean={np.mean(lat):.1f}ms, 95th={np.percentile(lat, 95):.1f}ms")
    
    print("\n--- Experiment 6: Ablations ---")
    # Quick ablation run
    abl_df = run_ablations(test_pairs[:2], stress_scenarios, cfg, model, thresh_01)
    print(abl_df)
    abl_df.to_csv("results/tables/ablations.csv", index=False)
    
    print("\nEvaluation complete! Results in results/figures/ and results/tables/")


if __name__ == "__main__":
    main()
