from __future__ import annotations
import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adas.sensing.noise import add_noise
from adas.sensing.kalman import KalmanFilter
from adas.ml.labels import compute_labels
from adas.ml.features import compute_features
from adas.ml.model import train_model, save_model


def process_dataset(pairs: list[pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    dt = cfg["sim"]["dt"]
    s_cfg = cfg["sensing"]
    ml_cfg = cfg["ml"]
    
    all_X = []
    all_y = []
    
    rng = np.random.RandomState(s_cfg["noise_seed"])
    
    for pair in pairs:
        # 1. Run Kalman filter over the pair
        n = len(pair)
        true_gap = pair["gap"].values
        true_vrel = pair["lead_speed"].values - pair["ego_speed"].values
        a_ego = pair["ego_acc"].values
        
        noisy_gap, noisy_vrel, valid = add_noise(
            true_gap, true_vrel, s_cfg["noise_gap_sigma"], s_cfg["noise_vrel_sigma"],
            s_cfg["dropout_rate"], rng
        )
        
        kf = KalmanFilter(dt, s_cfg["jerk_noise_density"], s_cfg["noise_gap_sigma"]**2, s_cfg["noise_vrel_sigma"]**2)
        kf.reset(noisy_gap[0], noisy_vrel[0])
        
        f_gap = np.zeros(n)
        f_vrel = np.zeros(n)
        f_alead = np.zeros(n)
        
        for i in range(n):
            kf.predict(a_ego[i])
            if valid[i]:
                kf.update(np.array([noisy_gap[i], noisy_vrel[i]]))
            state = kf.get_state()
            f_gap[i] = state[0]
            f_vrel[i] = state[1]
            f_alead[i] = state[2]
            
        pair_augmented = pair.copy()
        pair_augmented["filtered_gap"] = f_gap
        pair_augmented["filtered_vrel"] = f_vrel
        pair_augmented["filtered_a_lead"] = f_alead
        pair_augmented["filtered_lead_speed"] = pair["ego_speed"].values + f_vrel
        
        # 2. Features and Labels
        y = compute_labels(pair_augmented, ml_cfg["label_horizon_s"], ml_cfg["ttc_threshold"], dt)
        X = compute_features(pair_augmented, ml_cfg["feature_window_s"], dt, ml_cfg["ttc_cap"])
        
        # Drop the first few frames due to rolling windows and Kalman burn-in (e.g. 2 seconds)
        burn_in = int(2.0 / dt)
        
        all_X.append(X.iloc[burn_in:])
        all_y.append(y[burn_in:])
        
    X_df = pd.concat(all_X, ignore_index=True)
    y_arr = np.concatenate(all_y)
    
    return X_df, y_arr


def main() -> None:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
        
    splits_path = os.path.join(cfg["data"]["processed_dir"], "splits.pkl")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)
        
    print("Processing train set...")
    # Split train into train and val (80/20 by vehicle) for early stopping
    train_pairs = splits["train"]
    veh_ids = sorted(list(set(p["follower_id"].iloc[0] for p in train_pairs)))
    rng = np.random.RandomState(42)
    rng.shuffle(veh_ids)
    
    split_idx = int(0.8 * len(veh_ids))
    real_train_ids = set(veh_ids[:split_idx])
    real_train_pairs = [p for p in train_pairs if p["follower_id"].iloc[0] in real_train_ids]
    val_pairs = [p for p in train_pairs if p["follower_id"].iloc[0] not in real_train_ids]
    
    X_train, y_train = process_dataset(real_train_pairs, cfg)
    print("Processing validation set...")
    X_val, y_val = process_dataset(val_pairs, cfg)
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
    
    pos_rate = np.mean(y_train)
    print(f"Positive rate in train: {pos_rate * 100:.2f}%")
    
    print("Training LightGBM model...")
    model = train_model(X_train, y_train, X_val, y_val, cfg)
    
    val_preds = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Validation ROC AUC: {val_auc:.4f}")
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/risk_model.lgb"
    save_model(model, out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
