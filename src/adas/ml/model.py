from __future__ import annotations
import lightgbm as lgb
import numpy as np
import pandas as pd


def train_model(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, cfg: dict) -> lgb.Booster:
    lgbm_cfg = cfg["control"]["ml"]["lgbm"] if "ml" in cfg["control"] else cfg["ml"]["lgbm"] # Handling dict path
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = neg_count / max(1, pos_count)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": lgbm_cfg["learning_rate"],
        "num_leaves": lgbm_cfg["num_leaves"],
        "scale_pos_weight": scale_pos_weight,
        "seed": lgbm_cfg["seed"],
        "verbose": -1,
    }
    
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=lgbm_cfg["n_estimators"],
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=lgbm_cfg["early_stopping_rounds"])]
    )
    
    return booster

def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)

def save_model(model: lgb.Booster, path: str) -> None:
    model.save_model(path)

def load_model(path: str) -> lgb.Booster:
    return lgb.Booster(model_file=path)
