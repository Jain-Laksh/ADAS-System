from __future__ import annotations
import os
import pickle
import yaml
import numpy as np
import pandas as pd

from adas.sim.replay import run_replay, create_stress_scenario
from adas.control.acc_mpc import AccMPC
from adas.control.aeb_fsm import AEBFSM
from adas.control.supervisor import Supervisor
from adas.control.arbiter import Arbiter
from adas.ml.model import load_model


def main() -> None:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    splits_path = os.path.join(cfg["data"]["processed_dir"], "splits.pkl")
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    test_pairs = splits["test"]
    print(f"Loaded {len(test_pairs)} test pairs.")

    model_path = "results/risk_model.lgb"
    model = None
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Loaded risk model.")

    stress_scenarios = []
    e_cfg = cfg["eval"]
    for speed in e_cfg["stress_speeds"]:
        for decel in e_cfg["stress_decels"]:
            for gap in e_cfg["stress_gaps"]:
                scen = create_stress_scenario(speed, gap, decel, duration_s=10.0, dt=cfg["sim"]["dt"])
                stress_scenarios.append(scen)
    
    print(f"Created {len(stress_scenarios)} stress scenarios.")
    
    # Save test pairs and scenarios to process in evaluation
    os.makedirs("results", exist_ok=True)
    with open("results/eval_data.pkl", "wb") as f:
        pickle.dump({
            "test_pairs": test_pairs,
            "stress_scenarios": stress_scenarios,
        }, f)
        
    print("Replay setup complete. Run run_evaluation.py for full metrics.")

if __name__ == "__main__":
    main()
