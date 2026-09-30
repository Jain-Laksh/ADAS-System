from __future__ import annotations
import numpy as np
import pandas as pd
from adas.sim.replay import run_replay
from adas.eval.metrics import compute_replay_metrics
from adas.control.acc_mpc import AccMPC
from adas.control.aeb_fsm import AEBFSM
from adas.control.supervisor import Supervisor
from adas.control.arbiter import Arbiter


def _get_components(cfg: dict, mode: str, model=None, thresh: float=0.5):
    class RiskFn:
        def __init__(self, m):
            self.m = m
        def __call__(self, log, i):
            if self.m is None: return 0.0
            from adas.ml.features import compute_step_features
            f = compute_step_features(log, i, cfg["ml"]["feature_window_s"], cfg["sim"]["dt"])
            return self.m.predict(f)[0]

    acc = AccMPC(cfg)
    arb = Arbiter(cfg)
    
    if mode == "full_stack":
        return acc, AEBFSM(cfg), Supervisor(cfg), arb, RiskFn
    elif mode == "no_kalman":
        return acc, AEBFSM(cfg), Supervisor(cfg), arb, RiskFn # Filter disabled in replay.py logic
    elif mode == "no_supervisor_force":
        sup = Supervisor(cfg)
        sup.check = lambda g, e, l, t, r: (False, sup.__class__.check(sup, g, e, l, t, r)[1])
        return acc, AEBFSM(cfg), sup, arb, RiskFn
    elif mode == "no_supervisor_veto":
        sup = Supervisor(cfg)
        sup.check = lambda g, e, l, t, r: (sup.__class__.check(sup, g, e, l, t, r)[0], False)
        return acc, AEBFSM(cfg), sup, arb, RiskFn
    elif mode == "ttc_trigger_only":
        return acc, AEBFSM(cfg), Supervisor(cfg), arb, None
    elif mode == "no_aeb":
        return acc, None, None, None, None
    
    return acc, AEBFSM(cfg), Supervisor(cfg), arb, RiskFn


def run_ablations(pairs: list[pd.DataFrame], stress_scenarios: list[pd.DataFrame], cfg: dict, model=None, threshold=0.5) -> pd.DataFrame:
    modes = [
        "full_stack", 
        "no_supervisor_force", 
        "no_supervisor_veto", 
        "ttc_trigger_only", 
        "no_aeb"
    ]
    
    results = []
    
    for mode in modes:
        acc, aeb, sup, arb, RiskFnCls = _get_components(cfg, mode, model, threshold)
        
        # Stress scenarios
        cols = 0
        min_gaps = []
        for pair in stress_scenarios:
            risk_fn = RiskFnCls(model) if RiskFnCls else None
            
            # Simple TTC trigger fallback
            if mode == "ttc_trigger_only":
                def ttc_risk(log, i):
                    ttc_val = pair["gap"].iloc[i] / max(1e-6, pair["ego_speed"].iloc[i] - pair["lead_speed"].iloc[i])
                    return 1.0 if ttc_val < 2.0 else 0.0
                risk_fn = ttc_risk
                
            log = run_replay(pair, cfg, acc, aeb, sup, arb, risk_fn, threshold)
            m = compute_replay_metrics(log, cfg)
            cols += m["collisions"]
            min_gaps.append(m["min_gap"])
            
        results.append({
            "mode": mode,
            "stress_collisions": cols,
            "stress_mean_min_gap": float(np.mean(min_gaps)),
        })
        
    return pd.DataFrame(results)
