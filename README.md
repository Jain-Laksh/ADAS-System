# ADAS Longitudinal Control Stack

A Python-based simulation of an ADAS longitudinal control stack combining an MPC-based Adaptive Cruise Control (ACC) and an ML-triggered Autonomous Emergency Braking (AEB) system. The stack operates on a Kalman-filtered lead vehicle state and includes a rule-based safety supervisor.

Built with AI assistance in ~1 day for an ADAS Engineer application (Oct 4 deadline).

## Architecture

The system runs entirely as an offline log replay, requiring no vehicle hardware or real-time simulation. The stack consists of:
1. **Kalman Filter**: Tracks the lead vehicle using synthetic noisy radar measurements (range and range rate).
2. **ACC MPC**: Constrained MPC formulation using `cvxpy` and `OSQP`, providing comfort-bounded acceleration to maintain a speed-dependent time gap.
3. **ML Risk Predictor**: A LightGBM model trained on historical trajectory sequences predicting a critical event in the next 3 seconds. The trigger threshold is calibrated using split-conformal prediction for a bounded False Activation Rate (FAR).
4. **AEB FSM**: A 4-state machine (Monitor, Warn, Partial Brake, Full Brake) that initiates hard braking.
5. **Safety Supervisor**: An RSS-style rule-based minimum safe distance envelope that can force AEB or veto an ML trigger.
6. **Arbiter**: Selects the strongest deceleration request.

## Tech Stack
* Python 3.11+ (NumPy, pandas, SciPy, cvxpy, lightgbm)
* Tests: `pytest`
* Data: NGSIM US-101 (public dataset)

## Instructions

### 1. Data Preparation
Download the NGSIM US-101 `trajectories-0750am-0805am.txt` file and place it in `data/raw/`.
```bash
python scripts/prepare_data.py
```
This extracts ~1,600 leader-follower pairs, smooths their states, and splits them by vehicle ID.

### 2. ML Training
```bash
python scripts/train_risk_model.py
```
This script adds synthetic radar noise, filters the data through the Kalman filter to mimic online conditions, extracts feature windows, and trains the LightGBM risk model.

### 3. Evaluation
```bash
python scripts/run_replay.py
python scripts/run_evaluation.py
```
This evaluates the risk model's open-loop performance, tests conformal coverage, runs closed-loop stress scenarios, logs the MPC latency, and performs component ablations. Results are saved in `results/figures/` and `results/tables/`.

## Constraints & Assumptions
* **CPU only**: No GPU, CFD, or complex vehicle-dynamics simulator used.
* **Offline Replay**: Everything runs on recorded NGSIM datasets.
* **Radar Noise Model**: Assumed Gaussian noise with 0.3m gap and 0.15m/s relative speed variance, with a 2% dropout rate.
