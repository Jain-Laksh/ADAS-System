from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from adas.sensing.noise import add_noise
from adas.sensing.kalman import KalmanFilter
from adas.sim.vehicle import EgoVehicle


@dataclass
class ReplayLog:
    time: list[float] = field(default_factory=list)
    ego_x: list[float] = field(default_factory=list)
    ego_v: list[float] = field(default_factory=list)
    ego_a: list[float] = field(default_factory=list)
    lead_x: list[float] = field(default_factory=list)
    lead_v: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    filtered_gap: list[float] = field(default_factory=list)
    filtered_vrel: list[float] = field(default_factory=list)
    filtered_a_lead: list[float] = field(default_factory=list)
    ttc: list[float] = field(default_factory=list)
    aeb_state: list[str] = field(default_factory=list)
    acc_cmd: list[float] = field(default_factory=list)
    aeb_cmd: list[float] = field(default_factory=list)
    applied_cmd: list[float] = field(default_factory=list)
    risk_score: list[float] = field(default_factory=list)
    mpc_solve_ms: list[float] = field(default_factory=list)


def compute_ttc(gap: float, v_rel: float, cap: float = 10.0) -> float:
    if v_rel >= 0:
        return cap
    closing = -v_rel
    if closing < 1e-6:
        return cap
    return min(gap / closing, cap)


def run_replay(
    pair_df: pd.DataFrame,
    cfg: dict,
    acc_controller=None,
    aeb_fsm=None,
    supervisor=None,
    arbiter=None,
    risk_fn=None,
    conformal_threshold: float = 0.5,
) -> ReplayLog:
    dt = cfg["sim"]["dt"]
    tau = cfg["sim"]["actuator_tau"]
    sensing_cfg = cfg["sensing"]

    lead_x_arr = pair_df["lead_pos"].values
    lead_v_arr = pair_df["lead_speed"].values
    lead_a_arr = pair_df.get("lead_acc", pd.Series(np.zeros(len(pair_df)))).values
    ego_x0 = pair_df["ego_pos"].values[0]
    ego_v0 = pair_df["ego_speed"].values[0]

    n_steps = len(pair_df)
    rng = np.random.RandomState(sensing_cfg.get("noise_seed", 123))

    true_gap_arr = lead_x_arr - ego_x0
    v_rel_arr = lead_v_arr - ego_v0

    ego = EgoVehicle(ego_x0, ego_v0, dt, tau)
    kf = KalmanFilter(
        dt=dt,
        jerk_noise_density=sensing_cfg["jerk_noise_density"],
        r_gap=sensing_cfg["noise_gap_sigma"] ** 2,
        r_vrel=sensing_cfg["noise_vrel_sigma"] ** 2,
    )

    first_gap = lead_x_arr[0] - ego_x0
    first_vrel = lead_v_arr[0] - ego_v0
    kf.reset(first_gap, first_vrel)

    log = ReplayLog()
    a_prev = 0.0

    for i in range(n_steps):
        t = i * dt
        lx = lead_x_arr[i]
        lv = lead_v_arr[i]

        true_gap = lx - ego.x
        true_vrel = lv - ego.v

        noisy_gap = true_gap + rng.randn() * sensing_cfg["noise_gap_sigma"]
        noisy_vrel = true_vrel + rng.randn() * sensing_cfg["noise_vrel_sigma"]
        valid = rng.rand() > sensing_cfg["dropout_rate"]

        kf.predict(ego.a_actual)
        if valid:
            kf.update(np.array([noisy_gap, noisy_vrel]))

        state = kf.get_state()
        f_gap = max(state[0], 0.0)
        f_vrel = state[1]
        f_a_lead = state[2]
        f_lead_speed = ego.v + f_vrel

        log.time.append(t)
        log.ego_x.append(ego.x)
        log.ego_v.append(ego.v)
        log.ego_a.append(ego.a_actual)
        log.lead_x.append(lx)
        log.lead_v.append(lv)
        log.gap.append(true_gap)
        log.filtered_gap.append(f_gap)
        log.filtered_vrel.append(f_vrel)
        log.filtered_a_lead.append(f_a_lead)

        ttc = compute_ttc(f_gap, f_vrel)

        risk = 0.0
        if risk_fn is not None:
            risk = risk_fn(log, i)

        sup_force = False
        sup_veto = False
        if supervisor is not None:
            sup_force, sup_veto = supervisor.check(
                f_gap, ego.v, f_lead_speed, ttc, risk > conformal_threshold
            )

        aeb_state_str = "monitor"
        aeb_brake = 0.0
        if aeb_fsm is not None:
            req_decel = f_vrel ** 2 / (2.0 * max(f_gap, 0.1)) if f_vrel < 0 else 0.0
            aeb_state_val, aeb_brake = aeb_fsm.step(
                risk, conformal_threshold, ttc, req_decel, sup_force,
                ego.v, f_lead_speed,
            )
            aeb_state_str = aeb_state_val.value

        acc_cmd = 0.0
        solve_ms = 0.0
        if acc_controller is not None:
            acc_cmd, solve_ms = acc_controller.step(
                f_gap, ego.v, f_lead_speed, f_a_lead, a_prev,
            )

        if arbiter is not None:
            from adas.control.aeb_fsm import AEBState
            aeb_state_enum = AEBState(aeb_state_str)
            applied, a_prev = arbiter.arbitrate(acc_cmd, aeb_brake, aeb_state_enum, a_prev)
        else:
            applied = min(acc_cmd, aeb_brake) if aeb_brake < 0 else acc_cmd
            a_prev = applied

        ego.step(applied)

        log.ttc.append(ttc)
        log.aeb_state.append(aeb_state_str)
        log.acc_cmd.append(acc_cmd)
        log.aeb_cmd.append(aeb_brake)
        log.applied_cmd.append(applied)
        log.risk_score.append(risk)
        log.mpc_solve_ms.append(solve_ms)

    return log


def create_stress_scenario(
    lead_speed_0: float,
    gap_0: float,
    lead_decel: float,
    ego_speed_0: float | None = None,
    duration_s: float = 15.0,
    dt: float = 0.1,
) -> pd.DataFrame:
    if ego_speed_0 is None:
        ego_speed_0 = lead_speed_0
    n = int(duration_s / dt)
    time = np.arange(n) * dt
    lead_pos = np.zeros(n)
    lead_speed = np.zeros(n)
    lead_acc = np.zeros(n)

    lv = lead_speed_0
    lx = gap_0 + 0.0
    brake_start = int(3.0 / dt)
    for i in range(n):
        lead_pos[i] = lx
        lead_speed[i] = lv
        if i >= brake_start and lv > 0:
            la = lead_decel
        else:
            la = 0.0
        lead_acc[i] = la
        lv = max(0.0, lv + la * dt)
        lx += lv * dt

    return pd.DataFrame({
        "time": time,
        "lead_pos": lead_pos,
        "lead_speed": lead_speed,
        "lead_acc": lead_acc,
        "ego_pos": np.zeros(n),
        "ego_speed": np.full(n, ego_speed_0),
        "gap": lead_pos.copy(),
    })
