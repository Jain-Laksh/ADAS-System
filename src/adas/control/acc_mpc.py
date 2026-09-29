from __future__ import annotations
import time
import numpy as np
import cvxpy as cp


class AccMPC:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["control"]["acc"]
        self.N = self.cfg["horizon"]
        self.dt = self.cfg["dt"]
        self.time_gap = self.cfg["time_gap"]
        self.standstill_gap = self.cfg["standstill_gap"]

        self.p_gap = cp.Parameter(value=0.0)
        self.p_v_ego = cp.Parameter(value=0.0)
        self.p_a_prev = cp.Parameter(value=0.0)
        self.p_v_lead = cp.Parameter(self.N, value=np.zeros(self.N))

        self.v_ego = cp.Variable(self.N + 1)
        self.gap = cp.Variable(self.N + 1)
        self.a_ego = cp.Variable(self.N)
        self.slack = cp.Variable(self.N, nonneg=True)

        cost = 0
        constraints = [
            self.gap[0] == self.p_gap,
            self.v_ego[0] == self.p_v_ego,
        ]

        for k in range(self.N):
            constraints.append(self.gap[k + 1] == self.gap[k] + (self.p_v_lead[k] - self.v_ego[k]) * self.dt)
            constraints.append(self.v_ego[k + 1] == self.v_ego[k] + self.a_ego[k] * self.dt)

            constraints.append(self.a_ego[k] >= self.cfg["a_min"])
            constraints.append(self.a_ego[k] <= self.cfg["a_max"])
            constraints.append(self.v_ego[k] >= 0.0)
            constraints.append(self.v_ego[k] <= self.cfg["set_speed"])

            if k == 0:
                jerk = (self.a_ego[k] - self.p_a_prev) / self.dt
            else:
                jerk = (self.a_ego[k] - self.a_ego[k - 1]) / self.dt
            constraints.append(jerk >= -self.cfg["j_max"])
            constraints.append(jerk <= self.cfg["j_max"])

            constraints.append(self.gap[k] + self.slack[k] >= self.cfg["d_min"])

            d_des = self.standstill_gap + self.time_gap * self.v_ego[k]
            cost += self.cfg["q_gap"] * cp.square(self.gap[k] - d_des)
            cost += self.cfg["q_speed"] * cp.square(self.v_ego[k] - self.p_v_lead[k])
            cost += self.cfg["r_accel"] * cp.square(self.a_ego[k])
            cost += self.cfg["r_jerk"] * cp.square(jerk)
            cost += self.cfg["slack_penalty"] * self.slack[k]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def step(self, gap: float, ego_speed: float, lead_speed: float, lead_acc: float, a_prev: float) -> tuple[float, float]:
        t0 = time.time()
        
        self.p_gap.value = gap
        self.p_v_ego.value = ego_speed
        self.p_a_prev.value = a_prev

        v_lead = np.zeros(self.N)
        v_l = lead_speed
        for k in range(self.N):
            v_lead[k] = v_l
            v_l = max(0.0, v_l + lead_acc * self.dt * np.exp(-k * self.dt / 1.5))
        self.p_v_lead.value = v_lead

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            solve_ms = (time.time() - t0) * 1000.0
            
            if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                a_cmd = -3.0
            else:
                a_cmd = float(self.a_ego.value[0])
        except Exception:
            solve_ms = (time.time() - t0) * 1000.0
            a_cmd = -3.0

        return a_cmd, solve_ms
