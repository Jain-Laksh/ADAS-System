from __future__ import annotations


class Supervisor:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["control"]["supervisor"]

    def check(self, gap: float, ego_speed: float, lead_speed: float, ttc: float, risk_triggered: bool) -> tuple[bool, bool]:
        rho = self.cfg["response_time"]
        a_max = self.cfg["a_max"]
        b_min = self.cfg["b_min"]
        b_max = self.cfg["b_max"]

        d_min = ego_speed * rho + 0.5 * a_max * (rho ** 2) + \
                ((ego_speed + rho * a_max) ** 2) / (2 * b_min) - \
                (lead_speed ** 2) / (2 * b_max)
        d_min = max(0.0, d_min)

        force_brake = gap < d_min
        
        veto_trigger = gap > self.cfg["veto_gap_factor"] * d_min and ttc > self.cfg["veto_ttc"]

        return force_brake, veto_trigger
