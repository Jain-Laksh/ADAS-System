from __future__ import annotations
from adas.control.aeb_fsm import AEBState


class Arbiter:
    def __init__(self, cfg: dict) -> None:
        self.j_max = cfg["control"]["acc"]["j_max"]
        self.dt = cfg["control"]["acc"]["dt"]

    def arbitrate(self, acc_cmd: float, aeb_cmd: float, aeb_state: AEBState, a_prev: float) -> tuple[float, float]:
        if aeb_state in [AEBState.PARTIAL_BRAKE, AEBState.FULL_BRAKE]:
            applied = aeb_cmd
            a_prev = applied
        else:
            acc_rate_limited = max(a_prev - self.j_max * self.dt, min(a_prev + self.j_max * self.dt, acc_cmd))
            applied = min(acc_rate_limited, aeb_cmd) if aeb_cmd < 0 else acc_rate_limited
            a_prev = applied

        return applied, a_prev
