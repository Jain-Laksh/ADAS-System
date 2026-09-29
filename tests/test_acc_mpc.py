from __future__ import annotations
import numpy as np

from adas.control.acc_mpc import AccMPC


class TestAccMPC:
    def setup_method(self) -> None:
        self.cfg = {
            "control": {
                "acc": {
                    "standstill_gap": 5.0,
                    "time_gap": 1.5,
                    "horizon": 30,
                    "dt": 0.1,
                    "a_min": -3.0,
                    "a_max": 2.0,
                    "j_max": 2.5,
                    "set_speed": 35.0,
                    "q_gap": 5.0,
                    "q_speed": 1.0,
                    "r_accel": 1.0,
                    "r_jerk": 2.0,
                    "slack_penalty": 1000.0,
                    "d_min": 2.0,
                }
            }
        }

    def test_accel_limits(self) -> None:
        mpc = AccMPC(self.cfg)
        a_cmd, _ = mpc.step(gap=100.0, ego_speed=0.0, lead_speed=30.0, lead_acc=0.0, a_prev=0.0)
        assert a_cmd <= 2.0
        
        a_cmd, _ = mpc.step(gap=10.0, ego_speed=30.0, lead_speed=0.0, lead_acc=0.0, a_prev=0.0)
        assert a_cmd >= -3.0

    def test_jerk_limits(self) -> None:
        mpc = AccMPC(self.cfg)
        a_cmd, _ = mpc.step(gap=100.0, ego_speed=0.0, lead_speed=30.0, lead_acc=0.0, a_prev=0.0)
        dt = self.cfg["control"]["acc"]["dt"]
        jerk = (a_cmd - 0.0) / dt
        assert abs(jerk) <= 2.5 + 1e-3

    def test_gap_tracking(self) -> None:
        mpc = AccMPC(self.cfg)
        lead_speed = 20.0
        ego_speed = 20.0
        desired_gap = 5.0 + 1.5 * ego_speed
        
        a_cmd, _ = mpc.step(gap=desired_gap + 10.0, ego_speed=ego_speed, lead_speed=lead_speed, lead_acc=0.0, a_prev=0.0)
        assert a_cmd > 0.0
        
        a_cmd, _ = mpc.step(gap=desired_gap - 10.0, ego_speed=ego_speed, lead_speed=lead_speed, lead_acc=0.0, a_prev=0.0)
        assert a_cmd < 0.0

    def test_no_solver_failure(self) -> None:
        mpc = AccMPC(self.cfg)
        a_cmd, ms = mpc.step(gap=0.5, ego_speed=30.0, lead_speed=0.0, lead_acc=0.0, a_prev=2.0)
        assert ms >= 0.0
        assert a_cmd >= -3.0
