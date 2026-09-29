from __future__ import annotations
from adas.control.supervisor import Supervisor


class TestSupervisor:
    def setup_method(self) -> None:
        self.cfg = {
            "control": {
                "supervisor": {
                    "response_time": 0.5,
                    "a_max": 2.0,
                    "b_min": 4.0,
                    "b_max": 8.0,
                    "veto_gap_factor": 2.5,
                    "veto_ttc": 4.0,
                }
            }
        }

    def test_force_brake(self) -> None:
        sup = Supervisor(self.cfg)
        # Ego fast, lead slow -> large d_min
        force, veto = sup.check(gap=5.0, ego_speed=30.0, lead_speed=10.0, ttc=1.0, risk_triggered=False)
        assert force is True
        assert veto is False

    def test_veto_trigger(self) -> None:
        sup = Supervisor(self.cfg)
        # Ego slow, lead fast -> small d_min
        force, veto = sup.check(gap=100.0, ego_speed=10.0, lead_speed=30.0, ttc=10.0, risk_triggered=True)
        assert force is False
        assert veto is True

    def test_normal_pass_through(self) -> None:
        sup = Supervisor(self.cfg)
        force, veto = sup.check(gap=50.0, ego_speed=20.0, lead_speed=20.0, ttc=5.0, risk_triggered=False)
        assert force is False
        assert veto is False
