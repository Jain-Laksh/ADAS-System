from __future__ import annotations
import pytest

from adas.control.aeb_fsm import AEBState, AEBFSM


class TestAEBFSM:
    def setup_method(self) -> None:
        self.cfg = {
            "control": {
                "aeb": {
                    "debounce_cycles": 3,
                    "warn_hold_s": 0.4,
                    "ttc_partial": 2.5,
                    "ttc_full": 1.2,
                    "decel_full": 5.0,
                    "partial_brake": -4.0,
                    "full_brake": -8.0,
                    "exit_score_frac": 0.8,
                    "exit_ttc": 3.0,
                    "exit_hold_cycles": 5,
                }
            }
        }

    def test_monitor_to_warn(self) -> None:
        fsm = AEBFSM(self.cfg)
        for _ in range(2):
            state, brake = fsm.step(0.9, 0.5, 10.0, 0.0, False, 20.0, 20.0)
            assert state == AEBState.MONITOR
        
        state, brake = fsm.step(0.9, 0.5, 10.0, 0.0, False, 20.0, 20.0)
        assert state == AEBState.WARN
        assert brake == 0.0

    def test_warn_to_partial(self) -> None:
        fsm = AEBFSM(self.cfg)
        for _ in range(3):
            fsm.step(0.9, 0.5, 10.0, 0.0, False, 20.0, 20.0)
        
        # In WARN, hold for 0.4s (4 cycles total, since we just entered)
        for _ in range(4):
            fsm.step(0.9, 0.5, 10.0, 0.0, False, 20.0, 20.0)
            
        state, brake = fsm.step(0.9, 0.5, 2.0, 0.0, False, 20.0, 20.0)
        assert state == AEBState.PARTIAL_BRAKE
        assert brake == -4.0

    def test_partial_to_full(self) -> None:
        fsm = AEBFSM(self.cfg)
        fsm.state = AEBState.PARTIAL_BRAKE
        
        state, brake = fsm.step(0.9, 0.5, 1.0, 6.0, False, 20.0, 10.0)
        assert state == AEBState.FULL_BRAKE
        assert brake == -8.0

    def test_full_brake_latch(self) -> None:
        fsm = AEBFSM(self.cfg)
        fsm.state = AEBState.FULL_BRAKE
        
        state, brake = fsm.step(0.1, 0.5, 10.0, 0.0, False, 20.0, 10.0)
        assert state == AEBState.FULL_BRAKE
        
        state, brake = fsm.step(0.1, 0.5, 10.0, 0.0, False, 10.0, 15.0)
        assert state == AEBState.MONITOR

    def test_hysteresis(self) -> None:
        fsm = AEBFSM(self.cfg)
        fsm.state = AEBState.PARTIAL_BRAKE
        
        for _ in range(4):
            state, brake = fsm.step(0.1, 0.5, 5.0, 0.0, False, 20.0, 15.0)
            assert state == AEBState.PARTIAL_BRAKE
            
        state, brake = fsm.step(0.1, 0.5, 5.0, 0.0, False, 20.0, 15.0)
        assert state == AEBState.MONITOR
