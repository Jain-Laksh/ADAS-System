from __future__ import annotations
from enum import Enum


class AEBState(Enum):
    MONITOR = "monitor"
    WARN = "warn"
    PARTIAL_BRAKE = "partial_brake"
    FULL_BRAKE = "full_brake"


class AEBFSM:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["control"]["aeb"]
        self.state = AEBState.MONITOR
        self.warn_cycles = 0
        self.exit_cycles = 0

    def step(self, risk_score: float, threshold: float, ttc: float, required_decel: float,
             supervisor_force: bool, ego_speed: float, lead_speed: float) -> tuple[AEBState, float]:
        
        # Risk logic
        if risk_score > threshold:
            self.warn_cycles += 1
        else:
            self.warn_cycles = 0

        # Exit logic
        if risk_score < self.cfg["exit_score_frac"] * threshold and ttc > self.cfg["exit_ttc"]:
            self.exit_cycles += 1
        else:
            self.exit_cycles = 0
            
        should_exit = self.exit_cycles >= self.cfg["exit_hold_cycles"]
        
        # State transitions
        if self.state == AEBState.MONITOR:
            if self.warn_cycles >= self.cfg["debounce_cycles"]:
                self.state = AEBState.WARN
                
        elif self.state == AEBState.WARN:
            if should_exit:
                self.state = AEBState.MONITOR
            else:
                warn_s = self.warn_cycles * 0.1 # assuming dt = 0.1, or tracking cycles directly
                if (warn_s >= self.cfg["warn_hold_s"] and ttc < self.cfg["ttc_partial"]) or supervisor_force:
                    self.state = AEBState.PARTIAL_BRAKE
                    
        elif self.state == AEBState.PARTIAL_BRAKE:
            if required_decel > self.cfg["decel_full"] or ttc < self.cfg["ttc_full"]:
                self.state = AEBState.FULL_BRAKE
            elif should_exit:
                self.state = AEBState.MONITOR
                
        elif self.state == AEBState.FULL_BRAKE:
            if ego_speed <= lead_speed or ego_speed < 0.1:
                self.state = AEBState.MONITOR

        # Actions
        brake_req = 0.0
        if self.state == AEBState.PARTIAL_BRAKE:
            brake_req = self.cfg["partial_brake"]
        elif self.state == AEBState.FULL_BRAKE:
            brake_req = self.cfg["full_brake"]

        return self.state, brake_req
