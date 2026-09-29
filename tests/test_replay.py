from __future__ import annotations
import numpy as np
import pytest
import pandas as pd

from adas.sim.replay import run_replay, ReplayLog, create_stress_scenario
from adas.sim.vehicle import EgoVehicle


class TestEgoVehicle:
    def test_constant_speed(self) -> None:
        ego = EgoVehicle(x0=0.0, v0=20.0, dt=0.1, tau=0.25)
        for _ in range(100):
            x, v, a = ego.step(0.0)
        assert abs(v - 20.0) < 0.01

    def test_speed_clips_at_zero(self) -> None:
        ego = EgoVehicle(x0=0.0, v0=1.0, dt=0.1, tau=0.05)
        for _ in range(200):
            x, v, a = ego.step(-10.0)
        assert v >= 0.0


class TestReplay:
    def _make_constant_pair(self, n: int = 200, speed: float = 20.0, gap: float = 30.0) -> pd.DataFrame:
        dt = 0.1
        time = np.arange(n) * dt
        ego_pos = speed * time
        lead_pos = ego_pos + gap
        return pd.DataFrame({
            "time": time,
            "lead_pos": lead_pos,
            "lead_speed": np.full(n, speed),
            "lead_acc": np.zeros(n),
            "ego_pos": ego_pos,
            "ego_speed": np.full(n, speed),
            "gap": np.full(n, gap),
        })

    def test_log_length(self) -> None:
        pair = self._make_constant_pair(n=100)
        cfg = {
            "sim": {"dt": 0.1, "actuator_tau": 0.25},
            "sensing": {
                "noise_gap_sigma": 0.0,
                "noise_vrel_sigma": 0.0,
                "dropout_rate": 0.0,
                "noise_seed": 42,
                "jerk_noise_density": 1.0,
            },
        }
        log = run_replay(pair, cfg)
        assert len(log.time) == 100
        assert len(log.gap) == 100

    def test_constant_speed_gap_stable(self) -> None:
        pair = self._make_constant_pair(n=300, speed=20.0, gap=30.0)
        cfg = {
            "sim": {"dt": 0.1, "actuator_tau": 0.25},
            "sensing": {
                "noise_gap_sigma": 0.0,
                "noise_vrel_sigma": 0.0,
                "dropout_rate": 0.0,
                "noise_seed": 42,
                "jerk_noise_density": 0.1,
            },
        }
        log = run_replay(pair, cfg)
        gaps = np.array(log.gap)
        assert np.all(gaps > 25.0)
        assert np.all(gaps < 35.0)


class TestStressScenario:
    def test_creates_valid_df(self) -> None:
        df = create_stress_scenario(25.0, 30.0, -6.0)
        assert len(df) == 150
        assert (df["lead_speed"] >= 0).all()
        assert df["lead_pos"].iloc[-1] >= df["lead_pos"].iloc[0]
