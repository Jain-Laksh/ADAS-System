from __future__ import annotations
import numpy as np
import pytest

from adas.sensing.kalman import KalmanFilter
from adas.sensing.noise import add_noise


class TestKalmanConvergence:
    def test_constant_speed_convergence(self) -> None:
        dt = 0.1
        n_steps = 200
        true_gap = 20.0
        true_vrel = 0.0
        a_ego = 0.0

        kf = KalmanFilter(dt=dt, jerk_noise_density=1.0, r_gap=0.09, r_vrel=0.0225)
        kf.reset(true_gap + 2.0, true_vrel + 1.0)

        for _ in range(n_steps):
            kf.predict(a_ego)
            z = np.array([true_gap, true_vrel])
            kf.update(z)

        state = kf.get_state()
        assert abs(state[0] - true_gap) < 0.1
        assert abs(state[1] - true_vrel) < 0.05

    def test_filtered_rmse_below_noisy(self) -> None:
        dt = 0.1
        n_steps = 500
        rng = np.random.RandomState(42)

        ego_speed = 20.0
        lead_speed = 22.0
        a_ego = 0.0
        a_lead = -0.5
        gap_0 = 25.0

        true_gap = np.zeros(n_steps)
        true_vrel = np.zeros(n_steps)
        v_l = lead_speed
        v_e = ego_speed
        d = gap_0
        for i in range(n_steps):
            true_gap[i] = d
            true_vrel[i] = v_l - v_e
            v_l = max(0, v_l + a_lead * dt)
            d = d + (v_l - v_e) * dt

        noisy_gap, noisy_vrel, valid = add_noise(
            true_gap, true_vrel, gap_sigma=0.3, vrel_sigma=0.15, dropout_rate=0.0, rng=rng
        )

        kf = KalmanFilter(dt=dt, jerk_noise_density=1.0, r_gap=0.09, r_vrel=0.0225)
        kf.reset(noisy_gap[0], noisy_vrel[0])

        filtered_gap = np.zeros(n_steps)
        filtered_vrel = np.zeros(n_steps)
        for i in range(n_steps):
            kf.predict(a_ego)
            kf.update(np.array([noisy_gap[i], noisy_vrel[i]]))
            state = kf.get_state()
            filtered_gap[i] = state[0]
            filtered_vrel[i] = state[1]

        burn_in = 50
        noisy_gap_rmse = np.sqrt(np.mean((noisy_gap[burn_in:] - true_gap[burn_in:]) ** 2))
        filtered_gap_rmse = np.sqrt(np.mean((filtered_gap[burn_in:] - true_gap[burn_in:]) ** 2))
        assert filtered_gap_rmse < noisy_gap_rmse

    def test_predict_only_on_dropout(self) -> None:
        dt = 0.1
        kf = KalmanFilter(dt=dt)
        kf.reset(20.0, 0.0)
        P_before = kf.get_covariance().copy()

        kf.predict(0.0)
        P_after = kf.get_covariance()

        assert np.all(np.diag(P_after) >= np.diag(P_before))
        state = kf.get_state()
        assert np.isfinite(state).all()
