from __future__ import annotations
import numpy as np


class KalmanFilter:
    def __init__(
        self,
        dt: float,
        jerk_noise_density: float = 1.0,
        r_gap: float = 0.09,
        r_vrel: float = 0.0225,
    ) -> None:
        self.dt = dt
        self.x = np.zeros(3)
        self.P = np.eye(3) * 100.0

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt

        self.F = np.array([
            [1.0, dt, 0.5 * dt2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])

        self.B = np.array([
            [-0.5 * dt2],
            [-dt],
            [0.0],
        ])

        q = jerk_noise_density ** 2
        self.Q = q * np.array([
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ])

        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        self.R = np.array([
            [r_gap, 0.0],
            [0.0, r_vrel],
        ])

    def reset(self, gap: float, v_rel: float) -> None:
        self.x = np.array([gap, v_rel, 0.0])
        self.P = np.eye(3) * 100.0

    def predict(self, a_ego: float) -> None:
        u = np.array([a_ego])
        self.x = self.F @ self.x + self.B.flatten() * a_ego
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()
