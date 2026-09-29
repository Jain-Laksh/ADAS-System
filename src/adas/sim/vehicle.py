from __future__ import annotations


class EgoVehicle:
    def __init__(self, x0: float, v0: float, dt: float, tau: float = 0.25) -> None:
        self.x = x0
        self.v = v0
        self.a_actual = 0.0
        self.dt = dt
        self.tau = tau

    def step(self, a_cmd: float) -> tuple[float, float, float]:
        alpha = self.dt / self.tau
        self.a_actual += alpha * (a_cmd - self.a_actual)
        self.v = max(0.0, self.v + self.a_actual * self.dt)
        self.x += self.v * self.dt
        return self.x, self.v, self.a_actual
