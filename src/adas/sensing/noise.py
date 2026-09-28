from __future__ import annotations
import numpy as np


def add_noise(
    gap: np.ndarray,
    v_rel: np.ndarray,
    gap_sigma: float = 0.3,
    vrel_sigma: float = 0.15,
    dropout_rate: float = 0.02,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.RandomState()
    n = len(gap)
    noisy_gap = gap + rng.randn(n) * gap_sigma
    noisy_vrel = v_rel + rng.randn(n) * vrel_sigma
    valid = rng.rand(n) > dropout_rate
    return noisy_gap, noisy_vrel, valid
