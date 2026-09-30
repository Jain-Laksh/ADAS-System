from __future__ import annotations
import numpy as np

from adas.ml.conformal import calibrate_threshold, compute_far


class TestConformal:
    def test_threshold_monotonicity(self) -> None:
        scores = np.linspace(0.0, 1.0, 1000)
        labels = np.zeros(1000)
        
        t_01 = calibrate_threshold(scores, labels, 0.01)
        t_05 = calibrate_threshold(scores, labels, 0.05)
        
        assert t_01 > t_05

    def test_coverage_guarantee(self) -> None:
        rng = np.random.RandomState(42)
        scores = rng.rand(2000)
        labels = np.zeros(2000)
        
        alpha = 0.05
        thresh = calibrate_threshold(scores, labels, alpha)
        
        far_metrics = compute_far(scores, labels, thresh)
        sample_far = far_metrics["sample_far"]
        
        # FAR should be close to alpha
        assert sample_far <= alpha + 0.01

    def test_edge_cases(self) -> None:
        scores = np.zeros(100)
        labels = np.ones(100)
        
        # All ones (no negative samples)
        thresh = calibrate_threshold(scores, labels, 0.01)
        assert thresh == 1.0
        
        far = compute_far(scores, labels, thresh)
        assert far["sample_far"] == 0.0
