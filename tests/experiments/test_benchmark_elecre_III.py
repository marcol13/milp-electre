import pytest
import string
import numpy as np

from experiments.benchmarks.electre_III import compare_electre3


class ValuedHelperProblem:
    def __init__(self):
        self.name = "test"
        self.alternatives = 4
        self.criteria = 3
        self.thresholds = {"indifference": 0.05, "preference": 0.15, "veto": 0.25}
        self.is_cost = [False, True, False]
        self.labels = list(string.ascii_lowercase[:4])
        self.data = np.array([[0, 1, 0], [1, 0, 1], [0.5, 0.5, 0.5], [0.75, 0.25, 0.75]], dtype=np.float64)

    def generate_weights(self, n: int):
        return [1 for _ in range(n)]
    
    def create_dict(self, data):
        assert len(data) == self.criteria
        return dict(zip(range(self.criteria), data))


class TestElectreIIIBenchmark:
    def test_benchmark_elecre_III_partial(self):
        settings = {
            "alternatives": 4,
            "criteria": 3,
            "thresholds": {"indifference": 0.05, "preference": 0.15, "veto": 0.25},
            "is_cost_threshold": 0.5,
            "mode": "partial",
            "all_results": False,
        }

        vp = ValuedHelperProblem()

        metrics = compare_electre3(vp, settings)

        assert False
