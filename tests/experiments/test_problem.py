import pytest

from experiments.benchmarks.problem import Problem

class TestProblem:
    def test_generate_weights(self):
        n = 5
        thresholds = {
            "indifference": 0.05,
            "preference": 0.15,
            "veto": 0.25
        }
        is_cost = [True, False, True, False, True]
        labels = ["a", "b", "c", "d", "e"]
        problem = Problem("test", 5, 5, thresholds, is_cost, labels)
        weights = problem.generate_weights(n)
        assert len(weights) == n
        assert sum(weights) == 1
        assert all([0 <= w <= 1 for w in weights])