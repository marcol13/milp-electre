import pytest

from mcdalp.outranking.ranking import Ranking
from experiments.metrics import Metrics
from .data import *

class TestKendall:
    ranking1 = Ranking("crisp", outranking1)
    ranking2 = Ranking("crisp", outranking2)
    ranking3 = Ranking("crisp", outranking3)
    ranking4 = Ranking("crisp", outranking4)

    def test_same_matrices(self):
        metrics = Metrics(self.ranking1, self.ranking1)
        assert metrics.kendall_tau() == 1

    def test_opposite_matrices(self):
        metrics = Metrics(self.ranking1, self.ranking2)
        assert metrics.kendall_tau() == -1

    def test_indifferent_matrix(self):
        metrics = Metrics(self.ranking1, self.ranking3)
        assert metrics.kendall_tau() == 0

    def test_two_ranks_matrices(self):
        metrics = Metrics(self.ranking1, self.ranking4)
        assert metrics.kendall_tau() == 0.5

class TestNHR:
    ranking1 = Ranking("crisp", outranking1)
    ranking2 = Ranking("crisp", outranking2)
    ranking3 = Ranking("crisp", outranking3)
    ranking4 = Ranking("crisp", outranking4)
    ranking5 = Ranking("crisp", outranking6)

    def test_no_common_leaders(self):
        metrics = Metrics(self.ranking1, self.ranking2)
        assert metrics.normalized_hit_ratio() == 0

    def test_all_common_leaders(self):
        metrics = Metrics(self.ranking1, self.ranking1)
        assert metrics.normalized_hit_ratio() == 1

    def test_part_common_leaders(self):
        metrics = Metrics(self.ranking1, self.ranking3)
        assert metrics.normalized_hit_ratio() == 0.25

    def test_with_incomparibility(self):
        metrics = Metrics(self.ranking4, self.ranking5)
        assert metrics.normalized_hit_ratio() == 0.5

class TestRDM:
    ranking1 = Ranking("crisp", outranking1)
    ranking2 = Ranking("crisp", outranking2)
    ranking3 = Ranking("crisp", outranking7)
    ranking4 = Ranking("crisp", outranking7.T)
    ranking5 = Ranking("crisp", outranking5)
    ranking6 = Ranking("crisp", outranking8)
    ranking7 = Ranking("crisp", outranking3)

    def test_same_matrices(self):
        metrics = Metrics(self.ranking1, self.ranking1)
        assert metrics.rank_difference_measure() == 0

    def test_opposite_matrices_even(self):
        metrics = Metrics(self.ranking1, self.ranking2)
        assert metrics.rank_difference_measure() == 1

    def test_opposite_matrices_odd(self):
        metrics = Metrics(self.ranking3, self.ranking4)
        assert metrics.rank_difference_measure() == 1

    def test_indifferent_matrix(self):
        metrics = Metrics(self.ranking1, self.ranking7)
        assert metrics.rank_difference_measure() == 0.75

    def test_two_ranks_matrices_even(self):
        metrics = Metrics(self.ranking1, self.ranking5)
        assert metrics.rank_difference_measure() == 5/8
        
    def test_two_ranks_matrices_odd(self):
        metrics = Metrics(self.ranking3, self.ranking6)
        assert metrics.rank_difference_measure() == 8/12

class TestNRD:
    ranking1 = Ranking("crisp", outranking1)
    ranking2 = Ranking("crisp", outranking2)
    ranking3 = Ranking("crisp", outranking3)
    ranking4 = Ranking("crisp", outranking5)

    def test_same_matrices(self):
        metrics = Metrics(self.ranking1, self.ranking1)
        assert metrics.normalized_rank_difference() == 0

    def test_opposite_matrices(self):
        metrics = Metrics(self.ranking1, self.ranking2)
        assert metrics.normalized_rank_difference() == 1

    def test_indifferent_matrix(self):
        metrics = Metrics(self.ranking1, self.ranking3)
        assert metrics.normalized_rank_difference() == 0.5

    def test_incomparible(self):
        metrics = Metrics(self.ranking1, self.ranking4)
        assert metrics.normalized_rank_difference() == 26/48

    

