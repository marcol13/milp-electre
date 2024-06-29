import numpy as np
import numpy.typing as npt

from experiments.helpers import get_relation, get_relation_score
from mcdalp.outranking.ranking import Ranking
from mcdalp.core.types import RankingMode

class Metrics:
    def __init__(self, r1: Ranking, r2: Ranking, mode: RankingMode = "partial"):
        assert r1.size == r2.size
        assert r1.rank_matrix.shape == r2.rank_matrix.shape
        assert r1.scores == r2.scores

        self.r1 = r1
        self.r2 = r2
        self.size = r1.size
        self.mode = mode

    def kendall_tau(self):
        distance = self.__calculate_kendall_distance(self.r1.rank_matrix, self.r2.rank_matrix)
        kendall_coef = 1 - 4 * distance / (self.size * (self.size - 1))
        return kendall_coef
    
    def rank_difference_measure(self):
        rank_diff = np.abs(self.r1.positions - self.r2.positions)

        return np.sum(rank_diff) / self.__max_rank_diff(self.size)

    def normalized_hit_ratio(self):
        common_leaders = len(np.intersect1d(self.r1.leaders, self.r2.leaders))
        all_leaders = len(np.union1d(self.r1.leaders, self.r2.leaders))

        return common_leaders / all_leaders
    
    def normalized_rank_difference(self):
        measure = 0
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                r1_pref = get_relation(self.r1.rank_matrix[i][j], self.r1.rank_matrix[j][i])
                r2_pref = get_relation(self.r2.rank_matrix[i][j], self.r2.rank_matrix[j][i])

                measure += self.r1.scores.score_matrix[r1_pref][r2_pref]

        return measure / (2 * self.size * (self.size - 1))
   
    def __calculate_kendall_distance(self, x: npt.ArrayLike, y: npt.ArrayLike):
        shape = x.shape
        assert shape == y.shape
        if shape == None:
            raise ValueError("Matrices must be an numpy 2d array")
        distance = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i == j:
                    continue
                rel_x = get_relation(x[i][j], x[j][i])
                rel_y = get_relation(y[i][j], y[j][i])
                distance += get_relation_score(rel_x, rel_y)
        
        distance *= 1/8
        return distance
    
    def __max_rank_diff(self, size: int):
        if size % 2 == 0:
            return (size // 2) * size
        else:
            return -(size // -2) * (size - 1)
        
    def make_measurement(self):
        if self.mode == "partial":
            return {
                "normalized_rank_difference": self.normalized_rank_difference(),
                "normalized_hit_ratio": self.normalized_hit_ratio(),
                "rank_difference": self.rank_difference_measure(),
            }
        else:
            return {
                "kendall_tau": self.kendall_tau(),
                "normalized_hit_ratio": self.normalized_hit_ratio(),
                "rank_difference": self.rank_difference_measure(),
            }