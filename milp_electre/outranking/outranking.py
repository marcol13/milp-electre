import numpy as np
from pulp import LpVariable, LpInteger, LpProblem, LpMinimize
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.types import RankingType

class Outranking:
    def __init__(self, credibility, scores):
        self.size = credibility.shape[0]
        self.scores = scores
        self.problem = LpProblem("Maximize support", LpMinimize)
        self.variables = {}

    def create_variable_matrix(self, name):
        return np.array([LpVariable(f"{name}_{i}_{k}", 0, 1, LpInteger) if i != k else 0 for i in range(self.size) for k in range(self.size)]).reshape(self.size)

    def solve(self, mode: RankingType ="complete"):
        pass

    def get_preference(i: int, j: int):
        if i > j:
            return PositivePreference
        elif j > i:
            return NegativePreference
        elif i == j == 1:
            return Indifference
        else:
            return Incomparible