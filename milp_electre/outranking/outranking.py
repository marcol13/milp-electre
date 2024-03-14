import numpy as np
from pulp import LpVariable, LpInteger, LpProblem, LpMinimize, LpStatus
from core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from core.types import RankingType
from collections import defaultdict

class Outranking:
    def __init__(self, credibility, scores):
        self.credibility = credibility.matrix
        self.size = self.credibility.shape[0]
        self.scores = scores
        self.problem = LpProblem("Maximize_support", LpMinimize)
        self.variables = {}

    def create_variable_matrix(self, name):
        return np.array([LpVariable(f"{name}_{i}_{k}", 0, 1, LpInteger) if i != k else 0 for i in range(self.size) for k in range(self.size)]).reshape((self.size, self.size))

    def solve(self, mode: RankingType ="complete"):
        pass

    def verbose(self):
        print("Status:", LpStatus[self.problem.status])
        print()

        print(self.problem.constraints)

        print()

        vars = np.array([x.name.split("_") + [x.varValue] for x in self.problem.variables()])
        rels = list(set(vars[:,0]))
        matrices = defaultdict(lambda: np.eye(self.size), {rel: np.eye(self.size) for rel in rels})

        for rel, i, j, value in vars:
            matrices[rel][int(i)][int(j)] = value

        for key in matrices.keys():
            print(f"Matrix {key}:")
            print(matrices[key])
            print()

        print(f"Objective function: {self.problem.objective}")

    @staticmethod
    def get_preference(i: int, j: int):
        if i > j:
            return PositivePreference
        elif j > i:
            return NegativePreference
        elif i == j == 1:
            return Indifference
        else:
            return Incomparible