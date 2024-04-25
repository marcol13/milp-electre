import numpy as np
from pulp import LpVariable, LpInteger, LpProblem, LpMinimize, LpStatus
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.types import RankingModeType
from ..core.const import RankingMode
from collections import defaultdict
from itertools import permutations
from abc import ABC, abstractmethod

class Outranking(ABC):
    def __init__(self, credibility, scores):
        self.credibility = credibility.matrix
        self.size = credibility.get_size()
        self.scores = scores
        self.problem = LpProblem("Maximize_support", LpMinimize)
        self.variables = {}

        self.upper_matrix_ids = np.triu_indices(self.size, 1)
        self.upper_matrix_ids = np.column_stack(self.upper_matrix_ids)

        self.unique_permutations = list(permutations(range(self.size), 3))

    def create_variable_matrix(self, name):
        return np.array([LpVariable(f"{name}_{i}_{k}", 0, 1, LpInteger) if i != k else 0 for i in range(self.size) for k in range(self.size)]).reshape((self.size, self.size))

    def solve(self, mode: RankingModeType):
        if mode == "partial":
            self.solve_partial()
        elif mode == "complete":
            self.solve_complete()
        else:
            raise ValueError("Invalid mode")

    @abstractmethod
    def solve_partial(self):
        pass

    @abstractmethod
    def solve_complete(self):
        pass

    def create_variables(self, relations: list[str]) -> dict:
        variables = dict()
        for relation in relations:
            variables[relation] = self.create_variable_matrix(relation)
        return variables
    
    def add_contraints(self, mode: RankingModeType, problem, variables, size, unique_permutations):
        if mode == RankingMode.PARTIAL:
            for i in range(size):
                for j in range(size):
                    if i != j:
                        problem += variables["outranking"][i][j] - variables["outranking"][j][i] <= variables["pp"][i][j], f"Positive preference [{i}-{j}]"
                        problem += variables["outranking"][j][i] - variables["outranking"][i][j] <= variables["pn"][i][j], f"Negative preference [{i}-{j}]"
                        problem += variables["outranking"][i][j] + variables["outranking"][j][i] - 1 <= variables["i"][i][j], f"Indifference [{i}-{j}]"
                        problem += 1 - variables["outranking"][i][j] - variables["outranking"][j][i] <= variables["r"][i][j], f"Incomparability [{i}-{j}]"
                        problem += variables["pp"][i][j] + variables["pn"][i][j] + variables["r"][i][j] + variables["i"][i][j] == 1, f"Only one relation [{i}, {j}]"

            for i, k, p in unique_permutations:
                problem += variables["outranking"][i][k] >= variables["outranking"][i][p] + variables["outranking"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

            return problem
        elif mode == RankingMode.COMPLETE:
            for i in range(size):
                for j in range(size):
                    if i != j:
                        problem += variables["p"][i][j] + variables["p"][j][i] >= 1, f"Weak preference [{i}-{j}]"
                        problem += variables["z"][i][j] == variables["p"][i][j] + variables["p"][j][i] - 1, f"Incomparability [{i}-{j}]"

            for i, k, p in unique_permutations:
                problem += variables["p"][i][k] >= variables["p"][i][p] + variables["p"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

            return problem
        else:
            raise ValueError("Invalid mode")

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

    def get_outranking(self, relation_array: str):
        variables = np.array([x.name.split("_") + [x.varValue] for x in self.problem.variables()])
        variables = variables[variables[:, 0] == relation_array]
        outranking = np.eye(self.size)
        for _, i, j, value in variables:
            outranking[int(i)][int(j)] = value
        return outranking

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