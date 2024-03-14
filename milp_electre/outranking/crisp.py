from .outranking import Outranking
from core.const import RELATIONS
from core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from pulp import lpSum
from itertools import permutations
import numpy as np

class CrispOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)
        self.solve()

    def solve(self, mode="complete"):
        pass

    def solve_partial(self):
        self.variables["outranking"] = self.create_variable_matrix("outranking")
        self.variables["pp"] = self.create_variable_matrix("pp")
        self.variables["pn"] = self.create_variable_matrix("pn")
        self.variables["i"] = self.create_variable_matrix("i")
        self.variables["r"] = self.create_variable_matrix("r")
        problem_relations = [{"var": self.variables["pp"], "rel": PositivePreference}, {"var": self.variables["pn"], "rel": NegativePreference}, {"var": self.variables["i"], "rel": Indifference}, {"var": self.variables["r"], "rel": Incomparible}]

        upper_matrix_ids = np.triu_indices(self.size, 1)
        upper_matrix_ids = np.column_stack(upper_matrix_ids)
        self.problem += lpSum([relation["var"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), relation["rel"]) for [i, j] in upper_matrix_ids for relation in problem_relations])

        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.problem += self.variables["outranking"][i][j] - self.variables["outranking"][j][i] <= self.variables["pp"][i][j], f"Positive preference [{i}-{j}]"
                    self.problem += self.variables["outranking"][j][i] - self.variables["outranking"][i][j] <= self.variables["pn"][i][j], f"Negative preference [{i}-{j}]"
                    self.problem += self.variables["outranking"][i][j] + self.variables["outranking"][j][i] - 1 <= self.variables["i"][i][j], f"Indifference [{i}-{j}]"
                    self.problem += 1 - self.variables["outranking"][i][j] - self.variables["outranking"][j][i] <= self.variables["r"][i][j], f"Incomparability [{i}-{j}]"
                    self.problem += self.variables["pp"][i][j] + self.variables["pn"][i][j] + self.variables["r"][i][j] + self.variables["i"][i][j] == 1, f"Only one relation [{i}, {j}]"
        
        unique_permutations = list(permutations(range(self.size), 3))
        for i, k, p in unique_permutations:
            self.problem += self.variables["outranking"][i][k] >= self.variables["outranking"][i][p] + self.variables["outranking"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

        self.problem.solve()

        self.verbose()

    def solve_complete(self):
        pass