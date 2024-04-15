from .outranking import Outranking
from ..core.const import RELATIONS
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from pulp import lpSum

class CrispOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)
        self.solve()

    def solve_partial(self):
        self.variables["outranking"] = self.create_variable_matrix("outranking")
        self.variables["pp"] = self.create_variable_matrix("pp")
        self.variables["pn"] = self.create_variable_matrix("pn")
        self.variables["i"] = self.create_variable_matrix("i")
        self.variables["r"] = self.create_variable_matrix("r")
        problem_relations = [{"var": self.variables["pp"], "rel": PositivePreference}, {"var": self.variables["pn"], "rel": NegativePreference}, {"var": self.variables["i"], "rel": Indifference}, {"var": self.variables["r"], "rel": Incomparible}]

        self.problem += lpSum([relation["var"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), relation["rel"]) for [i, j] in self.upper_matrix_ids for relation in problem_relations])

        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.problem += self.variables["outranking"][i][j] - self.variables["outranking"][j][i] <= self.variables["pp"][i][j], f"Positive preference [{i}-{j}]"
                    self.problem += self.variables["outranking"][j][i] - self.variables["outranking"][i][j] <= self.variables["pn"][i][j], f"Negative preference [{i}-{j}]"
                    self.problem += self.variables["outranking"][i][j] + self.variables["outranking"][j][i] - 1 <= self.variables["i"][i][j], f"Indifference [{i}-{j}]"
                    self.problem += 1 - self.variables["outranking"][i][j] - self.variables["outranking"][j][i] <= self.variables["r"][i][j], f"Incomparability [{i}-{j}]"
                    self.problem += self.variables["pp"][i][j] + self.variables["pn"][i][j] + self.variables["r"][i][j] + self.variables["i"][i][j] == 1, f"Only one relation [{i}, {j}]"
        
        for i, k, p in self.unique_permutations:
            self.problem += self.variables["outranking"][i][k] >= self.variables["outranking"][i][p] + self.variables["outranking"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

        self.problem.solve()

    def solve_complete(self):
        self.variables["p"] = self.create_variable_matrix("p")
        self.variables["z"] = self.create_variable_matrix("z")

        self.problem += lpSum([self.variables["p"][i][j] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), PositivePreference) + self.variables["p"][j][i] * self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), NegativePreference) + self.variables["z"][i][j] * (self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), Indifference) - self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), PositivePreference) - self.scores.get_distance(self.get_preference(self.credibility[i][j], self.credibility[j][i]), NegativePreference)) for [i, j] in self.upper_matrix_ids])  

        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.problem += self.variables["p"][i][j] + self.variables["p"][j][i] >= 1, f"Weak preference [{i}-{j}]"
                    self.problem += self.variables["z"][i][j] == self.variables["p"][i][j] + self.variables["p"][j][i] - 1, f"Incomparability [{i}-{j}]"

        for i, k, p in self.unique_permutations:
            self.problem += self.variables["p"][i][k] >= self.variables["p"][i][p] + self.variables["p"][p][k] - 1.5, f"Transition [{i}-{k}-{p}]"

        self.problem.solve()
        