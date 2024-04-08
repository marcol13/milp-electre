from .outranking import Outranking
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from pulp import lpSum

class StochasticOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)
        self.stochastic_credibility = self.credibility
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
        stochastic_relations = [{"var": self.stochastic_credibility[PositivePreference], "rel": PositivePreference}, {"var": self.stochastic_credibility[NegativePreference], "rel": NegativePreference}, {"var": self.stochastic_credibility[Indifference], "rel": Indifference}, {"var": self.stochastic_credibility[Incomparible], "rel": Incomparible}]

        self.problem += lpSum([s_relation["var"][i][j] * p_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], p_relation["rel"]) for s_relation in stochastic_relations for p_relation in problem_relations for [i, j] in self.upper_matrix_ids])

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

        # self.verbose()

    def solve_complete(self):
        self.variables["p"] = self.create_variable_matrix("p")
        self.variables["z"] = self.create_variable_matrix("z")
        stochastic_relations = [{"var": self.stochastic_credibility[PositivePreference], "rel": PositivePreference}, {"var": self.stochastic_credibility[NegativePreference], "rel": NegativePreference}, {"var": self.stochastic_credibility[Indifference], "rel": Indifference}]

        self.problem += lpSum([s_relation["var"][i][j] * (self.variables["p"][i][j] - self.variables["z"][i][j]) * self.scores.get_distance(s_relation["rel"], PositivePreference) + (self.variables["p"][j][i] - self.variables["z"][i][j]) * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], NegativePreference) + self.variables["z"][i][j] * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], Indifference) for [i, j] in self.upper_matrix_ids for s_relation in stochastic_relations])
        
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.problem += self.variables["p"][i][j] + self.variables["p"][j][i] >= 1, f"Weak preference [{i}-{j}]"
                    self.problem += self.variables["z"][i][j] == self.variables["p"][i][j] + self.variables["p"][j][i] - 1, f"Incomparability [{i}-{j}]"

        for i, k, p in self.unique_permutations:
            self.problem += self.variables["p"][i][k] >= self.variables["p"][i][p] + self.variables["p"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

        self.problem.solve()

        # self.verbose()