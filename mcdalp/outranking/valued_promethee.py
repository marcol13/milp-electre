from .outranking import Outranking
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.const import RankingMode
from pulp import lpSum

class ValuedPrometheeOutranking(Outranking):
    def __init__(self, credibility, scores):
        super().__init__(credibility, scores)
        self.valued_credibility = self.credibility

    def solve_partial(self):
        self.variables = self.create_variables(["outranking", "pp", "pn", "i", "r"])
        reversed_matrix = self.valued_credibility.T
        problem_relations = [{"var": self.variables["pp"], "rel": PositivePreference}, {"var": self.variables["pn"], "rel": NegativePreference}, {"var": self.variables["i"], "rel": Indifference}, {"var": self.variables["r"], "rel": Incomparible}]
        valued_relations = [{"var": self.valued_credibility, "rel": PositivePreference}, {"var": reversed_matrix, "rel": NegativePreference}, {"var": 1 - self.valued_credibility - reversed_matrix, "rel": Indifference}]

        self.problem += lpSum([s_relation["var"][i][j] * p_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], p_relation["rel"]) for s_relation in valued_relations for p_relation in problem_relations for [i, j] in self.upper_matrix_ids])
        self.problem = self.add_contraints(RankingMode.PARTIAL, self.problem, self.variables, self.size, self.unique_permutations)

        self.problem.solve()

    def solve_complete(self):
        self.variables = self.create_variables(["p", "z"])
        reversed_matrix = self.valued_credibility.T
        valued_relations = [{"var": self.valued_credibility, "rel": PositivePreference}, {"var": reversed_matrix, "rel": NegativePreference}, {"var": 1 - self.valued_credibility - reversed_matrix, "rel": Indifference}]

        self.problem += lpSum([s_relation["var"][i][j] * (self.variables["p"][i][j] - self.variables["z"][i][j]) * self.scores.get_distance(s_relation["rel"], PositivePreference) + (self.variables["p"][j][i] - self.variables["z"][i][j]) * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], NegativePreference) + self.variables["z"][i][j] * s_relation["var"][i][j] * self.scores.get_distance(s_relation["rel"], Indifference) for [i, j] in self.upper_matrix_ids for s_relation in valued_relations])
        self.problem = self.add_contraints(RankingMode.COMPLETE, self.problem, self.variables, self.size, self.unique_permutations)

        self.problem.solve()
